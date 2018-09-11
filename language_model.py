#!/usr/bin/env python3
import os
import contextlib
import logging
import pickle
import random
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import texar as tx

from options import *
if 'captioning' not in locals():
    captioning = False
from model import Seq2Seq, Seq2SeqAttention, Seq2SeqFastAttention
from utils import to_onehot, get_grad_norm
from evaluate import evaluate_model_
from tensorboardX import SummaryWriter
from logger import LossLogger

from nltk.translate.bleu_score import sentence_bleu

if captioning:
    from data_loader import get_ann_loader, get_img_loader
    from caption_vocab import Vocabulary
    from caption_model import EncoderCNN, DecoderRNN

if hasattr(train_config, 'seed') and train_config.seed is not None:
    seed = train_config.seed
    random.seed(seed)
    tf.set_random_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def run_model(model, encoder, batch, target_vocab, teach_rate, device, verbose=False):
    target_vocab_size = target_vocab.size
    eos_id = target_vocab.eos_token_id
    if captioning:
        images, tgt_sents, lengths = batch
        ret = {
            'images': images,
            'tgt_sents': tgt_sents,
        }
    else:
        src_sents, tgt_sents = batch['source_text_ids'], batch['target_text_ids']
        src_sents = torch.tensor(src_sents, dtype=torch.long, device=device)
        tgt_sents = torch.tensor(tgt_sents, dtype=torch.long, device=device)
        ret = {
            'src_sents': src_sents,
            'tgt_sents': tgt_sents,
        }
    batch_size = tgt_sents.shape[0]

    if train_config.enable_cross_entropy:
        ret['ce'] = {}

        if captioning:
            src = encoder(images)
            src = src.detach()
        else:
            src = src_sents

        if train_config.enable_xe:
            logits_xe = model(src, tgt_sents[:, :-1])
            tgt_sents_ = tgt_sents[:, 1:]
            flatten_logits_xe = logits_xe.contiguous().view(-1, logits_xe.shape[-1])
            flatten_tgt_sents_ = tgt_sents_.contiguous().view(-1)
            xel = criterion_cross_entropy(flatten_logits_xe, flatten_tgt_sents_)

            ret['ce']['xe'] = {
                'logits': logits_xe,
                'loss': xel,
            }

        else:
            xel = 0.

        if train_config.enable_pg:
            ret['ce']['pg'] = {}

            ids_sample, logprobs_sample = model(
                src,
                tgt_sents[:, :-1],
                max_decode_length=train_config.max_decode_length,
                beam=-1)
            logits_greedy = model(
                src,
                tgt_sents[:, :-1],
                max_decode_length=train_config.max_decode_length,
                beam=1)
            logprobs_greedy, ids_greedy = logits_greedy.max(-1)

            def seq_tolist(ids):
                a = ids.tolist()
                try:
                    return a[:a.index(eos_id)]
                except ValueError:
                    return a
            def tolist(ids):
                return list(map(seq_tolist, ids.cpu().numpy()))

            seq_sample = tolist(ids_sample)
            seq_greedy = tolist(ids_greedy)
            seq_target = tolist(tgt_sents[:, 1:])

            pgl = []
            for seq_s, seq_g, seq_t, logprob_sample \
            in zip(seq_sample, seq_greedy, seq_target, logprobs_sample):
                reward = ( sentence_bleu([seq_t], seq_s)
                          -sentence_bleu([seq_t], seq_g))
                pgl.append(reward * -logprob_sample[:len(seq_sample)].sum())
            pgl = torch.stack(pgl).mean()

        else:
            pgl = 0.

        cel = train_config.xe_w * xel + train_config.pg_w * pgl

        ret['ce'].update({
            'loss': cel,
        })

    if train_config.enable_bleu:
        tgt_sents_onehot = to_onehot(tgt_sents, target_vocab_size, dtype=torch.float)
        ret['tgt_sents_onehot'] = tgt_sents_onehot

        gamma = train_config.gamma
        if gamma == 0:
            beam = 1
        else:
            beam = 0
        max_decode_length = train_config.max_decode_length
        if max_decode_length is None:
            max_decode_length = tgt_sents.shape[1] - 1
        if random.random() < train_config.fix_teach_gap:
            n = train_config.teach_gap + train_config.teach_cont
            r = random.randrange(n)
            teach_flags = [not (i % n < train_config.teach_gap)
                           for i in range(r, r + max_decode_length)]
            #logging.info("teach flags: {}".format("".join(str(int(flag)) for flag in teach_flags)))
        else:
            teach_flags = [random.random() < teach_rate
                           for i in range(max_decode_length)]
        teach_flags = [True] + teach_flags

        if captioning:
            src = encoder(images)
            src = src.detach()
        else:
            src = src_sents
        logits_mb = model(
            src,
            tgt_sents[:, :-1],
            max_decode_length=train_config.max_decode_length,
            beam=beam,
            teach_flags=teach_flags)

        probs = F.softmax(logits_mb, dim=-1)
        probs = torch.cat([tgt_sents_onehot[:, :1], probs], dim=1)

        if hasattr(train_config, "teach_X") and not train_config.teach_X:
            X = probs
        else:
            X = []
            for t in range(probs.shape[1]):
                X.append((tgt_sents_onehot if teach_flags[t] else probs)[:, t])
            X[0] = torch.tensor(X[0], requires_grad=True)
            X = torch.stack(X, dim=1)
        gen_probs, gen_ids = X.max(-1)
        Y = tgt_sents_onehot

        def length_mask(X):
            l = X.shape[1]
            mask = [torch.ones(X.shape[0], device=device)] * 2
            for t in range(l-1):
                mask.append(mask[-1] * (1 - X[:, t, eos_id]))
            mask = torch.stack(mask, dim=1)
            lenX = torch.sum(mask, dim=1) - 1
            return mask, lenX
        maskY, lenY = length_mask(Y)
        if train_config.soft_length_mask:
            maskX, lenX = length_mask(X)
        else:
            assert X.shape == Y.shape, "X.shape={}, Y.shape={}".format(X.shape, Y.shape)
            maskX, lenX = maskY, lenY

        mbl, mbls_ = criterion_bleu(
            tgt_sents, X, lenY, lenX, maskY, maskX,
            min_fn=train_config.min_fn,
            min_c=train_config.min_c,
            enable_prec=train_config.enable_prec,
            enable_recall=train_config.enable_recall,
            recall_w=train_config.recall_w,
            device=device, verbose=verbose)

        ret['mb'] = {
            'logits': logits_mb,
            'probs': probs,
            'gen_probs': gen_probs,
            'gen_ids': gen_ids,
            'loss': mbl,
            'mbls_': mbls_,
            'X': X,
            'Y': Y,
        }

    bleu_w = train_config.bleu_w
    if bleu_w == 0.:
        loss = cel
    elif bleu_w == 1.:
        loss = mbl
    else:
        loss = (1. - bleu_w) * cel + bleu_w * mbl

    ret['loss'] = loss

    return ret

if __name__ == '__main__':
    logging.root.handlers = []
    print('logging file: {}'.format(logging_file))
    logging.basicConfig(
        level=logging.INFO,
        filename=logging_file,
        filemode='a',
        format='%(asctime)s | %(levelname)s | %(message)s',
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    writer = SummaryWriter(os.path.join(logdir, "log"))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('device: {}'.format(device))

    if captioning:
        # Load vocabulary wrapper
        with open(caption_config.vocab_path, 'rb') as f:
            target_vocab = pickle.load(f)
        
        # Build data loader
        train_data_loader = get_ann_loader(
            caption_config.train_image_dir,
            caption_config.train_caption_path,
            target_vocab,
            caption_config.train_batch_size,
            shuffle=True,
            num_workers=caption_config.num_workers,
            device=device)
        val_data_loader = get_img_loader(
            caption_config.val_image_dir,
            caption_config.val_caption_path,
            target_vocab,
            caption_config.val_batch_size,
            shuffle=False,
            num_workers=caption_config.num_workers,
            device=device)
        test_data_loader = val_data_loader
        data_loaders = {
            'train': train_data_loader,
            'val': val_data_loader,
            'test': test_data_loader,
        }

        # Build the models
        encoder = EncoderCNN().to(device)
        decoder = DecoderRNN(encoder.outdim,
                             caption_config.embed_size,
                             caption_config.hidden_size,
                             len(target_vocab),
                             caption_config.num_layers).to(device)
        model = decoder

    else:
        training_data = tx.data.PairedTextData(hparams=data_config.training_data_hparams)
        valid_data = tx.data.PairedTextData(hparams=data_config.valid_data_hparams)
        test_data = tx.data.PairedTextData(hparams=data_config.test_data_hparams)
        data_iterator = tx.data.FeedableDataIterator({
            'train': training_data,
            'val': valid_data,
            'test': test_data,
        })
        data_batch = data_iterator.get_next()

        source_vocab = training_data.source_vocab
        target_vocab = training_data.target_vocab

        model = Seq2SeqAttention(
            src_emb_dim=model_config.embdim,
            trg_emb_dim=model_config.embdim,
            src_vocab_size=source_vocab.size,
            trg_vocab_size=target_vocab.size,
            src_hidden_dim=model_config.dim,
            trg_hidden_dim=model_config.dim,
            ctx_hidden_dim=model_config.dim,
            attention_mode='dot',
            batch_size=data_config.training_data_hparams['batch_size'],
            bidirectional=model_config.bidir,
            pad_token_src=int(source_vocab.pad_token_id),
            pad_token_trg=int(target_vocab.pad_token_id),
            nlayers=model_config.nlayerssrc,
            nlayers_trg=model_config.nlayerstgt,
            dropout=train_config.dropout
        ).to(device)

    criterion_cross_entropy = nn.CrossEntropyLoss(ignore_index=int(target_vocab.pad_token_id))
    criterion_bleu = mBLEU(train_config.max_order)

    step = 0

    def _load_model(epoch, step=None):
        name = "model.epoch{}".format(epoch)
        if step is not None:
            name += ".{}".format(step)
        ckpt = os.path.join(logdir, name)
        logging.info('loading model from {} ...'.format(ckpt))
        model.load_state_dict(torch.load(ckpt))
        #if captioning:
        #    ckpt = caption_config.encoder_model_path
        #    logging.info('loading encoder from {} ...'.format(ckpt))
        #    encoder.load_state_dict(torch.load(ckpt))

    def _save_model(epoch, step=None):
        name = 'model.epoch{}'.format(epoch)
        if step is not None:
            name += '.{}'.format(step)
        ckpt = os.path.join(logdir, name)
        logging.info('saving model into {} ...'.format(ckpt))
        torch.save(model.state_dict(), ckpt)

    ids_to_words = target_vocab.ids_to_words if captioning else \
        (lambda ids: sess.run(target_vocab.map_ids_to_tokens(ids)))

    def _train_epoch(sess, model, optimizer, pretrain, losses, verbose=verbose_config.verbose):
        global teach_rate
        global step
        if captioning:
            data_loader = data_loaders['train']
            encoder.eval()
        else:
            data_iterator.restart_dataset(sess, 'train')
            feed_dict = {data_iterator.handle: data_iterator.get_handle(sess, 'train')}
        model.train()

        if not captioning:
            def _get_data_loader():
                while True:
                    try:
                        yield sess.run(data_batch, feed_dict=feed_dict)
                    except tf.errors.OutOfRangeError:
                        break
            data_loader = _get_data_loader()

        for batch_i, batch in enumerate(data_loader):
            if batch_i >= train_config.train_batches:
                break

            sample_verbose = verbose and (step + 1) % verbose_config.steps_sample == 0
            if captioning:
                images, tgt_ids, lengths = batch
                res = run_model(model, encoder, batch, target_vocab, teach_rate=teach_rate,
                                device=device, verbose=sample_verbose)
            else:
                tgt_ids = batch['target_text_ids']
                res = run_model(model, None, batch, target_vocab, teach_rate=teach_rate,
                                device=device, verbose=sample_verbose)
            batch_size = tgt_ids.shape[0]
            if train_config.enable_cross_entropy:
                cel = res['ce']['loss']
                cel_ = cel.cpu().data.numpy()
            else:
                cel_ = -1.
            if train_config.enable_bleu:
                probs = res['mb']['X']
                if sample_verbose and verbose_config.probs_verbose:
                    probs.retain_grad()
                gen_ids = res['mb']['gen_ids']
                gen_probs = res['mb']['gen_probs']
                mbl = res['mb']['loss']
                mbl_ = mbl.cpu().data.numpy()
            else:
                mbl_ = -1.
            loss = res['loss']
            if pretrain:
                if sample_verbose:
                    logging.info('pretraining')
                loss = cel
            loss_ = loss.cpu().data.numpy()

            if train_config.enable_bleu and sample_verbose and verbose_config.probs_verbose:
                mbls_ = res['mb']['mbls_']
                grad_ = []
                for order in range(1, criterion_bleu.max_order + 1):
                    optimizer.zero_grad()
                    mbls_[order-1].backward(retain_graph=True)
                    grad_.append(probs.grad)
                grad_ = torch.stack(grad_, dim=1)

            optimizer.zero_grad()
            loss.backward()
            if train_config.clip_grad_norm is None:
                grad_norm = get_grad_norm(model.parameters())
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_config.clip_grad_norm)

            if train_config.enable_bleu and sample_verbose:
                samples = min(verbose_config.samples, batch_size)
                gen_words, tgt_words = map(ids_to_words, (gen_ids, tgt_ids))
                if verbose_config.probs_verbose:
                    gen_grads = torch.gather(probs.grad, -1, gen_ids.unsqueeze(-1)).squeeze(-1)
                    max_grads, max_ids = probs.grad.min(-1)
                    max_probs = torch.gather(probs, -1, max_ids.unsqueeze(-1)).squeeze(-1)
                    max_words = ids_to_words(max_ids)
                    max_grad_, max_id_ = grad_.min(-1)
                    max_word_ = ids_to_words(max_id_)
                for sample_i, (gen_sent, tgt_sent) in enumerate(zip(gen_words, tgt_words)):
                    if sample_i >= samples:
                        break
                    l = list(tgt_sent).index(target_vocab.eos_token.encode()) + 1
                    logging.info('tgt: {}'.format(b' '.join(tgt_sent[:l]).decode()))
                    logging.info('gen: {}'.format(b' '.join(gen_sent[:l]).decode()))
                    if verbose_config.probs_verbose:
                        logging.info('max: {}'.format(b' '.join(max_words[sample_i][:l]).decode()))
                        logging.info('gen probs:\n{}'.format(gen_probs[sample_i][:l]))
                        logging.info('gen grads:\n{}'.format(gen_grads[sample_i][:l]))
                        logging.info('max probs:\n{}'.format(max_probs[sample_i][:l]))
                        logging.info('max grads:\n{}'.format(max_grads[sample_i][:l]))
                        for order in range(1, criterion_bleu.max_order + 1):
                            logging.info('{}-gram max: {}'.format(order, b' '.join(max_word_[sample_i][order-1][:l]).decode()))
                            logging.info('{}-gram max grads:\n{}'.format(order, max_grad_[sample_i][order-1][:l]))
            losses.append([loss_, cel_, mbl_, grad_norm])
            writer.add_scalar('train/loss', loss_, step)
            writer.add_scalar('train/cel', cel_, step)
            writer.add_scalar('train/mbl', mbl_, step)
            writer.add_scalar('train/grad_norm', grad_norm, step)
            step += 1
            if step % verbose_config.steps_loss == 0:
                logging.info('step: {}\tloss: {:.3f}\tcel: {:.3f}\tmbl: {:.3f}\tgrad_norm: {:.3f}'.format(
                    step, loss_, cel_, mbl_, grad_norm))

            optimizer.step()

            if step % verbose_config.steps_eval == 0:
                _eval_on_dev_set()
                if captioning:
                    encoder.eval()
                model.train()
                #losses.plot(os.path.join(logdir, 'train_losses'))

            if train_config.checkpoints and step % verbose_config.steps_ckpt == 0:
                _save_model(epoch, step)

            if train_config.enable_bleu and step % train_config.teach_rate_anneal_steps == 0:
                teach_rate *= train_config.teach_rate_anneal
                logging.info("teach rate: {}".format(teach_rate))

    def _test_decode(sess, model, mode, out_path, losses, device, verbose=False):
        model.eval()
        if captioning:
            encoder.eval()
            data_loader = data_loaders[mode]
            bleu = evaluate_model_(
                model, encoder, sess, None, data_loader, target_vocab, ids_to_words,
                verbose_config.eval_max_decode_length, verbose_config.eval_batches,
                writer, step, logdir, verbose_config.eval_print_samples)
        else:
            data_iterator.restart_dataset(sess, mode)
            feed_dict = {data_iterator.handle: data_iterator.get_handle(sess, mode)}
            bleu = evaluate_model_(
                model, None, sess, feed_dict, data_batch, target_vocab, ids_to_words,
                verbose_config.eval_max_decode_length, verbose_config.eval_batches,
                writer, step, logdir, verbose_config.eval_print_samples)
        bleu *= 100
        logging.info("epoch #{} BLEU: {:.6f}".format(epoch, bleu))
        losses.append((bleu,))
        writer.add_scalar('{}/BLEU'.format(mode), bleu, step)

    def _eval_on_dev_set(mode='val'):
        logging.info('evaluating on {} dataset...'.format(mode))
        _test_decode(
            sess,
            model,
            mode,
            os.path.join(logdir, '{}.epoch{}'.format(mode, epoch)),
            eval_losses,
            device)

    if not captioning:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    with (tf.Session(config=sess_config) if not captioning else contextlib.suppress()) as sess:
        logging.info('running_mode: {}'.format(args.running_mode))

        if not captioning:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.tables_initializer())

        if train_config.start_epoch is None:
            ckpts = list(filter(lambda s: s.startswith("model.epoch"), os.listdir(logdir)))
            if len(ckpts) == 0:
                epoch = 0
            else:
                def get_epoch_and_step(s):
                    s = s[len('model.epoch'):]
                    s = s.split('.')
                    return (int(s[0]), int(s[1]) if len(s) >= 2 else 0)
                epoch, step = max(map(get_epoch_and_step, ckpts))
                _load_model(epoch, step)
        else:
            epoch = train_config.start_epoch
            _load_model(epoch)

        if args.running_mode == 'train':
            if train_config.enable_bleu:
                teach_rate = train_config.initial_teach_rate
                if teach_rate is None:
                    teach_rate = float(input('initial_teach_rate = '))
                logging.info("teach rate: {}".format(teach_rate))
            else:
                teach_rate = None

            optimizer = optim.Adam(model.parameters(), lr=train_config.lr)

            train_losses = LossLogger(("loss", "cel", "mbl", "grad_norm"), os.path.join(logdir, "train_loss"))
            eval_losses = LossLogger(("bleu",), os.path.join(logdir, "eval_loss"))

            _eval_on_dev_set()

            while epoch < train_config.max_epochs:
                logging.info('training epoch #{}:'.format(epoch))
                _train_epoch(sess, model, optimizer, epoch < train_config.pretrain, train_losses)
                logging.info('training epoch #{} finished.'.format(epoch))
                epoch += 1
                _eval_on_dev_set()
                if train_config.checkpoints:
                    _save_model(epoch, step)

            logging.info('all training epochs finished.')

        logging.info('testing...')
        _test_decode(
            sess,
            model,
            'test',
            os.path.join(logdir, 'test'),
            LossLogger(("bleu",), os.path.join(logdir, "test_loss")),
            device)

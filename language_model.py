#!/usr/bin/env python3
import os
import logging
import random
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import texar as tx

from options import *
from model import Seq2Seq, Seq2SeqAttention, Seq2SeqFastAttention
from criterions.matrixBLEUave import mBLEU
from utils import strip_eos, onehot_initialization, find_valid_length, get_grad_norm
from evaluate import evaluate_model_
from logger import LossLogger

if hasattr(train_config, 'seed') and train_config.seed is not None:
    seed = train_config.seed
    random.seed(seed)
    tf.set_random_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def run_model(model, batch, target_vocab, teach_rate, device, verbose=False):
    src_sents, tgt_sents = batch['source_text_ids'], batch['target_text_ids']
    tgt_sents_onehot = torch.tensor(
        onehot_initialization(tgt_sents, target_vocab.size),
        dtype=torch.float,
        device=device)
    src_sents = torch.tensor(src_sents, dtype=torch.long, device=device)
    tgt_sents = torch.tensor(tgt_sents, dtype=torch.long, device=device)
    batch_size = tgt_sents.shape[0]

    ret = {
        'src_sents': src_sents,
        'tgt_sents': tgt_sents,
        'tgt_sents_onehot': tgt_sents_onehot,
    }

    if train_config.enable_cross_entropy:
        logits_ce = model(src_sents, tgt_sents[:, :-1])

        tgt_sents_ = tgt_sents[:, 1:]
        flatten_logits_ce = logits_ce.contiguous().view(-1, logits_ce.shape[-1])
        flatten_tgt_sents_ = tgt_sents_.contiguous().view(-1)
        cel = criterion_cross_entropy(flatten_logits_ce, flatten_tgt_sents_)

        ret['ce'] = {
            'logits': logits_ce,
            'loss': cel,
        }

    if train_config.enable_bleu:
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
        logits_mb = model(
            src_sents,
            tgt_sents[:, :-1],
            max_decode_length=train_config.max_decode_length,
            beam=beam,
            teach_flags=teach_flags)

        probs = F.softmax(logits_mb, dim=-1)
        probs = torch.cat([tgt_sents_onehot[:, :1], probs], dim=1)
        gen_probs, gen_ids = probs.max(-1)

        if hasattr(train_config, "teach_X") and not train_config.teach_X:
            X = probs
        else:
            X = []
            for t in range(probs.shape[1]):
                X.append((tgt_sents_onehot if teach_flags[t] else probs)[:, t])
            X[0] = torch.tensor(X[0], requires_grad=True)
            X = torch.stack(X, dim=1)
        Y = tgt_sents_onehot

        eos_id = target_vocab.eos_token_id
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
            Y, X, lenY, lenX, maskY, maskX,
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('device: {}'.format(device))

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

    criterion_bleu = mBLEU(train_config.max_order)
    criterion_cross_entropy = nn.CrossEntropyLoss(ignore_index=int(target_vocab.pad_token_id))

    step = 0

    def _save_model(epoch, step=0):
        name = 'model.epoch{}'.format(epoch)
        if step != 0:
            name += '.{}'.format(step)
        ckpt = os.path.join(logdir, name)
        logging.info('saving model into {} ...'.format(ckpt))
        torch.save(model.state_dict(), ckpt)

    def _train_epoch(sess, model, optimizer, pretrain, losses, verbose=verbose_config.verbose):
        global teach_rate
        def ids_to_words(ids):
            return sess.run(target_vocab.map_ids_to_tokens(ids), feed_dict=feed_dict)
        global step
        data_iterator.restart_dataset(sess, 'train')
        feed_dict = {data_iterator.handle: data_iterator.get_handle(sess, 'train')}
        model.train()
        for batch_i in range(train_config.train_batches):
            try:
                batch = sess.run(data_batch, feed_dict=feed_dict)
                sample_verbose = verbose and (step + 1) % verbose_config.steps_sample == 0
                batch_size = batch['target_text_ids'].shape[0]
                res = run_model(model, batch, target_vocab, teach_rate=teach_rate,
                                device=device, verbose=sample_verbose)
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
                grad_norm = get_grad_norm(model.parameters())

                if train_config.enable_bleu and sample_verbose:
                    def onehot(x):
                        return torch.tensor(onehot_initialization(x, target_vocab.size), dtype=torch.float, device=device)
                    samples = min(verbose_config.samples, batch_size)
                    tgt_ids = batch['target_text_ids']
                    gen_words, tgt_words = map(ids_to_words, (gen_ids, tgt_ids))
                    if verbose_config.probs_verbose:
                        gen_grads = (probs.grad * onehot(gen_ids)).sum(-1)
                        max_grads, max_ids = probs.grad.min(-1)
                        max_probs = (probs * onehot(max_ids)).sum(-1)
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
                step += 1
                if step % verbose_config.steps_loss == 0:
                    logging.info('step: {}\tloss: {:.3f}\tcel: {:.3f}\tmbl: {:.3f}\tgrad_norm: {:.3f}'.format(
                        step, loss_, cel_, mbl_, grad_norm))

                optimizer.step()

                if step % verbose_config.steps_eval == 0:
                    _eval_on_dev_set()
                    losses.plot(os.path.join(logdir, 'train_losses'))

                if train_config.checkpoints and step % verbose_config.steps_ckpt == 0:
                    _save_model(epoch, step)

                if train_config.enable_bleu and step % train_config.teach_rate_anneal_steps == 0:
                    teach_rate *= train_config.teach_rate_anneal
                    logging.info("teach rate: {}".format(teach_rate))

            except tf.errors.OutOfRangeError:
                break

    def _test_decode(sess, model, mode, out_path, losses, device, verbose=False):
        data_iterator.restart_dataset(sess, mode)
        feed_dict = {data_iterator.handle: data_iterator.get_handle(sess, mode)}
        bleu = evaluate_model_(model, sess, feed_dict, data_batch, target_vocab, verbose_config.eval_max_decode_length, verbose_config.eval_batches, verbose_config.eval_print_samples)
        logging.info("epoch #{} BLEU: {}".format(epoch, bleu))
        losses.append((bleu,))

    def _eval_on_dev_set(mode='val'):
        logging.info('evaluating on {} dataset...'.format(mode))
        _test_decode(
            sess,
            model,
            mode,
            os.path.join(logdir, '{}.epoch{}'.format(mode, epoch)),
            eval_losses,
            device)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    with tf.Session(config=sess_config) as sess:
        logging.info('running_mode: {}'.format(args.running_mode))

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        def _load_model(epoch, step=0):
            name = "model.epoch{}".format(epoch)
            if step != 0:
                name += ".{}".format(step)
            ckpt = os.path.join(logdir, name)
            logging.info('loading model from {} ...'.format(ckpt))
            model.load_state_dict(torch.load(ckpt))
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
                    _save_model(epoch)

            logging.info('all training epochs finished.')

        logging.info('testing...')
        _test_decode(
            sess,
            model,
            'test',
            os.path.join(logdir, 'test'),
            LossLogger(("bleu",), os.path.join(logdir, "test_loss")),
            device)

#!/usr/bin/env python3
import os
import logging
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import texar as tx

from options import *
from model import Seq2Seq, Seq2SeqAttention, Seq2SeqFastAttention
from criterions.matrixBLEU import mBLEU
from utils import strip_eos, onehot_initialization, find_valid_length
from evaluate import evaluate_model_

seed = train_config.seed
tf.set_random_seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

class LossLogger(object):
    def __init__(self, names, path):
        self.names = names
        if os.path.exists(path):
            with open(path, 'r') as f:
                names_ = tuple(f.readline().strip().split())
                assert self.names == names_, "given names: {} prev names: {}".format("\t".join(self.names), "\t".join(names_))
                self.a = [list(map(float, line.strip().split())) for line in f]
        else:
            with open(path, 'w') as f:
                print('\t'.join(names), file=f)
            self.a = []
        self.f = open(path, 'a', 1)
    def append(self, e):
        self.a.append(e)
        print('\t'.join(map(lambda x: "{:.6f}".format(x), e)), file=self.f)
    def recent(self, k):
        k = min(k, len(self.a))
        return list(map(np.mean, zip(*self.a[-k:])))
    def recent_repr(self, k):
        v = self.recent(k)
        return "\t".join("{}: {:.3f}".format(name, val) for name, val in zip(self.names, v))

def run_model(model, batch, target_vocab, device, verbose=False):
    src_sents, tgt_sents = batch['source_text_ids'], batch['target_text_ids']
    if train_config.enable_bleu:
        tgt_sents_onehot = torch.tensor(
            onehot_initialization(tgt_sents[:, 1:], target_vocab.size),
            dtype=torch.float,
            device=device)
    src_sents = torch.tensor(src_sents, dtype=torch.long, device=device)
    tgt_sents = torch.tensor(tgt_sents, dtype=torch.long, device=device)
    batch_size = tgt_sents.shape[0]

    logits = model(src_sents, tgt_sents[:, :-1])
    probs = F.softmax(logits, dim=-1)
    gen_probs, gen_ids = probs.max(-1)

    if train_config.enable_bleu:
        X = probs
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
        if train_config.softlengthmask:
            maskX, lenX = length_mask(X)
        else:
            assert X.shape == Y.shape, "X.shape={}, Y.shape={}".format(X.shape, Y.shape)
            maskX, lenX = maskY, lenY

        mbl = criterion_bleu(Y, X, lenY, lenX, maskY, maskX,
            recall_w=train_config.recall_w, device=device, verbose=verbose)
    else:
        mbl = torch.tensor(0.)

    tgt_sents_ = tgt_sents[:, 1:]
    flatten_logits = logits[:, : tgt_sents_.shape[1], :].contiguous().view([-1, logits.shape[-1]])
    flatten_tgt_sents = tgt_sents_.contiguous().view([-1])
    cel = criterion_cross_entropy(flatten_logits, flatten_tgt_sents)

    ret = {
        'mbl': mbl,
        'cel': cel,
        'logits': logits,
        'probs': probs,
        'gen_probs': gen_probs,
        'gen_ids': gen_ids,
    }
    if train_config.enable_bleu:
        ret.update({
            'X': X,
            'Y': Y,
        })
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
    data_iterator = tx.data.TrainTestDataIterator(
        train=training_data, val=valid_data, test=test_data)
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

    criterion_bleu = mBLEU(train_config.maxorder)
    criterion_cross_entropy = nn.CrossEntropyLoss(ignore_index=int(target_vocab.pad_token_id))

    step = 0

    def _train_epoch(sess, model, optimizer, pretrain, losses, verbose=verbose_config.verbose):
        def ids_to_words(ids):
            return sess.run(target_vocab.map_ids_to_tokens(ids), feed_dict=feed_dict)
        global step
        data_iterator.switch_to_train_data(sess)
        feed_dict = {tx.global_mode(): tf.estimator.ModeKeys.TRAIN}
        model.train()
        for batch_i in range(10000000):
            try:
                if batch_i >= train_config.train_batches:
                    break
                batch = sess.run(data_batch, feed_dict=feed_dict)
                sample_verbose = verbose and (step + 1) % verbose_config.steps_sample == 0
                batch_size = batch['target_text_ids'].shape[0]
                res = run_model(model, batch, target_vocab,
                                device=device, verbose=sample_verbose)
                probs = res['probs']
                if sample_verbose and verbose_config.probs_verbose:
                    probs.retain_grad()
                gen_ids = res['gen_ids']
                gen_probs = res['gen_probs']
                cel = res['cel']
                mbl = res['mbl']
                cel_ = cel.cpu().data.numpy()
                mbl_ = mbl.cpu().data.numpy()
                bleuw = train_config.bleuw
                if bleuw == 1.:
                    loss = mbl
                elif bleuw == 0.:
                    loss = cel
                else:
                    loss = cel * (1. - bleuw) + mbl * bleuw
                if pretrain:
                    if sample_verbose:
                        logging.info('pretraining')
                    loss = cel
                loss_ = loss.cpu().data.numpy()
                optimizer.zero_grad()
                loss.backward()
                if sample_verbose:
                    def onehot(x):
                        return torch.tensor(onehot_initialization(x, target_vocab.size), dtype=torch.float, device=device)
                    samples = min(verbose_config.samples, batch_size)
                    tgt_ids = batch['target_text_ids'][:, 1:]
                    gen_words, tgt_words = map(ids_to_words, (gen_ids, tgt_ids))
                    if verbose_config.probs_verbose:
                        gen_grads = (probs.grad * onehot(gen_ids)).sum(-1)
                        max_grads, max_ids = probs.grad.min(-1)
                        max_probs = (probs * onehot(max_ids)).sum(-1)
                        max_words = ids_to_words(max_ids)
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
                losses.append([loss_, cel_, mbl_])
                step += 1
                if step % verbose_config.steps_loss == 0:
                    logging.info('step: {}\tloss: {:.3f}\tcel: {:.3f}\tmbl: {:.3f}'.format(
                        step, loss_, cel_, mbl_))
                optimizer.step()
            except tf.errors.OutOfRangeError:
                break

    def _test_decode(sess, model, mode, out_path, losses, device, verbose=False):
        if mode == 'test':
            data_iterator.switch_to_test_data(sess)
        else:
            data_iterator.switch_to_val_data(sess)
        feed_dict = {tx.global_mode(): tf.estimator.ModeKeys.PREDICT}
        bleu = evaluate_model_(model, sess, feed_dict, data_batch, target_vocab, train_config.maxdecodelength, train_config.eval_batches, verbose_config.eval_print_samples)
        logging.info("epoch #{} BLEU: {}".format(epoch, bleu))
        losses.append((bleu,))

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    with tf.Session(config=sess_config) as sess:
        logging.info('running_mode: {}'.format(args.running_mode))

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        def _load_model(epoch):
            ckpt = os.path.join(logdir, "model.epoch{}".format(epoch))
            logging.info('loading model from {} ...'.format(ckpt))
            model.load_state_dict(torch.load(ckpt))
        if train_config.start_epoch is None:
            ckpts = list(filter(lambda s: s.startswith("model.epoch"), os.listdir(logdir)))
            if len(ckpts) == 0:
                epoch = 0
            else:
                epoch = max(map(lambda s: int(s[len('model.epoch'):]), ckpts))
                _load_model(epoch)
        else:
            epoch = train_config.start_epoch
            _load_model(epoch)

        if args.running_mode == 'train':
            optimizer = optim.Adam(model.parameters(), lr=train_config.lr)

            train_losses = LossLogger(("loss", "cel", "mbl"), os.path.join(logdir, "train_loss"))
            eval_losses = LossLogger(("bleu",), os.path.join(logdir, "eval_loss"))

            def _eval_on_dev_set():
                logging.info('evaluating on dev test...')
                _test_decode(
                    sess,
                    model,
                    'dev',
                    os.path.join(logdir, 'dev.epoch{}'.format(epoch)),
                    eval_losses,
                    device)

            _eval_on_dev_set()

            while epoch < train_config.max_epochs:
                logging.info('training epoch #{}:'.format(epoch))
                _train_epoch(sess, model, optimizer, epoch < train_config.pretrain, train_losses)
                logging.info('training epoch #{} finished.'.format(epoch))
                epoch += 1
                _eval_on_dev_set()
                if train_config.checkpoints:
                    ckpt = os.path.join(logdir, 'model.epoch{}'.format(epoch))
                    logging.info('saving model into {} ...'.format(ckpt))
                    torch.save(model.state_dict(), ckpt)

            logging.info('all training epochs finished.')

        logging.info('testing...')
        _test_decode(
            sess,
            model,
            'test',
            os.path.join(logdir, 'test'),
            LossLogger(("bleu",), os.path.join(logdir, "test_loss")),
            device)

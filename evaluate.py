"""Evaluation utils."""
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from data_utils import get_minibatch, get_autoencode_minibatch
from collections import Counter
import math
import numpy as np
import subprocess
import sys
import os
import operator
from socket import gethostname

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from texar.evals import sentence_bleu
from rouge import Rouge
rouge = Rouge()

import logging

import tensorflow as tf

plot_flag = (gethostname() not in ['quad-p40-0-0', 'quad-p40-0-1', 'dual-k40-0-1'])

if plot_flag:
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')

def dict_to_sorted_list(d):
    a = list(d.items())
    a.sort(key=operator.itemgetter(0))
    return a

def bleu_stats(hypothesis, reference):
    """Compute statistics for BLEU."""
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in range(1, 5):
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
        )
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats


def bleu(stats):
    """Compute BLEU given n-gram statistics."""
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)


def get_bleu(hypotheses, reference):
    """Get validation BLEU score for dev set."""
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    return 100 * bleu(stats)


def get_bleu_moses(hypotheses, reference):
    """Get BLEU score with moses bleu score."""
    with open('tmp_hypotheses.txt', 'w') as f:
        for hypothesis in hypotheses:
            f.write(' '.join(hypothesis) + '\n')

    with open('tmp_reference.txt', 'w') as f:
        for ref in reference:
            f.write(' '.join(ref) + '\n')

    hypothesis_pipe = '\n'.join([' '.join(hyp) for hyp in hypotheses])
    pipe = subprocess.Popen(
        ["perl", 'multi-bleu.perl', '-lc', 'tmp_reference.txt'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE
    )
    pipe.stdin.write(hypothesis_pipe)
    pipe.stdin.close()
    return pipe.stdout.read()

def decode_caption_batch(max_decode_length, decoder, encoder, src, tgt):
    features = encoder(src)
    return decoder(features, tgt, max_decode_length=max_decode_length, beam=1).max(-1)[1]

def decode_minibatch_(max_decode_length, model, src, tgt):
    """Decode a minibatch."""
    return model(src, tgt, max_decode_length=max_decode_length, beam=1).max(-1)[1]

def decode_minibatch(
    config,
    model,
    input_lines_src,
    input_lines_trg,
    output_lines_trg_gold
):
    """Decode a minibatch."""
    for i in range(config['data']['max_trg_length']):

        decoder_logit = model(input_lines_src, input_lines_trg)
        word_probs = model.decode(decoder_logit)
        decoder_argmax = word_probs.max(-1)[1]
        next_preds = decoder_argmax[:, -1]

        input_lines_trg = torch.cat(
            (input_lines_trg, next_preds.unsqueeze(1)),
            1
        )

    return input_lines_trg


def model_perplexity(
    model, src, src_test, trg,
    trg_test, config, loss_criterion,
    src_valid=None, trg_valid=None, verbose=False,
):
    """Compute model perplexity."""
    # Get source minibatch
    losses = []
    for j in range(0, len(src_test['data']) // 100, config['data']['batch_size']):
        input_lines_src, output_lines_src, lens_src, mask_src = get_minibatch(
            src_test['data'], src['word2id'], j, config['data']['batch_size'],
            config['data']['max_src_length'], add_start=True, add_end=True
        )
        input_lines_src = Variable(input_lines_src.data, volatile=True)
        output_lines_src = Variable(input_lines_src.data, volatile=True)
        mask_src = Variable(mask_src.data, volatile=True)

        # Get target minibatch
        input_lines_trg_gold, output_lines_trg_gold, lens_src, mask_src = (
            get_minibatch(
                trg_test['data'], trg['word2id'], j,
                config['data']['batch_size'], config['data']['max_trg_length'],
                add_start=True, add_end=True
            )
        )
        input_lines_trg_gold = Variable(input_lines_trg_gold.data, volatile=True)
        output_lines_trg_gold = Variable(output_lines_trg_gold.data, volatile=True)
        mask_src = Variable(mask_src.data, volatile=True)

        decoder_logit = model(input_lines_src, input_lines_trg_gold)

        loss = loss_criterion(
            decoder_logit.contiguous().view(-1, decoder_logit.size(2)),
            output_lines_trg_gold.view(-1)
        )

        losses.append(loss.data[0])

    return np.exp(np.mean(losses))

def to_str(sent, encoding):
    return b' '.join(sent).decode(encoding)

def average_len(tgt):
    return sum(map(len, tgt)) / len(tgt)

def apply_on_sent_pair(fn):
    def func(pair):
        refs, hyp = pair
        return ([fn(ref) for ref in refs], fn(hyp))
    return func

def evaluate_model_(
    model, encoder, sess, feed_dict, data_loader, target_vocab, ids_to_words,
    max_decode_length, eval_batches, writer, step, logdir, print_samples=0,
    encoding='utf-8'
):
    captioning = (encoder is not None)
    bos_token = target_vocab.bos_token.encode(encoding)
    eos_token = target_vocab.eos_token.encode(encoding)
    def strip_bos_and_eos(sent):
        if sent and sent[0] == bos_token:
            sent = sent[1:]
        try:
            return sent[:sent.index(eos_token)]
        except ValueError:
            return sent

    sent_pairs = []

    if not captioning:
        data_batch = data_loader
        def _get_data_loader():
            while True:
                try:
                    yield sess.run(data_batch, feed_dict=feed_dict)
                except tf.errors.OutOfRangeError:
                    break
        data_loader = _get_data_loader()

    for batch_i, batch in enumerate(data_loader):
        if batch_i >= eval_batches:
            break

        print('eval batch #{}'.format(batch_i))

        # Decode a minibatch greedily TODO add beam search decoding
        if captioning:
            images, captions = batch
            batch_size = len(captions)
            gen = decode_caption_batch(
                max_decode_length, model, encoder, images,
                torch.tensor([target_vocab.bos_token_id] * batch_size, device='cuda').unsqueeze(1))
            tgt = ids_to_words(captions)
        else:
            batch_size = batch['target_text_ids'].shape[0]
            gen = decode_minibatch_(
                max_decode_length, model,
                torch.LongTensor(batch['source_text_ids']).cuda(),
                torch.LongTensor(batch['target_text_ids'][:, :1]).cuda()
            )
            tgt = batch['target_text'].tolist()
            tgt = [[x] for x in tgt]
        gen = ids_to_words(gen.data.cpu().numpy()).tolist()

        # Process outputs
        sent_pairs.extend(map(apply_on_sent_pair(strip_bos_and_eos), zip(tgt, gen)))

    if print_samples > 0:
        logging.info("eval samples:")
        def log_sent(sent, name):
            text = to_str(sent, encoding)
            logging.info('{}: {}'.format(name, text))
            writer.add_text('val/{}'.format(name), text, step)
        for sent_i, (tgts, gen) in enumerate(sent_pairs):
            if sent_i >= print_samples:
                break
            for tgt in tgts:
                log_sent(tgt, 'tgt')
            log_sent(gen, 'gen')

    sent_pairs = list(filter(lambda pair: pair[0][0], sent_pairs))
    sent_pairs.sort(key=lambda sent_pair: (average_len(sent_pair[0]), sent_pair[0]))

    sent_bleu_fn = lambda tgt, gen: sentence_bleu(tgt, gen, smooth=True)
    sent_bleus = [sent_bleu_fn(tgt, gen) for tgt, gen in sent_pairs]
    lens = [average_len(tgt) for tgt, gen in sent_pairs]
    with open(os.path.join(logdir, "eval_bleus_step{}".format(step)), "w") as f:
        for score, (tgt, gen) in zip(sent_bleus, sent_pairs):
            print("{:.6f}\t{}\t{}".format(score, tgt, gen), file=f)
    with open(os.path.join(logdir, "eval_lens"), "w") as f:
        for x in lens:
            print("{:.6f}".format(x), file=f)
    if plot_flag:
        plt.figure(figsize=(14, 10))
        plt.bar(np.arange(len(sent_pairs)), np.array(sent_bleus) * 100,
                width=1.0, facecolor='black', edgecolor='black')
        plt.bar(np.arange(len(sent_pairs)), -np.array(lens), width=1.0)
        plt.savefig(os.path.join(logdir, "eval_bleus_step{}.png".format(step)))
        plt.close()

    tgts, gens = zip(*sent_pairs)
    corpus_bleu_score = corpus_bleu(tgts, gens)

    sent_pairs = list(map(apply_on_sent_pair(lambda s: to_str(s, encoding)), sent_pairs))
    tgts, gens = zip(*sent_pairs)
    gens = tuple([gen if gen else ' ' for gen in gens])

    rouge_scores = rouge.get_scores(
        gens, tuple(map(operator.itemgetter(0), tgts)), avg=True)
    rouge_scores = dict_to_sorted_list(rouge_scores)
    rouge_scores = [(key, dict_to_sorted_list(value)) for key, value in rouge_scores]
    s = 'ROUGE:'
    for name, scores in rouge_scores:
        s += '\n{}:'.format(name)
        for name2, score in scores:
            writer.add_scalar('val/{}/{}'.format(name, name2), score, step)
            s += ' {}: {:.3f}'.format(name2, score)
    logging.info(s)

    return corpus_bleu_score

def evaluate_model(
    model, src, src_test, trg,
    trg_test, config, src_valid=None, trg_valid=None,
    verbose=True, metric='bleu'
):
    """Evaluate model."""
    preds = []
    ground_truths = []
    len_data = len(src_test['data'])
    for j in range(0, len_data, config['data']['batch_size']):
        print('eval progress: {}/{} = {:4.1%}'.format(j, len_data, j / len_data))
        if j / len_data > 0.1:
            break

        # Get source minibatch
        input_lines_src, output_lines_src, lens_src, mask_src = get_minibatch(
            src_test['data'], src['word2id'], j, config['data']['batch_size'],
            config['data']['max_src_length'], add_start=True, add_end=True
        )

        # Get target minibatch
        input_lines_trg_gold, output_lines_trg_gold, lens_src, mask_src = (
            get_minibatch(
                trg_test['data'], trg['word2id'], j,
                config['data']['batch_size'], config['data']['max_trg_length'],
                add_start=True, add_end=True
            )
        )

        # Initialize target with <s> for every sentence
        input_lines_trg = Variable(torch.LongTensor(
            [
                [trg['word2id']['<s>']]
                for i in range(input_lines_src.size(0))
            ]
        )).cuda()

        # Decode a minibatch greedily __TODO__ add beam search decoding
        input_lines_trg = decode_minibatch(
            config, model, input_lines_src,
            input_lines_trg, output_lines_trg_gold
        )

        # Copy minibatch outputs to cpu and convert ids to words
        input_lines_trg = input_lines_trg.data.cpu().numpy()
        input_lines_trg = [
            [trg['id2word'][x] for x in line]
            for line in input_lines_trg
        ]

        # Do the same for gold sentences
        output_lines_trg_gold = output_lines_trg_gold.data.cpu().numpy()
        output_lines_trg_gold = [
            [trg['id2word'][x] for x in line]
            for line in output_lines_trg_gold
        ]

        # Process outputs
        for sentence_pred, sentence_real, sentence_real_src in zip(
            input_lines_trg,
            output_lines_trg_gold,
            output_lines_src
        ):
            if '</s>' in sentence_pred:
                index = sentence_pred.index('</s>')
            else:
                index = len(sentence_pred)
            preds.append(['<s>'] + sentence_pred[:index + 1])

            if verbose:
                print(' '.join(['<s>'] + sentence_pred[:index + 1]))

            if '</s>' in sentence_real:
                index = sentence_real.index('</s>')
            else:
                index = len(sentence_real)
            if verbose:
                print(' '.join(['<s>'] + sentence_real[:index + 1]))
            if verbose:
                print('--------------------------------------')
            ground_truths.append(['<s>'] + sentence_real[:index + 1])

    if False:
        for pred, gt in zip(preds, ground_truths):
            print('pred: {}'.format(' '.join(pred)))
            print('grth: {}'.format(' '.join(gt)))
    return get_bleu(preds, ground_truths)


def evaluate_autoencode_model(
    model, src, src_test,
    config, src_valid=None,
    verbose=True, metric='bleu'
):
    """Evaluate model."""
    preds = []
    ground_truths = []
    for j in range(0, len(src_test['data']), config['data']['batch_size']):

        print('Decoding batch : %d out of %d ' % (j, len(src_test['data'])))
        input_lines_src, lens_src, mask_src = get_autoencode_minibatch(
            src_test['data'], src['word2id'], j, config['data']['batch_size'],
            config['data']['max_src_length'], add_start=True, add_end=True
        )

        input_lines_trg = Variable(torch.LongTensor(
            [
                [src['word2id']['<s>']]
                for i in range(input_lines_src.size(0))
            ]
        )).cuda()

        for i in range(config['data']['max_src_length']):

            decoder_logit = model(input_lines_src, input_lines_trg)
            word_probs = model.decode(decoder_logit)
            decoder_argmax = word_probs.data.cpu().numpy().argmax(axis=-1)
            next_preds = Variable(
                torch.from_numpy(decoder_argmax[:, -1])
            ).cuda()

            input_lines_trg = torch.cat(
                (input_lines_trg, next_preds.unsqueeze(1)),
                1
            )

        input_lines_trg = input_lines_trg.data.cpu().numpy()

        input_lines_trg = [
            [src['id2word'][x] for x in line]
            for line in input_lines_trg
        ]

        output_lines_trg_gold = input_lines_src.data.cpu().numpy()
        output_lines_trg_gold = [
            [src['id2word'][x] for x in line]
            for line in output_lines_trg_gold
        ]

        for sentence_pred, sentence_real in zip(
            input_lines_trg,
            output_lines_trg_gold,
        ):
            if '</s>' in sentence_pred:
                index = sentence_pred.index('</s>')
            else:
                index = len(sentence_pred)
            preds.append(sentence_pred[:index + 1])

            if verbose:
                print(' '.join(sentence_pred[:index + 1]))

            if '</s>' in sentence_real:
                index = sentence_real.index('</s>')
            else:
                index = len(sentence_real)
            if verbose:
                print(' '.join(sentence_real[:index + 1]))
            if verbose:
                print('--------------------------------------')
            ground_truths.append(sentence_real[:index + 1])

    return get_bleu(preds, ground_truths)

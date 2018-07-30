"""Evaluation utils."""
import sys

sys.path.append('/u/subramas/Research/nmt-pytorch')

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from data_utils import get_minibatch, get_autoencode_minibatch
from collections import Counter
import math
import numpy as np
import subprocess
import sys

import tensorflow as tf
from utils import strip_eos


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


def decode_minibatch_(
    max_decode_length,
    model,
    input_lines_src,
    input_lines_trg
):
    """Decode a minibatch."""
    for i in range(max_decode_length):

        decoder_logit = model(input_lines_src, input_lines_trg)
        word_probs = model.decode(decoder_logit)
        decoder_argmax = word_probs.max(-1)[1]
        next_preds = decoder_argmax[:, -1]

        input_lines_trg = torch.cat(
            (input_lines_trg, next_preds.unsqueeze(1)),
            1
        )

    return input_lines_trg[:, 1:]

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

def evaluate_model_(
    model, sess, feed_dict, data_batch, target_vocab, max_decode_length,
    eval_batches,
    print_samples=0
):
    """Evaluate model."""
    gens, tgts = [], []
    strip_eos_fn = strip_eos(target_vocab.eos_token.encode())
    for batch_i in range(10000000):
        try:
            if batch_i >= eval_batches:
                break

            batch = sess.run(data_batch, feed_dict=feed_dict)
            batch_size = batch['target_text_ids'].shape[0]

            print('eval batch #{}'.format(batch_i))

            # Decode a minibatch greedily __TODO__ add beam search decoding
            gen = decode_minibatch_(
                max_decode_length, model,
                torch.LongTensor(batch['source_text_ids']).cuda(),
                torch.LongTensor(batch['target_text_ids'][:, :1]).cuda()
            )

            gen = gen.data.cpu().numpy()
            gen = sess.run(target_vocab.map_ids_to_tokens(gen), feed_dict=feed_dict)

            tgt = batch['target_text'][:, 1:]

            # Process outputs
            gen, tgt = map(lambda x: x.tolist(), (gen, tgt))
            gen, tgt = map(strip_eos_fn, (gen, tgt))
            gens.extend(gen)
            tgts.extend(tgt)
        except tf.errors.OutOfRangeError:
            break

    if print_samples > 0:
        print("eval samples:")
        for sent_i, (gen, tgt) in enumerate(zip(gens, tgts)):
            if sent_i >= print_samples:
                break
            print('gen: {}'.format(b' '.join(gen).decode()))
            print('tgt: {}'.format(b' '.join(tgt).decode()))
    return get_bleu(gen, tgt)

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

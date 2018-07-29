#!/usr/bin/env python3
"""Main script to run things"""
from data_utils import read_nmt_data, get_minibatch, read_config, hyperparam_string
from model import Seq2Seq, Seq2SeqAttention, Seq2SeqFastAttention
from criterions.matrixBLEU import mBLEU
from utils import onehot_initialization
from evaluate import evaluate_model
import math
import numpy as np
import logging
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    help="path to json config",
    required=True
)
args = parser.parse_args()
config_file_path = args.config
config = read_config(config_file_path)
experiment_name = hyperparam_string(config)
save_dir = config['data']['save_dir']
load_dir = config['data']['load_dir']
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='log/%s' % (experiment_name),
    filemode='w'
)

# define a new Handler to log to console as well
console = logging.StreamHandler()
# optional, set the logging level
console.setLevel(logging.INFO)
# set a format which is the same for console use
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)


print('Reading data ...')

src, trg = read_nmt_data(
    src=config['data']['src'],
    config=config,
    trg=config['data']['trg']
)

src_test, trg_test = read_nmt_data(
    src=config['data']['test_src'],
    config=config,
    trg=config['data']['test_trg']
)

batch_size = config['data']['batch_size']
max_length = config['data']['max_src_length']
src_vocab_size = len(src['word2id'])
trg_vocab_size = len(trg['word2id'])

logging.info('Model Parameters:')
logging.info('Task: %s' % (config['data']['task']))
logging.info('Model: %s' % (config['model']['seq2seq']))
logging.info('Source Language: %s' % (config['model']['src_lang']))
logging.info('Target Language: %s' % (config['model']['trg_lang']))
logging.info('Source Word Embedding Dim: %s' % (config['model']['dim_word_src']))
logging.info('Target Word Embedding Dim: %s' % (config['model']['dim_word_trg']))
logging.info('Source RNN Hidden Dim: %s' % (config['model']['dim']))
logging.info('Target RNN Hidden Dim: %s' % (config['model']['dim']))
logging.info('Source RNN Depth: %d' % (config['model']['n_layers_src']))
logging.info('Target RNN Depth: %d' % (config['model']['n_layers_trg']))
logging.info('Source RNN Bidirectional: %s' % (config['model']['bidirectional']))
logging.info('Batch Size: %d' % (config['data']['batch_size']))
logging.info('Valid Batch Size: %d' % (config['data']['valid_batch_size']))
logging.info('Optimizer: %s' % (config['training']['optimizer']))
logging.info('Learning Rate: %f' % (config['training']['lrate']))

logging.info('Found %d words in src' % (src_vocab_size))
logging.info('Found %d words in trg' % (trg_vocab_size))

weight_mask = torch.ones(trg_vocab_size).cuda()
weight_mask[trg['word2id']['<pad>']] = 0
criterion_cross_entropy = nn.CrossEntropyLoss(weight=weight_mask).cuda()
criterion_bleu = mBLEU(4)

if config['model']['seq2seq'] == 'vanilla':

    model = Seq2Seq(
        src_emb_dim=config['model']['dim_word_src'],
        trg_emb_dim=config['model']['dim_word_trg'],
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        src_hidden_dim=config['model']['dim'],
        trg_hidden_dim=config['model']['dim'],
        batch_size=batch_size,
        bidirectional=config['model']['bidirectional'],
        pad_token_src=src['word2id']['<pad>'],
        pad_token_trg=trg['word2id']['<pad>'],
        nlayers=config['model']['n_layers_src'],
        nlayers_trg=config['model']['n_layers_trg'],
        dropout=0.,
    ).cuda()

elif config['model']['seq2seq'] == 'attention':

    model = Seq2SeqAttention(
        src_emb_dim=config['model']['dim_word_src'],
        trg_emb_dim=config['model']['dim_word_trg'],
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        src_hidden_dim=config['model']['dim'],
        trg_hidden_dim=config['model']['dim'],
        ctx_hidden_dim=config['model']['dim'],
        attention_mode='dot',
        batch_size=batch_size,
        bidirectional=config['model']['bidirectional'],
        pad_token_src=src['word2id']['<pad>'],
        pad_token_trg=trg['word2id']['<pad>'],
        nlayers=config['model']['n_layers_src'],
        nlayers_trg=config['model']['n_layers_trg'],
        dropout=0.,
    ).cuda()

elif config['model']['seq2seq'] == 'fastattention':

    model = Seq2SeqFastAttention(
        src_emb_dim=config['model']['dim_word_src'],
        trg_emb_dim=config['model']['dim_word_trg'],
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        src_hidden_dim=config['model']['dim'],
        trg_hidden_dim=config['model']['dim'],
        batch_size=batch_size,
        bidirectional=config['model']['bidirectional'],
        pad_token_src=src['word2id']['<pad>'],
        pad_token_trg=trg['word2id']['<pad>'],
        nlayers=config['model']['n_layers_src'],
        nlayers_trg=config['model']['n_layers_trg'],
        dropout=0.,
    ).cuda()

# __TODO__ Make this more flexible for other learning methods.
if config['training']['optimizer'] == 'adam':
    lr = config['training']['lrate']
    optimizer = optim.Adam(model.parameters(), lr=lr)
elif config['training']['optimizer'] == 'adadelta':
    optimizer = optim.Adadelta(model.parameters())
elif config['training']['optimizer'] == 'sgd':
    lr = config['training']['lrate']
    optimizer = optim.SGD(model.parameters(), lr=lr)
else:
    raise NotImplementedError("Learning method not recommend for task")

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

losses = LossLogger(("loss", "cel", "mbl", "bll"), os.path.join("log", "{}.loss".format(experiment_name)))

bleus = LossLogger(("bleu",), os.path.join("log", "{}.bleu".format(experiment_name)))

pretrain_epochs = config["data"]["pretrain_epochs"]

for epoch_i in range(config['data']['last_epoch'], 1000):
    if load_dir:
        logging.info('loading model from {} ...'.format(load_dir))
        model.load_state_dict(torch.load(load_dir))

    for batch_i in range(0, len(src['data']), batch_size):
        if batch_i >= 0:
            break
        verbose = (batch_i % config['management']['print_samples'] == 0)
        input_lines_src, _, lens_src, mask_src = get_minibatch(
            src['data'], src['word2id'], batch_i,
            batch_size, max_length, add_start=True, add_end=True
        )
        input_lines_trg, output_lines_trg, lens_trg, mask_trg = get_minibatch(
            trg['data'], trg['word2id'], batch_i,
            batch_size, max_length, add_start=True, add_end=True
        )

        decoder_logit = model(input_lines_src, input_lines_trg)
        optimizer.zero_grad()

        X = torch.nn.functional.softmax(decoder_logit, dim=-1)
        Y = torch.tensor(
            onehot_initialization(output_lines_trg, trg_vocab_size),
            dtype=torch.float,
            device='cuda')

        eos_id = trg['word2id']['</s>']
        def length_mask(X):
            l = X.shape[1]
            mask = [torch.ones(X.shape[0], device='cuda')]
            for t in range(l):
                mask.append(mask[-1] * (1 - X[:, t, eos_id]))
            mask = torch.stack(mask, dim=1)
            lenX = torch.sum(mask, dim=1)
            return mask, lenX
        maskY, lenY = length_mask(Y)
        maskX, lenX = maskY, lenY

        mbl, mbl_ = criterion_bleu(Y, X, lenY, lenX, maskY, maskX, device='cuda', verbose=verbose)
        bll = torch.exp(-mbl)
        mbl = mbl.mean()
        mbl_ = mbl_.mean(0)
        bll = bll.mean()

        cel = criterion_cross_entropy(
            decoder_logit.contiguous().view(-1, trg_vocab_size),
            output_lines_trg.view(-1)
        )

        bleu_w = config['model']['bleu_w']
        if bleu_w == 0.:
            loss = cel
        elif bleu_w == 1.:
            loss = mbl
        else:
            loss = cel * (1. - bleu_w) + mbl * bleu_w
        if epoch_i < pretrain_epochs:
            loss = cel

        losses.append(list(map(lambda x: x.data.cpu().numpy(), (loss, cel, mbl, bll))))

        X.retain_grad()
        
        if verbose:
            for order in range(1, 5):
                optimizer.zero_grad()
                mbl_[order-1].backward(retain_graph=True)
                g = (X.grad[:, :, :Y.shape[-1]] * Y).sum(-1)
                print('grad({}):'.format(order))
                print(g[:5])
                print('grad({}) argmax:'.format(order))
                gw = X.grad[:, :, :Y.shape[-1]].min(-1)[1].cpu().numpy()
                gw = [[trg['id2word'][word_id] for word_id in sent] for sent in gw]
                for sent in gw[:5]:
                    print(' '.join(sent))

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        if verbose:
            t = X.grad[:, :, :Y.shape[-1]]
            g = (t * Y).sum(-1)
            print('grad:')
            print(g[:5])
            print('grad argmax:')
            gw = t.min(-1)[1].cpu().numpy()
            gw = [[trg['id2word'][word_id] for word_id in sent] for sent in gw]
            for sent in gw[:5]:
                print(' '.join(sent))
            w = torch.tensor(onehot_initialization(X.max(-1)[1], trg_vocab_size)[:, :, :Y.shape[-1]], dtype=torch.float, device='cuda')
            print('grad_:')
            print((t * w).sum(-1)[:5])

        monitor_loss_freq = config['management']['monitor_loss']
        if batch_i % monitor_loss_freq == 0:
            logging.info('epoch#{} batch{} {}'.format(
                epoch_i, batch_i, losses.recent_repr(monitor_loss_freq)))

        if (
            config['management']['print_samples'] and
            batch_i % config['management']['print_samples'] == 0
        ):
            word_probs = model.decode(
                decoder_logit
            ).data.cpu().numpy().argmax(axis=-1)

            output_lines_trg = output_lines_trg.data.cpu().numpy()
            samples = min(5, config['data']['batch_size'])
            for sentence_pred, sentence_real in zip(
                word_probs[:samples], output_lines_trg[:samples]
            ):
                sentence_pred = [trg['id2word'][x] for x in sentence_pred]
                sentence_real = [trg['id2word'][x] for x in sentence_real]

                if '</s>' in sentence_real:
                    index = sentence_real.index('</s>')
                    sentence_real = sentence_real[:index]
                    sentence_pred = sentence_pred[:index]
                #if '</s>' in sentence_pred:
                #    sentence_pred = sentence_pred[:sentence_pred.index('</s>')]

                logging.info('Pred : %s ' % (' '.join(sentence_pred)))
                logging.info('-----------------------------------------------')
                logging.info('Real : %s ' % (' '.join(sentence_real)))
                logging.info('===============================================')

        if batch_i % config['management']['checkpoint_freq'] == 0:

            logging.info('Evaluating model when batch_i = {} ...'.format(batch_i))
            bleu = evaluate_model(
                model, src, src_test, trg,
                trg_test, config, verbose=False,
                metric='bleu',
            )

            bleus.append((bleu,))
            logging.info('Epoch#%d batch%d BLEU: %.5f' % (epoch_i, batch_i, bleu))

            if save_dir:
                dir = os.path.join(
                    save_dir,
                    experiment_name + '__epoch_%d__minibatch_%d' % (epoch_i, batch_i) + '.model')
                logging.info('saving model into {} ...'.format(dir))
                torch.save(model.state_dict(), dir)
                load_dir = dir

        optimizer.step()
    
    print('epoch #{} eval...'.format(epoch_i))

    bleu = evaluate_model(
        model, src, src_test, trg,
        trg_test, config, verbose=False,
        metric='bleu',
    )

    bleus.append((bleu,))
    logging.info('Epoch#%d BLEU: %.5f' % (epoch_i, bleu))

    if save_dir:
        dir = os.path.join(
            save_dir,
            experiment_name + '__epoch_%d' % (epoch_i) + '.model')
        logging.info('saving model into {} ...'.format(dir))
        torch.save(model.state_dict(), dir)
        load_dir = dir

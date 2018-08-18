import os
import argparse
import importlib

def flag(s):
    return bool(int(s))

def get_data_name(config):
    return '{}__'.format(config.training_data_hparams['source_dataset']['vocab_file'].split('/')[-2])

def get_model_name(config):
    name = ''
    name += 'model_{}__'.format(config.task)
    name += 'dim_{}__'.format(config.dim)
    name += 'embdim_{}__'.format(config.embdim)
    name += 'nlayerssrc_{}__'.format(config.nlayerssrc)
    name += 'nlayerstgt_{}__'.format(config.nlayerstgt)
    name += 'bidir_{}__'.format(config.bidir)
    return name

def get_train_name(config):
    name = ''
    name += 'bleu_w_{}__'.format(config.bleu_w)
    name += 'max_order_{}__'.format(config.max_order)
    name += 'dropout_{}__'.format(config.dropout)
    name += 'soft_length_mask_{}__'.format(config.soft_length_mask)
    name += 'recall_w_{}__'.format(config.recall_w)
    name += 'max_decode_length_{}__'.format(config.max_decode_length)
    name += 'gamma_{}__'.format(config.gamma)
    name += 'lr_{}__'.format(config.lr)
    name += 'pretrain_{}__'.format(config.pretrain)
    if config.enable_bleu:
        name += 'fix_rate_{}_{}_{}__'.format(config.fix_teach_gap, config.teach_gap, config.teach_cont)
        name += 'teach_anneal_{}_{}_{}__'.format(config.initial_teach_rate, config.teach_rate_anneal, config.teach_rate_anneal_steps)
        if hasattr(config, 'teach_X'):
            name += 'teach_X_{}__'.format(config.teach_X)
    if hasattr(config, 'seed'):
        name += 'seed_{}__'.format(config.seed)
    return name

argparser = argparse.ArgumentParser()
argparser.add_argument('--train', type=str, default='train_config')
argparser.add_argument('--model', type=str, default='model_config')
argparser.add_argument('--data', type=str, default='data_configs')
argparser.add_argument('--verbose', type=str, default='verbose_config')
argparser.add_argument('--running_mode', type=str, default='train')
argparser.add_argument('--caption', type=str, default='')
args = argparser.parse_args()
train_config = importlib.import_module(args.train)
model_config = importlib.import_module(args.model)
data_config = importlib.import_module(args.data)
verbose_config = importlib.import_module(args.verbose)
if args.caption:
    captioning = True
    caption_config = importlib.import_module(args.caption)
else:
    captioning = False
mBLEU = train_config.mBLEU

if hasattr(train_config, "exp_name"):
    exp_name = train_config.exp_name
else:
    raise Exception("train config has no exp_name")
    exp_name = get_data_name(data_config) + get_model_name(model_config) + get_train_name(train_config)

logdir = os.path.join('log', exp_name)
if not os.path.exists(logdir):
    os.makedirs(logdir)

logging_file = os.path.join(logdir, 'logging.txt')

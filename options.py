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
    name += 'bleuw_{}__'.format(config.bleuw)
    name += 'maxorder_{}__'.format(config.maxorder)
    name += 'dropout_{}__'.format(config.dropout)
    name += 'softlengthmask_{}__'.format(config.softlengthmask)
    name += 'recall_{}__'.format(config.recall_w)
    name += 'max_decode_length_{}__'.format(config.max_decode_length)
    name += 'lr_{}__'.format(config.lr)
    name += 'pretrain_{}__'.format(config.pretrain)
    if config.enable_bleu:
        name += 'teach_anneal_{}_{}_{}__'.format(config.initial_teach_rate, config.teach_rate_anneal, config.teach_rate_anneal_steps)
    if hasattr(config, 'teach_X'):
        name += 'teach_X_{}__'.format(config.teach_X)
    name += 'seed_{}__'.format(config.seed)
    return name

argparser = argparse.ArgumentParser()
argparser.add_argument('--train_config', type=str, default='train_config')
argparser.add_argument('--model_config', type=str, default='model_config')
argparser.add_argument('--data_config', type=str, default='data_configs')
argparser.add_argument('--verbose_config', type=str, default='verbose_config')
argparser.add_argument('--running_mode', type=str, default='train')
args = argparser.parse_args()
train_config = importlib.import_module(args.train_config)
model_config = importlib.import_module(args.model_config)
data_config = importlib.import_module(args.data_config)
verbose_config = importlib.import_module(args.verbose_config)

exp_name = get_data_name(data_config) + get_model_name(model_config) + get_train_name(train_config)

logdir = os.path.join('log', exp_name)
if not os.path.exists(logdir):
    os.makedirs(logdir)

logging_file = os.path.join(logdir, 'logging.txt')

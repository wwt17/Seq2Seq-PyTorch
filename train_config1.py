bleuw = 1.0
enable_bleu = True
maxorder = 4
dropout = 0.
softlengthmask = False
maxdecodelength = 50

optimizer = "adam"
lr = 0.001

pretrain = 8

seed = 0

load_dir = None

start_epoch = 8
max_epochs = 1000
train_batches = 1
eval_batches = 1

if bleuw != 0:
    assert enable_bleu

checkpoints = False

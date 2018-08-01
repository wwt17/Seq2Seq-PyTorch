bleuw = 0.0
enable_bleu = False
maxorder = 4
dropout = 0.
softlengthmask = False
maxdecodelength = 50

optimizer = "adam"
lr = 0.0001

pretrain = 1000

seed = 0

load_dir = None

start_epoch = None
max_epochs = 1000
train_batches = 10000000
eval_batches = 20

if bleuw != 0:
    assert enable_bleu

checkpoints = True

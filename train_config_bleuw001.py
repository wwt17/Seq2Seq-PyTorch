mode = 'soft'
bleuw = 0.01
enable_cross_entropy = True
enable_bleu = True
recall_w = 0.0
maxorder = 4
dropout = 0.
softlengthmask = False
maxdecodelength = 50

optimizer = "adam"
lr = 1e-4

pretrain = 0

seed = 0

load_dir = None

start_epoch = 22
max_epochs = 1000
train_batches = 10000000
eval_batches = 10000000

if bleuw != 0:
    assert enable_bleu

checkpoints = True

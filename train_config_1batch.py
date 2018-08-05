mode = 'soft'
bleuw = 1.0
enable_cross_entropy = False
enable_bleu = True
recall_w = 0.5
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
train_batches = 1
eval_batches = 1

if bleuw != 0:
    assert enable_bleu

checkpoints = False

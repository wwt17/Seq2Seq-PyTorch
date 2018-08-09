initial_teach_rate = 0.5
teach_rate_anneal = 1.0
teach_rate_anneal_steps = 100
mode = 'soft'
bleuw = 1.
enable_cross_entropy = True
enable_bleu = True
recall_w = 0.0
maxorder = 4
dropout = 0.
softlengthmask = False
max_decode_length = None

optimizer = "adam"
lr = 1e-4

pretrain = 0

seed = 0

load_dir = None

start_epoch = 22
max_epochs = 1000
train_batches = 10000000

if bleuw != 0:
    assert enable_bleu

checkpoints = True

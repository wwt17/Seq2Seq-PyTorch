initial_teach_rate = 0.9
teach_rate_anneal = 1.0
teach_rate_anneal_steps = 100
gamma = 1.0
bleu_w = 1.0
enable_cross_entropy = True
enable_bleu = True
enable_prec = True
enable_recall = False
recall_w = 0.0
max_order = 4
dropout = 0.
soft_length_mask = False
max_decode_length = None

optimizer = "adam"
lr = 1e-5

pretrain = 0

load_dir = None

start_epoch = 22
max_epochs = 1000
train_batches = 1
eval_batches = 1

if bleu_w != 1:
    assert enable_cross_entropy
if bleu_w != 0:
    assert enable_bleu
if recall_w != 1:
    assert enable_prec
if recall_w != 0:
    assert enable_recall

checkpoints = False

fix_teach_gap = 1.0
teach_gap = 1
teach_cont = 4
initial_teach_rate = 1.0
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
min_fn = 'min'
min_c = 1.
dropout = 0.
soft_length_mask = False
max_decode_length = None

optimizer = "adam"
lr = 1e-5

pretrain = 0

load_dir = None

start_epoch = None
max_epochs = 10000
train_batches = 10000000
eval_batches = 10000000

if bleu_w != 1:
    assert enable_cross_entropy
if bleu_w != 0:
    assert enable_bleu
if recall_w != 1:
    assert enable_prec
if recall_w != 0:
    assert enable_recall

checkpoints = True

exp_name = "fix"

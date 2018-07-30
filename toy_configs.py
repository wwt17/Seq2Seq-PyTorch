training_data_hparams = {
    'num_epochs': 1,
    'batch_size': 32,
    'allow_smaller_final_batch': False,
    'source_dataset': {
        "files": ['nmt_data/toy_copy/train/sources.txt'],
        'vocab_file': 'nmt_data/toy_copy/train/sources.vocab',
        'max_seq_length': 50
    },
    'target_dataset': {
        "files": ['nmt_data/toy_copy/train/targets.txt'],
        'vocab_file': 'nmt_data/toy_copy/train/targets.vocab',
        'max_seq_length': 50
    }
}

valid_data_hparams = {
    'num_epochs': 1,
    'batch_size': 32,
    'allow_smaller_final_batch': False,
    'source_dataset': {
        "files": ['nmt_data/toy_copy/dev/sources.txt'],
        'vocab_file': 'nmt_data/toy_copy/train/sources.vocab',
    },
    'target_dataset': {
        "files": ['nmt_data/toy_copy/dev/targets.txt'],
        'vocab_file': 'nmt_data/toy_copy/train/targets.vocab',
    }
}

test_data_hparams = {
    'num_epochs': 1,
    'batch_size': 32,
    'allow_smaller_final_batch': False,
    'source_dataset': {
        "files": ['nmt_data/toy_copy/test/sources.txt'],
        'vocab_file': 'nmt_data/toy_copy/train/sources.vocab',
    },
    'target_dataset': {
        "files": ['nmt_data/toy_copy/test/targets.txt'],
        'vocab_file': 'nmt_data/toy_copy/train/targets.vocab',
    }
}

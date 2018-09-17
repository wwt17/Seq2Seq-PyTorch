training_data_hparams = {
    'shuffle': True,
    'num_epochs': 1,
    'batch_size': 80,
    'allow_smaller_final_batch': False,
    'source_dataset': {
        "files": ['data/wmt14/train.en'],
        'vocab_file': 'data/wmt14/vocab.en',
        'max_seq_length': 50
    },
    'target_dataset': {
        'files': ['data/wmt14/train.fr'],
        'vocab_file': 'data/wmt14/vocab.fr',
        'max_seq_length': 50
    }
}

test_data_hparams = {
    'shuffle': False,
    'num_epochs': 1,
    'batch_size': 80,
    'allow_smaller_final_batch': False,
    'source_dataset': {
        "files": ['data/wmt14/test.en'],
        'vocab_file': 'data/wmt14/vocab.en'
    },
    'target_dataset': {
        'files': ['data/wmt14/test.fr'],
        'vocab_file': 'data/wmt14/vocab.fr'
    }
}

valid_data_hparams = test_data_hparams

encoding = 'cp1252'

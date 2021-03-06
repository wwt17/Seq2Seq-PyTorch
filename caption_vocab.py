import nltk
import pickle
import argparse
import numpy as np
import torch
from collections import Counter
from pycocotools.coco import COCO


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)

    @property
    def size(self):
        return len(self.word2idx)

    def __call__(self, word):
        return self.word2idx.get(word, self.word2idx['<unk>'])

    def ids_to_words(self, ids):
        if isinstance(ids, int):
            return self.idx2word[ids].encode()
        if isinstance(ids, (list, tuple)):
            return list(map(self.ids_to_words, ids))
        if isinstance(ids, torch.Tensor):
            ids = ids.data.cpu().numpy()
        return np.vectorize(lambda idx: self.idx2word[idx].encode())(ids)

    @property
    def pad_token(self):
        return '<pad>'
    @property
    def pad_token_id(self):
        return self.word2idx[self.pad_token]
    @property
    def bos_token(self):
        return '<start>'
    @property
    def bos_token_id(self):
        return self.word2idx[self.bos_token]
    @property
    def eos_token(self):
        return '<end>'
    @property
    def eos_token_id(self):
        return self.word2idx[self.eos_token]
    @property
    def unk_token(self):
        return '<unk>'
    @property
    def unk_token_id(self):
        return self.word2idx[self.unk_token]

def build_vocab(json, threshold):
    """Build a simple vocabulary wrapper."""
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def main(args):
    vocab = build_vocab(json=args.caption_path, threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='data/annotations/captions_train2014.json', 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)

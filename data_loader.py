from __future__ import print_function, division, absolute_import, with_statement, unicode_literals, generators
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from caption_vocab import Vocabulary
from pycocotools.coco import COCO
from torch.nn.utils.rnn import pack_padded_sequence

# Image preprocessing, normalization for the pretrained resnet
crop_size = 224
normalizer = transforms.Normalize((0.485, 0.456, 0.406),
                                  (0.229, 0.224, 0.225))
train_transform = transforms.Compose([
    transforms.RandomCrop(crop_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalizer])
eval_transform = transforms.Compose([
    transforms.Resize([crop_size, crop_size]),
    transforms.ToTensor(),
    normalizer])

def tokenize_and_encapsulate(vocab):
    def fn(caption):
        """Convert caption (string) to word ids."""
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        ids = []
        ids.append(vocab.bos_token_id)
        ids.extend(map(vocab, tokens))
        ids.append(vocab.eos_token_id)
        return ids
    return fn

class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.vocab = vocab
        self.fn = tokenize_and_encapsulate(self.vocab)
        self.transform = transform

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

class CocoAnnDataset(CocoDataset):
    def __init__(self, root, json, vocab, transform=None):
        super(CocoAnnDataset, self).__init__(root, json, vocab, transform)
        self.anns = list(self.coco.anns.values())

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        ann = self.anns[index]
        caption = ann['caption']
        img_id = ann['image_id']
        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.fn(caption)

class CocoImgDataset(CocoDataset):
    def __init__(self, root, json, vocab, transform=None):
        super(CocoImgDataset, self).__init__(root, json, vocab, transform)
        self.imgToAnns = list(self.coco.imgToAnns.items())
        self.imgToAnns.sort()

    def __len__(self):
        return len(self.imgToAnns)

    def __getitem__(self, index):
        """Returns one data pair (image and captions)."""
        vocab = self.vocab
        img_id, anns = self.imgToAnns[index]
        captions = (ann['caption'] for ann in anns)
        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, list(map(self.fn, captions))

def ann_collate_fn_on_device(device):
    def collate_fn(data):
        """Creates mini-batch tensors from the list of tuples (image, caption).
        
        We should build custom collate_fn rather than using default collate_fn, 
        because merging caption (including padding) is not supported in default.

        Args:
            data: list of tuple (image, caption). 
                - image: torch tensor of shape (3, 256, 256).
                - caption: list.

        Returns:
            images: torch tensor of shape (batch_size, 3, 256, 256).
            targets: torch tensor of shape (batch_size, padded_length).
            lengths: list; valid length for each padded caption.
        """
        # Sort a data list by caption length (descending order).
        #data.sort(key=lambda x: len(x[1]), reverse=True)
        images, captions = zip(*data)

        # Merge images (from tuple of 3D tensor to 4D tensor).
        images = torch.stack(images, 0).to(device)

        # Merge captions (from tuple of 1D tensor to 2D tensor).
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths), dtype=torch.long)
        for i, cap in enumerate(captions):
            targets[i, :len(cap)] = torch.tensor(cap, dtype=torch.long)
        targets = targets.to(device)
        #targets = pack_padded_sequence(targets, lengths, batch_first=True)[0]
        return images, targets, lengths

    return collate_fn

def img_collate_fn_on_device(device):
    def collate_fn(data):
        """Creates mini-batch tensors from the list of tuples (image, captions).
        
        We should build custom collate_fn rather than using default collate_fn, 
        because merging caption (including padding) is not supported in default.

        Args:
            data: list of tuple (image, captions). 
                - image: torch tensor of shape (3, 256, 256).
                - captions: list of list.

        Returns:
            images: torch tensor of shape (batch_size, 3, 256, 256).
            captions: same as input.
        """
        images, captions = zip(*data)

        # Merge images (from tuple of 3D tensor to 4D tensor).
        images = torch.stack(images, 0).to(device)

        return images, captions

    return collate_fn

def get_ann_loader(root, json, vocab, batch_size, transform=train_transform, shuffle=True, num_workers=0, device='cuda'):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoAnnDataset(root=root,
                          json=json,
                          vocab=vocab,
                          transform=transform)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=ann_collate_fn_on_device(device))
    return data_loader

def get_img_loader(root, json, vocab, batch_size, transform=eval_transform, shuffle=False, num_workers=0, device='cuda'):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoImgDataset(root=root,
                          json=json,
                          vocab=vocab,
                          transform=transform)
    
    # Data loader for COCO dataset
    # This will return (images, captions) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=img_collate_fn_on_device(device))
    return data_loader

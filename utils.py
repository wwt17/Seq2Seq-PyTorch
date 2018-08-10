import numpy as np
import torch
import random
from torch.autograd import Variable

def strip_eos(eos_token):
    return lambda sents: [sent[:sent.index(eos_token)]
        if eos_token in sent else sent for sent in sents]

def _all_idx(idx, axis):
    grid = np.ogrid[tuple(map(slice, idx.shape))]
    grid.insert(axis, idx)
    return tuple(grid)

def onehot_initialization(a, vocab_size):
    ncols = vocab_size
    out = np.zeros(a.shape + (ncols,), dtype=int)
    out[_all_idx(a, axis=2)] = 1
    return out

def find_valid_length(inputs, symbol='eos'):
    final_pad = Variable(torch.ones([inputs.shape[0], 1]).long(), requires_grad=False) * 2
    # 2: EOS
    if torch.cuda.is_available():
        final_pad = final_pad.cuda()
    a= torch.cat((inputs, final_pad), 1)
    b= torch.nonzero(torch.eq(a, 2))
    #"只实现了EOS的部分"
    # if eos exist, should contain the eos
    seq_len = []
    idx = 0
    while(idx<b.shape[0]):
        bid, pos = b[idx][0], b[idx][1]
        if (bid == len(seq_len)).cpu().data.numpy():
            seq_len.append(pos+1)
        idx += 1
    seq_len = torch.tensor(seq_len)
    seq_len = torch.clamp(seq_len, 0, inputs.shape[1])
    return seq_len

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

if __name__ == '__main__':
    a= Variable(torch.cuda.LongTensor([[1,2,3,4,5,6], [0,4,5,6,6,7]]))
    seq_len = find_valid_lenth(a, symbol='eos')
    print(seq_len)

import torch
import torch.nn as nn
import logging
import math

class mBLEU(nn.Module):
    def __init__(self, max_order=4):
        super(mBLEU, self).__init__()
        self.max_order = max_order

    def forward(self, Y, X, lenY, lenX, maskY, maskX, device, verbose=False):
        batch_size = X.shape[0]
        sizeX = X.shape[1]
        sizeY = Y.shape[1]

        zero = torch.tensor(0., device=device)
        one = torch.tensor(1., device=device)
        eye = torch.eye(sizeX, device=device)

        XY = X.bmm(Y.transpose(1, 2))
        XX = X.bmm(X.transpose(1, 2))
        XX = torch.max(XX, eye)

        maskX = maskX.unsqueeze(2)
        maskY = maskY.unsqueeze(2)
        matchY = maskX.bmm(maskY.transpose(1, 2))
        tiled_maskX = maskX.expand(-1, -1, sizeX+1)
        matchX = torch.min(tiled_maskX, tiled_maskX.transpose(1, 2))

        o = []
        for order in range(1, self.max_order + 1):
            matchY = XY[:, : sizeX - order + 1, : sizeY - order + 1] * matchY[:, 1:, 1:]
            matchX = XX[:, : sizeX - order + 1, : sizeX - order + 1] * matchX[:, 1:, 1:]
            cntY = matchY.sum(2)
            cntX = matchX.sum(2)
            o_order = torch.min(one, cntY / torch.max(one, cntX))
            if verbose:
                print('order {}:\n{}'.format(order, o_order[0]))
            o_order = o_order.sum(1)
            o_order = o_order / lenX
            o.append(o_order + 1e-10)
        o = torch.stack(o, 1)

        log_geo_mean = torch.log(o).mean(1)
        log_bp = torch.min(zero, one - lenY / lenX)
        log_bleu = log_bp + log_geo_mean
        neg_log_bleu = - log_bleu
        return neg_log_bleu

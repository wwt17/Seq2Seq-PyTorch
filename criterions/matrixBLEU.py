import torch
import torch.nn as nn
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
        cntY_ = []
        cntX_ = []
        tmp = []
        for order in range(1, self.max_order + 1):
            matchY = XY[:, : sizeX - order + 1, : sizeY - order + 1] * matchY[:, 1:, 1:]
            matchX = XX[:, : sizeX - order + 1, : sizeX - order + 1] * matchX[:, 1:, 1:]
            cntY = matchY.sum(2)
            cntX = matchX.sum(2)
            o_order = torch.min(one, cntY / torch.max(one, cntX))
            #o_order = torch.min(cntX, cntY) / torch.max(one, torch.max(cntX, cntY))
            if verbose:
                cntY_.append(cntY)
                cntX_.append(cntX)
                tmp.append(o_order)
            o_order = o_order.sum(1)
            o_order = o_order / torch.max(one, lenX - (order - 1))
            o.append(o_order)
        o = torch.stack(o, 1)
        if verbose:
            for b in range(min(5, batch_size)):
                print('sample#{}:'.format(b))
                l = int(lenY[b].data.cpu().numpy() + 1e-6) + 1
                for order in range(1, self.max_order + 1):
                    print('{}-gram:'.format(order))
                    ll = l - (order - 1)
                    print('cntY:')
                    print(cntY_[order-1][b, :ll])
                    print('cntX:')
                    print(cntX_[order-1][b, :ll])
                    print('o_order:')
                    print(tmp[order-1][b, :ll])

        log_o = torch.log(o)
        log_geomean = log_o.mean(1)
        log_bp = torch.min(zero, one - lenY / lenX)
        log_bleu = log_bp + log_geomean
        neg_log_bleu = - log_bleu
        return neg_log_bleu, -log_o

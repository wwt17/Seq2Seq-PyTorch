import torch
import torch.nn as nn
import logging

class mBLEU(nn.Module):
    def __init__(self, max_order=4):
        super(mBLEU, self).__init__()
        self.max_order = max_order

    def forward(self, Y, X, lenY, lenX, maskY, maskX, recall_w=0., device='cuda', verbose=False):
        batch_size = X.shape[0]
        sizeX = X.shape[1]
        sizeY = Y.shape[1]

        zero = torch.tensor(0., device=device)
        one = torch.tensor(1., device=device)

        XY = X.bmm(Y.transpose(1, 2))
        XX = torch.max(X.bmm(X.transpose(1, 2)), torch.eye(sizeX, device=device))
        YY = torch.max(Y.bmm(Y.transpose(1, 2)), torch.eye(sizeY, device=device))

        maskX = maskX.unsqueeze(2)
        maskY = maskY.unsqueeze(2)
        matchXY = maskX.bmm(maskY.transpose(1, 2))
        tiled_maskX = maskX.expand(-1, -1, sizeX+1)
        matchXX = torch.min(tiled_maskX, tiled_maskX.transpose(1, 2))
        tiled_maskY = maskY.expand(-1, -1, sizeY+1)
        matchYY = torch.min(tiled_maskY, tiled_maskY.transpose(1, 2))

        cntXY_ = []
        cntXX_ = []
        cntYX_ = []
        cntYY_ = []
        o_order_X_ = []
        o_order_Y_ = []
        o_X_ = []
        o_Y_ = []
        #o = None#
        for order in range(1, self.max_order + 1):
            matchXY = XY[:, : sizeX - order + 1, : sizeY - order + 1] * matchXY[:, 1:, 1:]
            matchXX = XX[:, : sizeX - order + 1, : sizeX - order + 1] * matchXX[:, 1:, 1:]
            matchYY = YY[:, : sizeY - order + 1, : sizeY - order + 1] * matchYY[:, 1:, 1:]
            cntXY = matchXY.sum(2)
            cntXX = matchXX.sum(2)
            o_order_X = torch.tanh(cntXY / torch.max(one, cntXX))
            #o_order_X = torch.min(one - 0.00, torch.min(cntXX, cntXY) / torch.max(one, torch.max(cntXX, cntXY)))
            cntYX = matchXY.sum(1)
            cntYY = matchYY.sum(2)
            o_order_Y = torch.tanh(cntYX / torch.max(one, cntYY))
            if verbose:
                cntXY_.append(cntXY)
                cntXX_.append(cntXX)
                cntYX_.append(cntYX)
                cntYY_.append(cntYY)
                o_order_X_.append(o_order_X)
                o_order_Y_.append(o_order_X)
            o_X = o_order_X.sum(1) / torch.max(one, lenX - (order - 1))
            o_X_.append(o_X)
            o_Y = o_order_Y.sum(1) / torch.max(one, lenY - (order - 1))
            o_Y_.append(o_Y)
            #o = torch.cat([o, o_order], dim=1) if o is not None else o_order#
        o_X_ = torch.stack(o_X_, 1)
        o_Y_ = torch.stack(o_Y_, 1)
        if verbose:
            for b in range(min(5, batch_size)):
                logging.info('sample#{}:'.format(b))
                l = int(lenY[b].data.cpu().numpy() + 1e-6)
                for order in range(1, self.max_order + 1):
                    logging.info('{}-gram:'.format(order))
                    ll = l - (order - 1)
                    logging.info('cntXY:\n{}'.format(cntXY_[order-1][b, :ll]))
                    logging.info('cntXX:\n{}'.format(cntXX_[order-1][b, :ll]))
                    logging.info('o_order_X:\n{}'.format(o_order_X_[order-1][b, :ll]))
                    logging.info('cntYX:\n{}'.format(cntYX_[order-1][b, :ll]))
                    logging.info('cntYY:\n{}'.format(cntYY_[order-1][b, :ll]))
                    logging.info('o_order_Y:\n{}'.format(o_order_Y_[order-1][b, :ll]))

        w = torch.tensor([0.1, 0.3, 0.3, 0.3], device=device)
        log_o_X_ = torch.log(o_X_ + 1e-9)
        log_o_X_weighted = log_o_X_ * w
        log_geomean_X = log_o_X_weighted.sum(1)
        log_o_Y_ = torch.log(o_Y_ + 1e-9)
        log_o_Y_weighted = log_o_Y_ * w
        log_geomean_Y = log_o_Y_weighted.sum(1)
        #log_bp = torch.min(zero, one - lenY / lenX)
        #log_bleu = log_bp + log_geomean
        #log_bleu = log_bp + log_o.sum(1)#
        return - ((1. - recall_w) * log_geomean_X + recall_w * log_geomean_Y).sum(), \
               - ((1. - recall_w) * log_o_X_weighted + recall_w * log_o_Y_weighted).sum(0)

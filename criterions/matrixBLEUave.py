import torch
import torch.nn as nn
import logging

class mBLEU(nn.Module):
    def __init__(self, max_order=4):
        super(mBLEU, self).__init__()
        self.max_order = max_order

    def forward(self, Y, X, lenY, lenX, maskY, maskX, enable_prec=True, enable_recall=False, recall_w=0., min_fn='min', min_c=1., device='cuda', verbose=False):
        batch_size = X.shape[0]
        sizeX = X.shape[1]
        sizeY = Y.shape[1]

        zero = torch.tensor(0., device=device)
        one = torch.tensor(1., device=device)
        if min_fn == 'min':
            min_f = lambda x: torch.min(torch.tensor(min_c, device=device), x)
        elif min_fn == 'tanh':
            min_f = lambda x: torch.tanh(x / min_c)
        else:
            raise NotImplementedError("min_fn = {}".format(min_fn))

        XY = torch.gather(X, 2, Y.unsqueeze(1).expand([-1, X.shape[1], -1]))
        if enable_prec:
            XX = torch.max(X.bmm(X.transpose(1, 2)), torch.eye(sizeX, device=device))
        if enable_recall:
            YY = torch.max(Y.bmm(Y.transpose(1, 2)), torch.eye(sizeY, device=device))

        maskX = maskX.unsqueeze(2)
        maskY = maskY.unsqueeze(2)
        matchXY = maskX.bmm(maskY.transpose(1, 2))
        if enable_prec:
            tiled_maskX = maskX.expand(-1, -1, sizeX+1)
            matchXX = torch.min(tiled_maskX, tiled_maskX.transpose(1, 2))
        if enable_recall:
            tiled_maskY = maskY.expand(-1, -1, sizeY+1)
            matchYY = torch.min(tiled_maskY, tiled_maskY.transpose(1, 2))

        if verbose:
            if enable_prec:
                cntXY_ = []
                cntXX_ = []
                o_order_X_ = []
            if enable_recall:
                cntYX_ = []
                cntYY_ = []
                o_order_Y_ = []
        if enable_prec:
            tot_X_ = []
            o_X_ = []
        if enable_recall:
            tot_Y_ = []
            o_Y_ = []
        for order in range(1, self.max_order + 1):
            matchXY = XY[:, : sizeX - order + 1, : sizeY - order + 1] * matchXY[:, 1:, 1:]
            if enable_prec:
                matchXX = XX[:, : sizeX - order + 1, : sizeX - order + 1] * matchXX[:, 1:, 1:]
            if enable_recall:
                matchYY = YY[:, : sizeY - order + 1, : sizeY - order + 1] * matchYY[:, 1:, 1:]
            if enable_prec:
                cntXY = matchXY.sum(2)
                cntXX = matchXX.sum(2)
                o_order_X = min_f(cntXY / torch.max(one, cntXX))
                #o_order_X = min_f(torch.min(cntXX, cntXY) / torch.max(one, torch.max(cntXX, cntXY)))
            if enable_recall:
                cntYX = matchXY.sum(1)
                cntYY = matchYY.sum(2)
                o_order_Y = min_f(cntYX / torch.max(one, cntYY))
            if verbose:
                if enable_prec:
                    cntXY_.append(cntXY)
                    cntXX_.append(cntXX)
                    o_order_X_.append(o_order_X)
                if enable_recall:
                    cntYX_.append(cntYX)
                    cntYY_.append(cntYY)
                    o_order_Y_.append(o_order_Y)
            if enable_prec:
                tot_X = torch.max(one, lenX - (order-1))
                o_X = o_order_X.sum(1)
                tot_X_.append(tot_X)
                o_X_.append(o_X)
            if enable_recall:
                tot_Y = torch.max(one, lenY - (order-1))
                o_Y = o_order_Y.sum(1)
                tot_Y_.append(tot_Y)
                o_Y_.append(o_Y)
        if enable_prec:
            tot_X_ = torch.stack(tot_X_, 1)
            o_X_ = torch.stack(o_X_, 1)
        if enable_recall:
            tot_Y_ = torch.stack(tot_Y_, 1)
            o_Y_ = torch.stack(o_Y_, 1)

        if verbose:
            for b in range(min(5, batch_size)):
                logging.info('sample#{}:'.format(b))
                l = int(lenY[b].data.cpu().numpy() + 1e-6)
                for order in range(1, self.max_order + 1):
                    logging.info('{}-gram:'.format(order))
                    ll = l - (order - 1)
                    if enable_prec:
                        logging.info('cntXY:\n{}'.format(cntXY_[order-1][b, :ll]))
                        logging.info('cntXX:\n{}'.format(cntXX_[order-1][b, :ll]))
                        logging.info('o_order_X:\n{}'.format(o_order_X_[order-1][b, :ll]))
                    if enable_recall:
                        logging.info('cntYX:\n{}'.format(cntYX_[order-1][b, :ll]))
                        logging.info('cntYY:\n{}'.format(cntYY_[order-1][b, :ll]))
                        logging.info('o_order_Y:\n{}'.format(o_order_Y_[order-1][b, :ll]))

        w = torch.tensor([0.1, 0.3, 0.3, 0.3], device=device)

        if enable_prec:
            o_X = o_X_.sum(0) / tot_X_.sum(0)
            neglog_o_X = -torch.log(o_X + 1e-9)
            neglog_o_X_weighted = neglog_o_X * w
            neglog_geomean_X = neglog_o_X_weighted.sum(-1)
        else:
            neglog_o_X_weighted = zero
            neglog_geomean_X = zero

        if enable_recall:
            o_Y = o_Y_.sum(0) / tot_Y_.sum(0)
            neglog_o_Y = -torch.log(o_Y + 1e-9)
            neglog_o_Y_weighted = neglog_o_Y * w
            neglog_geomean_Y = neglog_o_Y_weighted.sum(-1)
        else:
            neglog_o_Y_weighted = zero
            neglog_geomean_Y = zero

        neglog_bp = torch.max(zero, lenY.sum() / lenX.sum() - one)

        return (1. - recall_w) * neglog_geomean_X + recall_w * neglog_geomean_Y + neglog_bp, \
               (1. - recall_w) * neglog_o_X_weighted + recall_w * neglog_o_Y_weighted

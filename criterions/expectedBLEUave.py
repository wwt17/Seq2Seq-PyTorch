import torch
import torch.nn as nn
import logging

class mBLEU(nn.Module):
    def __init__(self, max_order=4):
        super(mBLEU, self).__init__()
        self.max_order = max_order

    def forward(self, Y, X, lenY, lenX, maskY, maskX, enable_prec=True, enable_recall=False, recall_w=0., min_fn='min', min_c=1., device='cuda', verbose=False):
        assert not enable_recall
        assert enable_prec
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
        Y_ = Y.unsqueeze(2).expand([-1, -1, Y.shape[1]])
        YY = (Y_ == Y_.transpose(1, 2)).float()

        maskX = maskX.unsqueeze(2)
        maskY = maskY.unsqueeze(2)
        matchXY = maskX.bmm(maskY.transpose(1, 2))
        tiled_maskY = maskY.expand(-1, -1, sizeY+1)
        matchYY = torch.min(tiled_maskY, tiled_maskY.transpose(1, 2))

        tot_X_ = []
        o_X_ = []
        for order in range(1, self.max_order + 1):
            matchXY = XY[:, : sizeX - order + 1, : sizeY - order + 1] * matchXY[:, 1:, 1:]
            matchYY = YY[:, : sizeY - order + 1, : sizeY - order + 1] * matchYY[:, 1:, 1:]
            cntYX = matchXY.sum(1)
            cntYY = matchYY.sum(2)
            o_order_X = (min_f(cntYY.unsqueeze(1) / (cntYX.unsqueeze(1) - matchXY + one))
                         * matchXY / torch.max(one, cntYY).unsqueeze(1)).sum(2)
            tot_X = torch.max(one, lenX - (order-1))
            o_X = o_order_X.sum(1)
            tot_X_.append(tot_X)
            o_X_.append(o_X)
        if enable_prec:
            tot_X_ = torch.stack(tot_X_, 1)
            o_X_ = torch.stack(o_X_, 1)
        if enable_recall:
            tot_Y_ = torch.stack(tot_Y_, 1)
            o_Y_ = torch.stack(o_Y_, 1)

        if verbose and False:
            for b in range(min(5, batch_size)):
                logging.info('sample#{}:'.format(b))
                l = int(lenY[b].data.cpu().numpy() + 1e-6)
                for order in range(1, self.max_order + 1):
                    logging.info('{}-gram:'.format(order))
                    ll = l - (order - 1)

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

import torch

def to_onehot(a, n, dtype=torch.long, device=None):
    if device is None:
        device = a.device
    ret = torch.zeros(a.shape + (n,), dtype=dtype, device=device)
    a = a.unsqueeze(-1)
    ret.scatter_(-1, a, torch.ones(a.shape, dtype=dtype, device=device))
    return ret

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

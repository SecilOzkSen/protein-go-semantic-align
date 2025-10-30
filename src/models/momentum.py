import torch

def clone_as_target(net):
    import copy
    k = copy.deepcopy(net).eval()
    for p in k.parameters():
        p.requires_grad_(False)
    return k

@torch.no_grad()
def momentum_update(q, k, m=0.999):
    for p_q, p_k in zip(q.parameters(), k.parameters()):
        p_k.data.mul_(m).add_(p_q.data, alpha=1.-m)
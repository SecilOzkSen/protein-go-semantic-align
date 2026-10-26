import torch
from .model import ResidualV3,pfresgo_soft_f

def main():
    torch.manual_seed(1)
    x=torch.randn(6,12)*.2+.3
    e=torch.randn(6,16)
    y=torch.zeros_like(x);y[:,[2,4]]=1
    model=ResidualV3(16);model.set_protein_normalization(e)
    edges=[torch.tensor([],dtype=torch.long) for _ in range(12)]
    edges[2]=torch.tensor([0,1]);edges[4]=torch.tensor([1])
    o=model(x,e)
    assert torch.equal(torch.argsort(x,dim=1),torch.argsort(o['prob'],dim=1))
    loss=pfresgo_soft_f(o['prob'],y,edges,0)
    assert torch.isfinite(loss)
    assert 0 <= float(loss.detach()) <= 1
    loss.backward()
    assert model.alpha_logit.grad is not None and torch.isfinite(model.alpha_logit.grad)
    assert model.boundary_head.weight.grad is not None
    # No feasible checkpoint is a supported RAW fallback in fit/evaluate.
    print('PASS: residual ranking, multi-threshold ancestor soft-F, gradients')
if __name__=='__main__':main()

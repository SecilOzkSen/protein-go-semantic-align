import torch
from .model import ProteinCalibrationV3, cafa_soft_f_loss

def main():
    torch.manual_seed(42)
    s=torch.randn(12,43)*.1+.3; p=torch.randn(12,32); y=torch.zeros(12,43)
    for i in range(12): y[i,:i%7+1]=1
    m=ProteinCalibrationV3(32);m.set_protein_normalization(p)
    out=m(s,p)
    assert out['scores'].shape==s.shape
    assert torch.equal(s.argsort(1),out['scores'].argsort(1))
    loss=cafa_soft_f_loss(out['scores'],y);loss.backward()
    assert m.global_boundary.grad is not None and m.global_boundary.grad.abs().sum()>0
    print('PASS: V3 shapes, ranking preservation, gradients, CAFA surrogate')
if __name__=='__main__':main()

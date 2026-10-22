"""Quick local smoke checks: shape, gradient, ranking invariance, cardinality pathway."""
import torch
from .model import CardinalityGuidedCalibrator, soft_f1_loss


def main():
    torch.manual_seed(42)
    x = torch.randn(6, 1943) * .1 + .3
    y = torch.zeros_like(x)
    y[:, :15] = 1
    m = CardinalityGuidedCalibrator()
    z = m(x)
    assert z['scores'].shape == x.shape
    assert z['cardinality'].shape == (6, 1)
    assert torch.equal(torch.argsort(x, dim=1), torch.argsort(z['scores'], dim=1))
    loss = soft_f1_loss(z['scores'], y) + .2 * torch.nn.functional.smooth_l1_loss(z['log_card'], torch.log1p(y.sum(1, keepdim=True)))
    loss.backward()
    assert m.card_head.weight.grad is not None and torch.isfinite(m.card_head.weight.grad).all()
    assert m.scale_head.weight.grad is not None and torch.isfinite(m.scale_head.weight.grad).all()
    print('PASS: shapes, order, gradients')

if __name__ == '__main__': main()

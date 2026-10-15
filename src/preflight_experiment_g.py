"""Run after installing Experiment G files into src/models and src/training."""
import torch
import torch.nn.functional as F
from src.models.alignment_model import LowRankResidualInteraction


def main():
    torch.manual_seed(42)
    model = LowRankResidualInteraction(dim=768, rank=128, residual_scale=0.02)
    p = F.normalize(torch.randn(4, 768), dim=-1).requires_grad_()
    g = F.normalize(torch.randn(1943, 768), dim=-1).requires_grad_()
    baseline = p @ g.T
    scores = model(p, g)
    assert scores.shape == (4, 1943)
    assert torch.allclose(scores, baseline, atol=1e-6, rtol=1e-6)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask[0, :4] = True
    mask[1, 5:8] = True
    mask[2, 9:20] = True
    mask[3, 21:27] = True
    loss = -(F.log_softmax(scores / 0.07, dim=1) * mask).sum(dim=1).div(mask.sum(1)).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert model.output.weight.grad is not None and model.output.weight.grad.abs().sum() > 0
    assert p.grad is not None and p.grad.abs().sum() > 0
    assert g.grad is not None and g.grad.abs().sum() > 0
    # Zero output means inner interaction projections receive gradients after the first update.
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    opt.step()
    opt.zero_grad()
    scores2 = model(p.detach(), g.detach())
    loss2 = -(F.log_softmax(scores2 / 0.07, dim=1) * mask).sum(dim=1).div(mask.sum(1)).mean()
    loss2.backward()
    assert model.protein_features.weight.grad is not None and model.protein_features.weight.grad.abs().sum() > 0
    assert model.go_features.weight.grad is not None and model.go_features.weight.grad.abs().sum() > 0
    # Check the [B,G,D] path matches [G,D] when broadcasted.
    with torch.no_grad():
        smallp = p.detach()[:2]
        smallg = g.detach()[:9]
        s2 = model(smallp, smallg)
        s3 = model(smallp, smallg[None].expand(2, -1, -1))
        assert torch.allclose(s2, s3, atol=1e-6, rtol=1e-6)
    print('EXPERIMENT G PREFLIGHT PASSED')
    print('initial cosine parity: yes')
    print('score shape: [4, 1943]')
    print('protein/GO/output/interaction gradients: yes')
    print('2D/3D GO scoring parity: yes')


if __name__ == '__main__':
    main()

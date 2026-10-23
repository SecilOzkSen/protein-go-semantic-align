"""Run: python -m src.protein_calibrationv2.test_smoke"""
import numpy as np
import torch
from .model import CardinalityGuidedCalibratorV2, soft_f1_loss
from .fit import cardinality_bin_weights, weighted_card_loss


def main():
    torch.manual_seed(42)
    n, g, d = 64, 1943, 1280
    scores = torch.randn(n, g) * .1 + .3
    protein = torch.randn(n, d)
    labels = torch.zeros(n, g)
    for i in range(n):
        labels[i, :[3, 8, 14, 25, 55, 105, 170][i % 7]] = 1
    model = CardinalityGuidedCalibratorV2(d)
    model.set_protein_normalization(protein)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    w, _, _ = cardinality_bin_weights(labels)
    for _ in range(2):
        out = model(scores, protein)
        assert out['scores'].shape == scores.shape
        assert torch.equal(torch.argsort(scores, dim=1), torch.argsort(out['scores'], dim=1))
        loss = soft_f1_loss(out['scores'], labels) + .2 * weighted_card_loss(
            out['log_card'], torch.log1p(labels.sum(1, keepdim=True)), w)
        opt.zero_grad()
        loss.backward()
        if _ == 0:
            opt.step()
    assert model.protein_encoder[0].weight.grad is not None
    assert torch.isfinite(model.protein_encoder[0].weight.grad).all()
    assert model.protein_encoder[0].weight.grad.abs().sum() > 0
    print('PASS: V2 shapes, ranking preservation, protein gradient, weighted loss')


if __name__ == '__main__':
    main()

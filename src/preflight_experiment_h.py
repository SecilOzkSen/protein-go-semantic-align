"""Run with python -m src.preflight_experiment_h after installing Experiment H files."""
import torch
from src.models.alignment_model import ProteinGoAligner


def main():
    torch.manual_seed(42)
    model = ProteinGoAligner(d_h=16, d_g=12, d_z=8, go_encoder=None)
    model.train()
    H = torch.randn(3, 280, 16)
    mask = torch.zeros(3, 280, dtype=torch.bool)
    mask[0, :280] = True
    mask[1, :70] = True
    mask[2, :192] = True
    G, P = 9, 4
    name = torch.randn(G, 12)
    definition = torch.randn(G, 12)
    parent = torch.randn(G, P, 12)
    parent_mask = torch.rand(G, P) > 0.5
    definition_mask = torch.rand(G) > 0.2
    global_go = torch.randn(G, 12)
    scores = model.score_fine_grained(H, mask, name, definition,
                                      definition_mask, parent, parent_mask, global_go)
    assert scores.shape == (3, G) and torch.isfinite(scores).all()
    scores.sum().backward()
    for module_name in ('proj_p', 'proj_local', 'proj_g', 'fusion_gate'):
        module = getattr(model, module_name)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()), module_name
    assert model.last_h_diagnostics['h/window_count_mean'] > 0
    model.eval()
    with torch.no_grad():
        a = model.score_fine_grained(H, mask, name, definition, definition_mask, parent, parent_mask, global_go)
        b = model.score_fine_grained(H, mask, name, definition, definition_mask, parent, parent_mask, global_go)
        assert torch.allclose(a, b, atol=1e-6)
    print('EXPERIMENT H PREFLIGHT PASSED: shape, finite scores, masks, gradients, deterministic eval')


if __name__ == '__main__':
    main()

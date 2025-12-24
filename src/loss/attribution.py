# Created by Secil Sen

def attribution_loss(alpha, delta_y, mask=None, reduce="mean"):
    # alpha, delta_y: [B, P, L] or [..., L]
    diff = (alpha - delta_y).abs()

    if mask is not None:
        # mask: True = valid residue
        mask_f = mask.to(diff.dtype)

        # expand mask if needed: [B, 1, L]
        while mask_f.dim() < diff.dim():
            mask_f = mask_f.unsqueeze(1)

        diff = diff * mask_f
        denom = mask_f.sum().clamp_min(1.0)
        val = diff.sum() / denom
    else:
        val = diff.mean() if reduce == "mean" else diff.sum()

    return val


def windowed_attr_loss(alpha_windows, win_weights, spans, delta_y_windows):
    # alpha_windows: (B,T,W,win)
    # win_weights: (B,T,W)
    # delta_y_windows: (B,T,W,win) (surrogate or true mask-out)
    per_win = (alpha_windows - delta_y_windows).abs().mean(dim=-1)  # (B,T,W)
    weighted = (per_win * win_weights).mean(dim=-1)                  # (B,T)
    return weighted.mean()

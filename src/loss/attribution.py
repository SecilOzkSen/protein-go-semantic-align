# Created by Secil Sen

def attribution_loss(alpha, delta_y, mask=None, reduce="mean"):
    # alpha: (..., L) veya pencere-bazlı birleştirilmiş L_eff
    # delta_y: alpha ile aynı şekil (normalize edilmiş veya skaler farklar)
    diff = (alpha - delta_y).abs()
    if mask is not None:
        diff = diff * (~mask).float()
        denom = mask.numel() - mask.sum().item()
        val = diff.sum() / max(1, denom)
    else:
        val = diff.mean() if reduce == "mean" else diff.sum()
    return val

def windowed_attr_loss(alpha_windows, win_weights, spans, delta_y_windows):
    # alpha_windows: (B,T,W,win)
    # win_weights: (B,T,W)
    # delta_y_windows: (B,T,W,win) (surrogate veya true mask-out)
    per_win = (alpha_windows - delta_y_windows).abs().mean(dim=-1)  # (B,T,W)
    weighted = (per_win * win_weights).mean(dim=-1)                  # (B,T)
    return weighted.mean()

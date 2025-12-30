import torch
@torch.no_grad()
def retrieval_metrics_from_scores(scores, pos_mask, ks=(1,5,10)):
    scores = scores.float()
    pos_mask = pos_mask.bool()

    # sort descending
    rank_idx = torch.argsort(scores, dim=1, descending=True)
    pos_ranked = torch.gather(pos_mask, 1, rank_idx)  # (B,T)

    pos_count = pos_mask.sum(dim=1)
    keep = pos_count > 0
    if keep.sum() == 0:
        return {f"R@{k}": 0.0 for k in ks} | {"MRR": 0.0, "nDCG@10": 0.0, "num": 0}

    pos_ranked = pos_ranked[keep]
    pos_count = pos_count[keep].float()
    out = {"num": int(pos_ranked.size(0))}

    # Recall@K
    for k in ks:
        k = min(k, pos_ranked.size(1))
        hits = pos_ranked[:, :k].sum(dim=1).float()
        out[f"R@{k}"] = (hits / pos_count.clamp_min(1.0)).mean().item()

    # MRR (first positive rank)
    first_pos = pos_ranked.float().argmax(dim=1)
    out["MRR"] = (1.0 / (first_pos.float() + 1.0)).mean().item()

    # nDCG@10
    k_ndcg = min(10, pos_ranked.size(1))
    rel = pos_ranked[:, :k_ndcg].float()
    denom = torch.log2(torch.arange(2, k_ndcg + 2, device=rel.device).float())
    dcg = (rel / denom).sum(dim=1)

    ideal_hits = torch.minimum(pos_count, torch.tensor(float(k_ndcg), device=rel.device))
    gains = (1.0 / denom)
    idcg = torch.zeros_like(dcg)
    for i in range(k_ndcg):
        idcg += gains[i] * (ideal_hits > i).float()

    out["nDCG@10"] = (dcg / idcg.clamp_min(1e-12)).mean().item()
    return out
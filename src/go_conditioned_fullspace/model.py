import torch
from torch import nn
import torch.nn.functional as F

class FullSpaceCrossScorer(nn.Module):
    """Shared GO-conditioned attention scorer. No GO-specific output weights."""
    def __init__(self, dim=768, hidden=128, dropout=.1, residual_scale=2.):
        super().__init__()
        self.p=nn.Linear(dim,hidden,bias=False)
        self.q=nn.Linear(dim,hidden,bias=False)
        self.v=nn.Linear(dim,hidden,bias=False)
        self.segment_gate=nn.Sequential(nn.Linear(hidden*2+3,64),nn.GELU(),
                                        nn.Dropout(dropout),nn.Linear(64,1))
        self.out=nn.Sequential(nn.Linear(hidden*2+5,128),nn.GELU(),
                               nn.Dropout(dropout),nn.Linear(128,1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)
        self.residual_scale=residual_scale

    def forward(self, windows, valid, go_segments, present, base):
        # windows [B,W,D], go_segments [G,2,D], present [G,2], base [B,G]
        B,W,D=windows.shape; G=go_segments.shape[0]
        pk=self.p(windows)                    # B,W,H
        pv=self.v(windows)
        q=self.q(go_segments)                 # G,2,H
        logits=torch.einsum('bwh,gsh->bgsw',pk,q)*(pk.shape[-1]**-.5)
        logits=logits.masked_fill(~valid[:,None,None,:],-1e4)
        a=logits.softmax(-1)
        ctx=torch.einsum('bgsw,bwh->bgsh',a,pv)
        query=q[None].expand(B,-1,-1,-1)
        sim=F.cosine_similarity(ctx,query,dim=-1)
        seg_feat=torch.cat([ctx,query,sim[...,None],
                            a.max(-1).values[...,None],
                            logits.amax(-1)[...,None]],-1)
        seg_logits=self.segment_gate(seg_feat).squeeze(-1)
        seg_logits=seg_logits.masked_fill(~present[None],-1e4)
        weights=seg_logits.softmax(-1)
        pooled=(weights[...,None]*ctx).sum(2)
        qpool=(weights[...,None]*query).sum(2)
        diff = (ctx[:, :, 0] - ctx[:, :, 1]).square().mean(-1).clamp_min(1e-8).sqrt()
        extras=torch.stack([base,sim.max(-1).values,
                            (sim*weights).sum(-1),diff,
                            weights[:,:,0]],-1)
        delta=self.out(torch.cat([pooled,qpool,extras],-1)).squeeze(-1)
        return self.residual_scale*base+delta

def full_score(model, windows, valid, go_segments, present, base, chunk=128):
    outs=[]
    for st in range(0,len(go_segments),chunk):
        ed=min(st+chunk,len(go_segments))
        outs.append(model(windows,valid,go_segments[st:ed],present[st:ed],base[:,st:ed]))
    return torch.cat(outs,dim=1)

from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class Config:
    dim: int = 768
    proj: int = 128
    hidden: int = 128
    dropout: float = .1

class SemanticOntologyModel(nn.Module):
    def __init__(self, graph, cfg=Config()):
        super().__init__()
        self.cfg=cfg
        self.p_proj=nn.Sequential(nn.LayerNorm(cfg.dim),nn.Linear(cfg.dim,cfg.proj),nn.GELU())
        self.g_proj=nn.Sequential(nn.LayerNorm(cfg.dim),nn.Linear(cfg.dim,cfg.proj),nn.GELU())
        self.semantic=nn.Sequential(nn.Linear(cfg.proj*4+3,cfg.hidden),nn.GELU(),nn.Dropout(cfg.dropout),nn.Linear(cfg.hidden,cfg.hidden),nn.GELU(),nn.Linear(cfg.hidden,1))
        # Target, parent/child means and maxima, counts, relative disagreement and missing flags
        self.ontology=nn.Sequential(nn.Linear(11,cfg.hidden//2),nn.GELU(),nn.Linear(cfg.hidden//2,1))
        self.gate=nn.Sequential(nn.Linear(11,cfg.hidden//2),nn.GELU(),nn.Linear(cfg.hidden//2,1))
        nn.init.zeros_(self.ontology[-1].weight)
        nn.init.zeros_(self.ontology[-1].bias)
        self.register_buffer('parent_adj',graph[0].float()) # [G,G] row target, col parent
        self.register_buffer('child_adj',graph[1].float())

    @staticmethod
    def aggregate(x, adj):
        cnt=adj.sum(-1)
        mean=(x @ adj.T)/cnt.clamp_min(1)[None,:]
        # Max: vectorized masked broadcast, only for G <= ~2000; chunked to cap memory
        blocks=[]
        for lo in range(0,adj.shape[0],128):
            mask=adj[lo:lo+128].bool()
            vals=x[:,None,:].masked_fill(~mask[None,:,:],float('-inf')).amax(-1)
            blocks.append(torch.where(mask.any(-1)[None,:],vals,torch.zeros_like(vals)))
        mx=torch.cat(blocks,dim=1)
        return mean,mx,cnt

    def forward(self,p,g,s,return_details=False):
        # p [B,D], g [G,D], s [B,G]
        B,G=s.shape
        if G!=self.parent_adj.shape[0] or g.shape[0]!=G: raise ValueError('GO graph/score mismatch')
        pp=self.p_proj(p.float()); gg=self.g_proj(g.float())
        pv=pp[:,None,:].expand(-1,G,-1); gv=gg[None,:,:].expand(B,-1,-1)
        # rank is a relative feature, not a candidate truncation
        order=torch.argsort(s,dim=1,descending=True)
        ranks=torch.empty_like(s).scatter_(1,order,torch.arange(G,device=s.device,dtype=s.dtype)[None,:].expand(B,-1)) / max(G-1,1)
        centered=(s-s.mean(1,keepdim=True))/(s.std(1,keepdim=True).clamp_min(1e-5))
        feats=torch.cat((pv,gv,(pv-gv).abs(),pv*gv,s[...,None],centered[...,None],ranks[...,None]),-1)
        sem=self.semantic(feats).squeeze(-1)+s
        pm,px,pc=self.aggregate(sem,self.parent_adj)
        cm,cx,cc=self.aggregate(sem,self.child_adj)
        features=torch.stack((sem,pm,px,cm,cx,torch.log1p(pc)[None,:].expand(B,-1),torch.log1p(cc)[None,:].expand(B,-1),px-sem,cx-sem,(pc>0).float()[None,:].expand(B,-1),(cc>0).float()[None,:].expand(B,-1)),dim=-1)
        delta=self.ontology(features).squeeze(-1)
        gate=torch.sigmoid(self.gate(features).squeeze(-1))
        final=sem+gate*delta
        if return_details: return {'final':final,'semantic':sem,'delta':delta,'gate':gate}
        return final,sem

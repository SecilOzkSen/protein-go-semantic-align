import torch
from torch import nn

class EmbSetCal(nn.Module):
    """Permutation-equivariant Transformer producing direct protein-GO logits."""
    def __init__(self, protein_dim=1280, go_dim=768, hidden=128, heads=4, correction=0.75):
        super().__init__()
        self.correction = correction  # retained constructor argument for compatibility, unused
        self.protein_proj = nn.Sequential(nn.LayerNorm(protein_dim), nn.Linear(protein_dim, hidden), nn.GELU())
        self.go_proj = nn.Sequential(nn.LayerNorm(go_dim), nn.Linear(go_dim, hidden), nn.GELU())
        self.input_proj = nn.Sequential(nn.Linear(hidden*4+1, hidden), nn.GELU())
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden*4, dropout=0.1, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.delta_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden,hidden), nn.GELU(), nn.Linear(hidden,1))

    def forward(self, p, g, score):
        p = self.protein_proj(p).unsqueeze(1).expand(-1,g.size(1),-1)
        g = self.go_proj(g)
        h = self.input_proj(torch.cat([p,g,abs(p-g),p*g,score.unsqueeze(-1)],dim=-1))
        return self.delta_head(self.encoder(h)).squeeze(-1)

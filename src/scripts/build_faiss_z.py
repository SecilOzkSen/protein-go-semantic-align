import os, json
import torch
import faiss
import numpy as np

def l2n(x: torch.Tensor, dim=-1, eps=1e-12):
    return x / (x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps))

@torch.no_grad()
def build_faiss_in_z(
    go_cache_path: str,          # torch.load(...) edilen cache: {'embs': [N,d_g], 'row2id': list/int, 'id2row': dict}
    proj_g,                      # callable: proj_g(G:[N,d_g])->[N,d_z]  (örn. model.proj_g veya aligner.W_g)
    out_index_path: str,         # yeni index dosyası (örn. .../go_faiss_z.faiss)
    out_meta_path: str,          # meta.json (phase bilgisi vs.)
    phase_tag: str,              # 'phase_1' gibi
    to_gpu: bool = False
):
    # 1) GO cache yükle
    cache = torch.load(go_cache_path, map_location="cpu")
    G = cache["embs"]                     # [N, d_g]  (torch.Tensor)
    row2id = cache.get("row2id", None)
    id2row = cache.get("id2row", None)
    assert isinstance(G, torch.Tensor) and G.dim() == 2
    N, d_g = G.shape

    # 2) Projeksiyon -> z
    G = G.float()
    Z = proj_g(G)                         # [N, d_z]
    assert isinstance(Z, torch.Tensor) and Z.dim() == 2
    Z = l2n(Z, dim=1).cpu().contiguous()
    N, d_z = Z.shape

    # 3) FAISS index (cosine ~ dot for L2-normalized)
    index = faiss.IndexFlatIP(d_z)
    Z_np = np.ascontiguousarray(Z.numpy(), dtype=np.float32)
    index.add(Z_np)
    assert index.ntotal == N

    # 4) Kaydet
    faiss.write_index(index, out_index_path)

    meta = {
        "phase": phase_tag,
        "space": "z",
        "d_g": int(d_g),
        "d_z": int(d_z),
        "ntotal": int(N),
        "row2id_exists": row2id is not None,
        "id2row_exists": id2row is not None,
    }
    with open(out_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✔ wrote index: {out_index_path} (d={d_z}, N={N})")
    print(f"✔ wrote meta : {out_meta_path}")

# --------- ÖRNEK KULLANIM ---------
if __name__ == "__main__":
    # Buraya kendi yollarını ve projeksiyon fonksiyonunu bağla
    GO_CACHE = "/path/to/phaseX/embeddings.pt"      # {'embs','row2id','id2row'}
    OUT_INDEX = "/path/to/phaseX/go_faiss_z.faiss"
    OUT_META  = "/path/to/phaseX/meta_z.json"
    PHASE_TAG = "phase_1"

    # Model/aligner içinden GO projeksiyonu: proj_g: [N,d_g] -> [N,d_z]
    # Örn: aligner.proj_g = nn.Linear(d_g, d_z, bias=False)
    from src.models.alignment_model import ProteinGoAligner
    aligner = ProteinGoAligner(d_h=1024, d_g=768, d_z=512, go_encoder=None, normalize=True)
    proj_g = aligner.proj_g  # nn.Linear; callable gibi kullanılabilir

    build_faiss_in_z(GO_CACHE, proj_g, OUT_INDEX, OUT_META, PHASE_TAG)
# End to End Pipeline (main.py)


## 1) Inputs and Caches

1. **GO embedding cache** is loaded and L2-normalized (for FAISS consistency). Each GO term has a fixed vector and ID↔row mapping. A `GoLookupCache(blob)` layer wraps access.
    
2. **FAISS index** is loaded for the current training phase, enabling efficient mining of hard/near negatives.
    
3. **ESM embedding store** is built (`ESMResidueStore`), which manages residue-level ESM embeddings, sequence lengths, and optional shard cache/manifest.
    

---

## 2) Dataset Construction (Few/Zero-shot splits and DAG expansion)

4. Training/validation protein IDs and **positive GO labels** (`pid2pos`) are loaded. Zero-shot, few-shot, and common term sets are packed into a **FewZeroConfig** (e.g., controlling the target ratio of few-shot terms).
    
5. Optionally, **DAG parents** and an **ancestor stoplist** are used to expand or filter labels (e.g., extend sparse positives upward, cap at maximum ancestors, apply gamma weighting).
    
6. `ProteinEmbDataset` prepares individual records (protein → positive GO set), combined with GO cache, DAG parents, few/zero configs, and the ESM store. The purpose: provide training-ready positive pairs and metadata for negative mining.
    

---

## 3) Batch Creation with Curriculum Mining

7. A **zero-shot mask** tensor is built to exclude disallowed terms during training.
    
8. `ContrastiveEmbCollator` constructs **bidirectional batches** (protein→GO, GO→protein). Positives are formed via cache; negatives are mined using FAISS plus curriculum parameters (hard/easy mix).
    
9. `CurriculumConfig` interpolates over training steps: `hard_frac`, `shortlist_M`, `k_hard`, `random_k`, `inbatch_easy`, `hier_up/down`. This ramps up difficulty gradually. W&B can log preview tables of these schedules.
    

---

## 4) Forward Pass - Semantic Alignment

10. **Protein side**: residue-level ESM embeddings are reduced to a protein vector via **attention/pooling** (with sliding-window support, `win_size`, `win_stride`).
    
11. **GO side**: embeddings are directly retrieved from the cache. GO terms are represented as **text embeddings**, enabling zero-shot transfer.
    
12. Both embeddings are projected into the same **alignment dimension** (e.g., 768). Cosine similarity with temperature scaling produces a score matrix (positives high, negatives low).
    

---

## 5) Losses

13. **Contrastive loss (InfoNCE)**: pulls positive protein-GO pairs together, pushes negatives apart (negatives mined via FAISS + curriculum).
    
14. **DAG-aware regularization (LDAG)**: enforces hierarchical consistency by keeping GO parent-child embeddings close.
    
15. **Attribution loss (Lattr)** - **early phases only**: enforces functional fidelity of residue attentions by aligning them with masking-based impact. Curriculum scheduling increases λ\_attr gradually.
    

**Total loss:**

$$
L = \lambda_{con} L_{contrastive} + \lambda_{attr} L_{attribution} + \lambda_{DAG} L_{DAG}
$$

(Attribution is phased in early; DAG applies throughout.)

---

## 6) Training Loop, Optimization, and Monitoring

16. Each step: forward pass → compute losses → `opt.zero_grad()` → `loss.backward()` → (optional grad clipping) → `opt.step()`. Running averages and global step are tracked.
    
17. **Evaluation at epoch end**: reports contrastive, DAG, attribution, entropy, and total losses. Logs are pushed to **W&B**.
    
18. **YAML-based configs** control everything: ID lists, cache paths, optimizer/training hyperparameters, model/attention settings, and multi-stage schedules (Stage A–D with `lambda_attr_start/max`). This ensures reproducible experiments.
    

---

## 7) Inference

19. After training, inference works as **retrieval**: a protein embedding is compared against all cached GO embeddings via cosine similarity; the top terms are predictions. This supports **zero-shot/few-shot generalization** over large vocabularies.
    
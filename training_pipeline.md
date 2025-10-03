# Training Pipeline (with Intuition, Mechanism, Effect)

## 1) Overview

**Intuition:** Protein function prediction is complex: proteins are long sequences, GO terms are structured, and negatives are abundant. A successful pipeline must (i) embed proteins and GO terms into a common space, (ii) progressively increase task difficulty, and (iii) ensure interpretability.

**How it works:**

-   Proteins are encoded by ESM residue embeddings.
    
-   GO terms are encoded by a text encoder with ontology tokens.
    
-   A trainer aligns them using contrastive + auxiliary losses.
    
-   Difficulty ramps up through two curricula: negative mining and ontology-aware phases.
    

**Effect:** Stable convergence, interpretable predictions, and improved generalization to unseen proteins and GO terms.

---

## 2) Data & Stores

### 2.1 ESM Residue Store

**Intuition:** Proteins exceed LM input limits. Overlaps are needed to cover them, but raw concatenation breaks residue alignment.

**How it works:** Residues are embedded with a sliding-window ESM model, stitched together, and overlaps trimmed to restore true sequence length.

**Effect:** Every residue is represented without loss, enabling downstream interpretability (residue-level attribution).

---

### 2.2 GO Vector Cache

**Intuition:** Negative sampling requires a large, searchable repository of GO embeddings.

**How it works:** Pre-computed GO embeddings are stored, normalized, and indexed in FAISS for efficient retrieval. The cache is mirrored into a MemoryBank for dynamic updates.

**Effect:** Enables scalable candidate mining with both static and dynamically refreshed negatives.

---

### 2.3 GO Text Store (Phase-Aware)

**Intuition:** GO terms encode relationships (“is\_a”, “part\_of”), but injecting all relation tokens at once destabilizes learning.

**How it works:** GO texts are phase-specific. In early phases, only plain definitions are used; later phases introduce `[IS_A]`, `[PART_OF]`, etc. `GoTextStore.update_phase_and_tokenize` re-encodes texts at each transition.

**Effect:** Smoothly increases semantic difficulty, letting the model adapt to richer ontology structure without collapse.

---

## 3) Datasets & Collation

**Intuition:** Each batch must teach the model to discriminate positives from hard negatives.

**How it works:**

-   **ProteinEmbDataset** links proteins to positives via `pid2pos` and optional DAG expansion.
    
-   **ContrastiveEmbCollator** collects positives and negatives, constructs candidate sets, and returns masks + unique GO ids.
    

**Effect:** Provides the trainer with clean, balanced mini-problems that scale across training.

---

## 4) Retrieval & Mining

### 4.1 VectorResources

**Intuition:** FAISS must search the same embedding space that the trainer updates. Any misalignment breaks negative mining.

**How it works:** Wraps FAISS + embeddings; `set_backends` ensures both use the same source (cache or MemoryBank).

**Effect:** Keeps retrieval consistent, preventing stale or mismatched negatives.

---

### 4.2 BatchBuilder (DAG-aware)

**Intuition:** Ancestors and descendants of a positive GO are trivial negatives; siblings are subtle. Mining must respect ontology.

**How it works:**

1.  Retrieve shortlist from FAISS (`shortlist_M`).
    
2.  Apply DAG masks (exclude ancestors/descendants).
    
3.  Select hard negatives (`k_hard`).
    
4.  Mix in easy negatives (`random_k`, in-batch).
    
5.  Optionally add siblings via weighted queues.
    

**Effect:** Produces a curriculum of negatives ranging from easy to hard, aligned with ontology structure.

---

## 5) Trainer & Losses

**Intuition:** A single contrastive loss is not enough; auxiliary signals guide learning and enforce interpretability.

**How it works:**

1.  **Contrastive loss (InfoNCE):** pushes proteins toward positives, away from negatives.
    
2.  **Teacher distillation:** stabilizes early learning by aligning student predictions with teacher scores (`KL-divergence` + τ temperature).
    
3.  **Attribution loss:** residue masking forces the model to identify residues critical for each GO.
    
4.  **Entropy regularization:** penalizes overconfident predictions to preserve diversity.
    

**Effect:** Balanced optimization: discriminative, stable, interpretable, and generalizable representations.

---

## 6) Curriculum Scheduler (Negative Mining)

**Intuition:** Learning progresses best when negatives become harder gradually, not all at once.

**How it works:**

-   Interpolates `hard_frac`, `shortlist_M`, `k_hard`, DAG mask radii, `random_k`, in-batch usage, and sibling toggles across steps.
    
-   Uses cosine/linear interpolation with warm-up.
    

**Effect:**

-   Early training: easy/random negatives, strict masks.
    
-   Mid training: more hard negatives, relaxed masks.
    
-   Late training: mostly hard negatives, siblings included.  
    The model gradually masters subtle functional distinctions.
    

---

## 7) Phase Management (Ontology Tokens & Reload)

**Intuition:** Semantic difficulty should also increase gradually.

**How it works:**

-   At scheduled epochs, `maybe_refresh_phase_resources` reloads: GO cache, FAISS index, MemoryBank, VectorResources, and DataLoaders.
    
-   `GoTextStore.update_phase_and_tokenize` introduces new ontology tokens.
    
-   Between phases, MemoryBank is partially refreshed with GO ids seen in the previous epoch.
    

**Effect:**

-   Early: plain GO definitions.
    
-   Later: relation tokens (`[IS_A]`, `[PART_OF]`) enrich semantics.
    
-   The dual curriculum ensures difficulty rises both on the **negative mining axis** and the **semantic axis**.
    

---

## 8) Training Flow

**Intuition:** Stability comes from combining per-step updates with per-epoch refreshes.

**How it works:**

1.  Initialize resources (phase 0 cache, FAISS, GO text store, datasets, MemoryBank, trainer).
    
2.  Each step: mine negatives via scheduler, compute losses, backprop.
    
3.  Each epoch: refresh MemoryBank, check phase transitions, validate, checkpoint.
    

**Effect:** Orchestrates short-term learning (steps) with long-term progression (epochs + phases), ensuring convergence and robustness.

---

## 9) Checkpoints & Safety

**Intuition:** Frequent reloads (phases, FAISS rebuilds) risk instability; checkpoints guard progress.

**How it works:** Saves step/epoch/best/final checkpoints; prunes old ones; logs refresh reasons. Fallback paths (cache embeddings if FAISS rebuild fails).

**Effect:** Resilience to interruptions and safe recovery at any training stage.

---

## 10) Practical Defaults

**Intuition:** Baselines should prevent collapse while leaving headroom for tuning.

**How it works:** Defaults:

-   Curriculum: cosine, warm-up ≈1 epoch, `hard_frac 0.2→0.8`, `k_hard 16→64`, masks relax, siblings later.
    
-   Losses: λ\_con=1.0, λ\_attr=0.1, entropy≈0.01, τ=1.5, λ\_vtrue=0.2.
    

**Effect:** Provides a safe, reproducible starting point for experiments.

---
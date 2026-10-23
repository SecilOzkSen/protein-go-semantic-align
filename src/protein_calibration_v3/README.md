# Protein calibration V3

One protein-conditioned affine head, frozen retriever, score statistics + frozen raw ESM protein embedding. No cardinality prediction. Trains with a batch CAFA-style soft F surrogate, evaluates with the existing exact CAFA functions. A batch surrogate is not mathematically identical to dataset-level Fmax.

V2 NPZ files can be reused without re-extraction. Do not overwrite V1/V2.

```bash
cd /workspace/protein-go-semantic-align
python -m src.protein_calibrationv3.test_smoke
BASE=/workspace/data_pfresgo/outputs/bp/calibration_study/infonce
python -m src.protein_calibrationv3.fit \
  --calibration "$BASE/calibrationv2/calibration.npz" \
  --valid "$BASE/calibrationv2/valid.npz" \
  --out-dir "$BASE/calibrationv3/cafa_soft_f_raw_esm" --seed 42
python -m src.protein_calibrationv3.evaluate \
  --test "$BASE/calibrationv2/test.npz" \
  --head-dir "$BASE/calibrationv3/cafa_soft_f_raw_esm" \
  --out "$BASE/calibrationv3/cafa_soft_f_raw_esm/test_results.json"
```

**Protocol:** Train on calibration-only proteins; choose head on validation soft surrogate; choose global threshold on validation exact Fmax; test threshold stays fixed. Existing test has already been inspected during V1/V2 exploration, so V3 test is exploratory rather than a new untouched benchmark.

**Caveat:** soft sigmoid assigns nonzero predicted mass to every protein; the surrogate approximates CAFA precision averaging and is not an exact differentiable implementation of the hard threshold Fmax sweep. Minibatch optimization adds approximation error. GO propagation and information-content weighting are not added; exact metrics reuse project `src.metrics.cafa`.

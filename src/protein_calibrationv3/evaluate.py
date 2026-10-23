"""Evaluate fixed validation-selected thresholds on untouched test."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .model import ProteinCalibrationV3
from .io import load_matrix, hash_go_ids
from .fit import predict
from src.metrics.cafa import compute_protein_centric_fmax, protein_centric_f_at_threshold


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--test', required=True)
    p.add_argument('--head-dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    test = load_matrix(a.test)
    directory = Path(a.head_dir)
    report = json.loads((directory/'validation_results.json').read_text())
    ckpt = torch.load(directory/'head.pt', map_location='cpu', weights_only=False)
    if hash_go_ids(test['go_ids']) != ckpt['go_hash'] or ckpt['go_hash'] != report['go_id_hash']:
        raise ValueError('Test GO ordering does not match calibration model')
    model = ProteinCalibrationV3(protein_dim=test['protein_embeddings'].shape[1]).to(a.device)
    model.load_state_dict(ckpt['model'], strict=True)
    calibrated = predict(model, test['scores'].astype('float32'), test['protein_embeddings'].astype('float32'), torch.device(a.device))
    raw_threshold = report['raw_valid_threshold']
    cal_threshold = report['calibrated_valid_threshold']
    results = {
        'raw_test_F_at_validation_threshold': protein_centric_f_at_threshold(test['labels'], test['scores'], raw_threshold),
        'calibrated_test_F_at_validation_threshold': protein_centric_f_at_threshold(test['labels'], calibrated['scores'], cal_threshold),
        'raw_validation_threshold': raw_threshold,
        'calibrated_validation_threshold': cal_threshold,
        'raw_test_oracle_Fmax_diagnostic_only': compute_protein_centric_fmax(test['labels'], test['scores'])[0],
        'calibrated_test_oracle_Fmax_diagnostic_only': compute_protein_centric_fmax(test['labels'], calibrated['scores'])[0],
    }
    path = Path(a.out); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

if __name__ == '__main__': main()

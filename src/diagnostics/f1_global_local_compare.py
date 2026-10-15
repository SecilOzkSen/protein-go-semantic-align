"""Compare global/local GO discrimination on identical terms and proteins.
Interpret local results as a non-causal diagnostic, not architecture proof.
"""
import argparse, json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def metrics(data, min_pos=3):
    y = data['labels'].astype(bool)
    result = {}
    for name, key in [('global', 'global_scores'), ('local_max', 'local_scores')]:
        scores = data[key]
        aucs, aps = [], []
        for j in range(y.shape[1]):
            if y[:, j].sum() < min_pos or (~y[:, j]).sum() < min_pos:
                continue
            aucs.append(roc_auc_score(y[:, j], scores[:, j]))
            aps.append(average_precision_score(y[:, j], scores[:, j]))
        # Per-protein retrieval is not threshold dependent
        r50 = []
        for i in range(len(y)):
            positive = int(y[i].sum())
            if positive:
                top = np.argpartition(-scores[i], min(50, len(scores[i]) - 1))[:50]
                r50.append(float(y[i, top].sum() / positive))
        result[name] = {'macro_GO_AUC': float(np.mean(aucs)),
                        'macro_GO_AP': float(np.mean(aps)),
                        'protein_mean_R_at_50': float(np.mean(r50)),
                        'GO_terms_evaluated': len(aucs)}
    return result


p = argparse.ArgumentParser()
p.add_argument('--valid', required=True)
p.add_argument('--test', required=True)
p.add_argument('--min_pos', type=int, default=3)
a = p.parse_args()
v = np.load(a.valid);
t = np.load(a.test)
if not np.array_equal(v['go_ids'], t['go_ids']):
    raise ValueError('GO ordering differs')
print(json.dumps({'validation': metrics(v, a.min_pos), 'test': metrics(t, a.min_pos),
                  'warning': 'Different GO sets may pass min_pos in each split; compare per-split methods, not naive val-test macro gaps. Window projection is out of training distribution. MaxSim changes score distribution and benefits from multiple comparisons.'},
                 indent=2))

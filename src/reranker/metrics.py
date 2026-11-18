import importlib.util
from pathlib import Path
import torch
import numpy as np

def evaluate(labels, scores, repo='/workspace/MZSGO', tag='embsetcal_val'):
    path=Path(repo)/'utils'/'util.py'
    if not path.is_file(): raise FileNotFoundError(path)
    spec=importlib.util.spec_from_file_location('embsetcal_mzsgo_original_metrics',path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    f,p,r,a,t=mod.evaluate_annotations(torch.from_numpy(np.asarray(labels,dtype=np.int32)),torch.from_numpy(np.asarray(scores,dtype=np.float32)),tag)[:5]
    return {'Fmax':float(f),'precision':float(p),'recall':float(r),'AUPR':float(a),'threshold':float(t)}

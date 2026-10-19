
import numpy as np
import torch
from .model import OffsetHead, features, normalize_features

def load_calibrator(path):
    obj=torch.load(path,map_location='cpu',weights_only=False)
    head=OffsetHead(hidden=obj['hidden'],max_offset=obj['max_offset'])
    head.load_state_dict(obj['state_dict'])
    head.eval()
    return head,obj

@torch.no_grad()
def calibrate(raw_scores,head,config):
    scores=np.asarray(raw_scores,dtype=np.float32)
    f=normalize_features(features(scores),config['feature_mean'],config['feature_std'])
    offsets=head(torch.from_numpy(f)).numpy()
    return scores+offsets[:,None]

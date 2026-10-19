
import numpy as np

def protein_f(y, s, t):
    y=np.asarray(y, dtype=bool)
    pred=np.asarray(s)>=float(t)
    tp=(pred & y).sum(1)
    npred=pred.sum(1)
    ntrue=y.sum(1)
    haspred=npred>0
    hastrue=ntrue>0
    precision=float(np.mean(tp[haspred]/npred[haspred])) if haspred.any() else 0.
    recall=float(np.mean(tp[hastrue]/ntrue[hastrue])) if hastrue.any() else 0.
    return 2*precision*recall/(precision+recall) if precision+recall else 0.

def exact_fmax(y, s):
    """Exact CAFA-style macro precision/recall harmonic mean, raw score thresholds."""
    y=np.asarray(y,dtype=bool)
    s=np.asarray(s,dtype=np.float32)
    if y.shape!=s.shape or y.ndim!=2 or not np.isfinite(s).all():
        raise ValueError("Invalid labels/scores")
    n,g=y.shape
    if n==0 or not y.any(): return 0.,0.
    order=np.argsort(-s,axis=1,kind='stable')
    sy=np.take_along_axis(y,order,axis=1)
    ss=np.take_along_axis(s,order,axis=1)
    tp=np.cumsum(sy,axis=1,dtype=np.int32)
    pc=tp/np.arange(1,g+1,dtype=np.float64)[None,:]
    dp=np.diff(pc,axis=1,prepend=np.zeros((n,1)))
    ntrue=y.sum(1)
    dr=np.zeros((n,g),dtype=np.float64)
    active=ntrue>0
    dr[active]=sy[active]/ntrue[active,None]
    first=np.zeros((n,g),dtype=np.int8)
    first[:,0]=1
    idx=np.argsort(-ss.ravel(),kind='stable')
    sorted_s=ss.ravel()[idx]
    starts=np.r_[0,np.flatnonzero(sorted_s[1:]!=sorted_s[:-1])+1]
    p_sum=np.cumsum(np.add.reduceat(dp.ravel()[idx],starts))
    r_sum=np.cumsum(np.add.reduceat(dr.ravel()[idx],starts))
    n_pred=np.cumsum(np.add.reduceat(first.ravel()[idx],starts))
    pmean=np.divide(p_sum,n_pred,out=np.zeros_like(p_sum),where=n_pred>0)
    rmean=r_sum/active.sum()
    f=np.divide(2*pmean*rmean,pmean+rmean,out=np.zeros_like(pmean),where=(pmean+rmean)>0)
    best=int(np.argmax(f))
    return float(f[best]),float(sorted_s[starts[best]])

def load_scores(path):
    d=np.load(path,allow_pickle=False)
    if 'scores' not in d or 'labels' not in d:
        raise ValueError("NPZ requires scores and labels")
    s=np.asarray(d['scores'],dtype=np.float32)
    y=np.asarray(d['labels'],dtype=np.float32)
    if s.shape!=y.shape or s.ndim!=2 or not np.isfinite(s).all():
        raise ValueError("Invalid NPZ matrices")
    if not np.isin(y,[0,1]).all(): raise ValueError("Labels must be binary")
    ids=np.asarray(d['go_ids']) if 'go_ids' in d else None
    return s,y,ids

import argparse, csv, json, os, subprocess, sys
from pathlib import Path

RUNNER = r"""
import os,sys,json,runpy
import numpy as np
from pathlib import Path
from src.training.trainer import OppTrainer

def compute(scores,labels):
    s=np.concatenate(scores).astype(np.float64)
    y=np.concatenate(labels).astype(bool)
    n,g=s.shape
    counts=y.sum(axis=1)
    if not np.all((counts>0)&(counts<g)): raise ValueError("Empty/all-positive labels")
    order=np.argsort(-s,axis=1,kind="stable")
    ordered=np.take_along_axis(y,order,axis=1)
    cum=np.cumsum(ordered,axis=1)
    ranks=np.arange(1,g+1)[None,:]
    ap=((cum/ranks)*ordered).sum(axis=1)/counts
    pbc=np.mean([ordered[i,:counts[i]].mean() for i in range(n)])
    auc=[]
    for i in range(n):
        neg=np.sort(s[i,~y[i]])
        pos=s[i,y[i]]
        lo=np.searchsorted(neg,pos,"left")
        hi=np.searchsorted(neg,pos,"right")
        auc.append(np.mean((lo+.5*(hi-lo))/len(neg)))
    return dict(n_proteins=n,n_go_terms=g,pairwise_accuracy=float(np.mean(auc)),pbc=float(pbc),protein_wise_map=float(np.mean(ap)))

original=OppTrainer.eval_epoch
def patched(self,loader,epoch_idx):
    ss=[];yy=[]
    score_fn=self._score_full_go
    def capture(batch):
        scores,weights=score_fn(batch)
        ss.append(scores.detach().float().cpu().numpy())
        yy.append(self._positive_mask(batch).detach().cpu().numpy())
        return scores,weights
    self._score_full_go=capture
    try: result=original(self,loader,epoch_idx)
    finally: self._score_full_go=score_fn
    if not ss: raise RuntimeError("No evaluation scores captured")
    report=compute(ss,yy)
    Path(os.environ["METRIC_OUT"]).write_text(json.dumps(report,indent=2))
    print("[METRIC SWEEP]",report,flush=True)
    return result
OppTrainer.eval_epoch=patched
sys.argv=["src.main"]+sys.argv[1:]
runpy.run_module("src.main",run_name="__main__")
"""


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--checkpoint-dir", required=True)
    a.add_argument("--config", required=True, help="Eval-only validation YAML")
    a.add_argument("--out", required=True)
    a.add_argument("--include-special", action="store_true")
    a.add_argument("--continue-on-error", action="store_true")
    args = a.parse_args()
    ckptdir = Path(args.checkpoint_dir)
    ckpts = sorted(ckptdir.glob("checkpoint_epoch*.pt"), key=lambda x: int(x.stem.split("epoch")[-1]))
    if args.include_special:
        ckpts += [ckptdir / x for x in ["best_primary.pt", "best_macro.pt", "checkpoint_final.pt"] if (ckptdir / x).is_file()]
    if not ckpts: raise SystemExit("No checkpoints found")
    out = Path(args.out);
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ck in ckpts:
        dest = out / (ck.stem + ".json")
        if dest.exists(): dest.unlink()
        env = os.environ.copy();
        env["METRIC_OUT"] = str(dest.resolve())
        cmd = [sys.executable, "-c", RUNNER, "--config", args.config, "--resume", str(ck.resolve())]
        print("Evaluating", ck.name, flush=True)
        with (out / (ck.stem + ".log")).open("w") as log:
            proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode or not dest.exists():
            print("FAILED", ck.name, "see log", flush=True)
            if not args.continue_on_error: raise SystemExit(1)
            continue
        row = {"checkpoint": ck.name, **json.loads(dest.read_text())}
        rows.append(row)
        with (out / "results.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row));
            w.writeheader();
            w.writerows(rows)
        print(ck.name, {k: round(row[k], 5) for k in ("pairwise_accuracy", "pbc", "protein_wise_map")}, flush=True)
    if not rows: raise SystemExit("No successful evaluations")
    best = {metric: max(rows, key=lambda x: x[metric]) for metric in ("pairwise_accuracy", "pbc", "protein_wise_map")}
    (out / "best_by_metric.json").write_text(json.dumps(best, indent=2))
    print("BEST", json.dumps({k: (v["checkpoint"], v[k]) for k, v in best.items()}, indent=2))


if __name__ == "__main__": main()

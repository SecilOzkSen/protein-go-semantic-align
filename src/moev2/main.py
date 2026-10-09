import argparse,json,logging,random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import load_dump,DumpDataset,graph_matrices
from .model import Config,SemanticOntologyModel
from src.loss.asymmetric_loss import AsymmetricLoss,AsymmetricLossConfig
from src.metrics.stargo_pfresgo_metrics import StarGOPFresGOEvaluator

def main():
    ap=argparse.ArgumentParser()
    for arg in ('train_dump','val_dump','go_graph_path','out_dir'): ap.add_argument('--'+arg,required=True,type=Path)
    ap.add_argument('--ontology',default='bp');ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--batch_size',type=int,default=16);ap.add_argument('--eval_batch_size',type=int,default=32)
    ap.add_argument('--epochs',type=int,default=12);ap.add_argument('--patience',type=int,default=3)
    ap.add_argument('--lr',type=float,default=2e-4);ap.add_argument('--lambda_sem',type=float,default=.2)
    ap.add_argument('--gamma_pos',type=float,default=0);ap.add_argument('--gamma_neg',type=float,default=4);ap.add_argument('--asl_clip',type=float,default=.05)
    ap.add_argument('--log_every',type=int,default=500);ap.add_argument('--check_only',action='store_true');ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    tr=load_dump(a.train_dump);va=load_dump(a.val_dump)
    if not np.array_equal(tr['go'],va['go']) or not np.allclose(tr['z'],va['z'],atol=.003): raise RuntimeError('GO ID/embedding mismatch')
    if set(tr['ids'])&set(va['ids']): raise RuntimeError('Train/validation overlap')
    parents,children=graph_matrices(tr['go'],a.go_graph_path)
    logging.info('CONTRACT train=%d val=%d G=%d DAG_edges=%d',len(tr['ids']),len(va['ids']),len(tr['go']),int(parents.sum()))
    if a.check_only:return
    a.out_dir.mkdir(parents=True,exist_ok=True)
    device=torch.device(a.device); cfg=Config(dim=tr['z'].shape[1]); model=SemanticOntologyModel((parents,children),cfg).to(device)
    gz=torch.from_numpy(tr['z']).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr)
    loss_fn=AsymmetricLoss(AsymmetricLossConfig(gamma_pos=a.gamma_pos,gamma_neg=a.gamma_neg,clip=a.asl_clip))
    train_dl=DataLoader(DumpDataset(tr),batch_size=a.batch_size,shuffle=True,num_workers=0)
    val_dl=DataLoader(DumpDataset(va),batch_size=a.eval_batch_size,shuffle=False,num_workers=0)
    evaluator=StarGOPFresGOEvaluator(goterms=tr['go'],ontology=a.ontology,go_graph_path=a.go_graph_path)
    # Exact step-zero semantic/final parity, before training
    model.eval()
    with torch.no_grad():
        p,s,y=next(iter(val_dl));out=model(p.to(device),gz,s.to(device),True)
        if not torch.equal(out['final'],out['semantic']): raise RuntimeError('Step-0 residual parity FAILED')
    logging.info('STEP0 parity PASSED')
    best=-1; bad=0;step=0
    for epoch in range(a.epochs):
        model.train(); running=0
        for p,s,y in train_dl:
            p,s,y=p.to(device),s.to(device),y.to(device)
            opt.zero_grad(set_to_none=True)
            o=model(p,gz,s,True)
            lf=loss_fn(o['final'],y);ls=loss_fn(o['semantic'],y)
            loss=lf+a.lambda_sem*ls
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            step+=1;running+=loss.item()
            if step%a.log_every==0: logging.info('train epoch=%d step=%d loss=%.6f final=%.6f semantic=%.6f gate=%.4f',epoch,step,loss.item(),lf.item(),ls.item(),o['gate'].mean().item())
        model.eval();preds=[];sems=[];labels=[];gates=[]
        with torch.no_grad():
            for p,s,y in val_dl:
                o=model(p.to(device),gz,s.to(device),True)
                preds.append(torch.sigmoid(o['final']).cpu().numpy());sems.append(torch.sigmoid(o['semantic']).cpu().numpy());labels.append(y.numpy());gates.append(o['gate'].mean().item())
        labels=np.concatenate(labels); preds=np.concatenate(preds);sems=np.concatenate(sems)
        fm=evaluator.evaluate(labels,preds);sm=evaluator.evaluate(labels,sems)
        logging.info('VAL epoch=%d final_Fmax=%.6f semantic_Fmax=%.6f final_macroAUPR=%.6f semantic_macroAUPR=%.6f gate=%.4f',epoch,fm['protein_fmax'],sm['protein_fmax'],fm['macro_aupr'],sm['macro_aupr'],float(np.mean(gates)))
        payload={'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'meta':{'go_ids':tr['go'].tolist(),'final_metrics':fm,'semantic_metrics':sm,'config':vars(cfg),'lambda_sem':a.lambda_sem}}
        torch.save(payload,a.out_dir/'checkpoint_last.pt')
        if fm['protein_fmax']>best+1e-7:
            best=fm['protein_fmax'];bad=0;torch.save(payload,a.out_dir/'checkpoint_best.pt')
            (a.out_dir/'best_metrics.json').write_text(json.dumps({'epoch':epoch,'final':fm,'semantic':sm},indent=2))
        else: bad+=1
        if bad>=a.patience: logging.info('EARLY STOP');break
    logging.info('DONE best_Fmax=%.6f',best)
if __name__=='__main__':main()

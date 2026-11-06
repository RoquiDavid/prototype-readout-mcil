# Hyperparameter grid of the baselines.
import argparse, json, os, sys, itertools
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from sklearn.metrics import f1_score

sys.path.insert(0, '/data/roqui')
from models_meld import PerceiverIO_MoE
from meld_dataset import load_meld_class_incremental
from meld_baselines import Naive, EWC, LwF, ER, DERpp, AGEM, DMoLE, CLMoE

MODEL_CONFIG = {'num_classes':7,'d_model':64,'num_experts':5,'r':8,'alpha':16,'tau':0.2,
    'num_latents':4,'text_dim':600,'audio_dim':300,'video_dim':35,
    'use_text':True,'use_audio':True,'use_video':False}

def _grid(cls, keys, values):
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

GRIDS = {
    'EWC':   (EWC,   _grid(EWC,   ['lambda_ewc'], [[1.0,100.0,1000.0,10000.0]])),
    'LwF':   (LwF,   _grid(LwF,   ['lambda_kd','temperature'], [[0.5,1.0,2.0,5.0],[2.0]])),
    'ER':    (ER,    _grid(ER,    ['buffer_size'], [[200,500]])),
    'DER++': (DERpp, _grid(DERpp, ['buffer_size','alpha','beta'], [[200,500],[0.5,0.2],[0.5,0.8]])),
    'AGEM':  (AGEM,  _grid(AGEM,  ['buffer_size'], [[200,500]])),
}
SEEDS = [11, 50, 99]

def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    os.environ['PYTHONHASHSEED']=str(s)

def _labels(b,d): return torch.clamp(b['labels'].to(d)[:,0],-3,3).round().long()+3

def evaluate_split(model, loader, device):
    model.eval(); c=t=0
    with torch.no_grad():
        for b in loader:
            y=_labels(b,device)
            pred=model(b['text'].to(device),b['audio'].to(device),b['video'].to(device))['logits'].argmax(1)
            c+=(pred==y).sum().item(); t+=y.size(0)
    return c/t if t else 0.0

def cl_avg(M): return float(M[-1,:].mean())

def cfg_name(kw): return "_".join(f"{k}{v}" for k,v in kw.items()) if kw else "default"

def run_one(method, cls, kwargs, seed, results_dir, epochs_task0, epochs_per_task, lr):
    set_seed(seed)
    train_loaders, val_loaders, _ = load_meld_class_incremental(
        data_dir='/data/roqui/meld_data', batch_size=32, seed=seed)
    T=len(train_loaders)
    set_seed(seed)
    model=PerceiverIO_MoE(**MODEL_CONFIG).to('cuda')
    learner=cls(model,'cuda',**kwargs)
    M=np.zeros((T,T))
    for tid in range(T):
        ep = epochs_task0 if tid==0 else epochs_per_task
        learner.train_task(train_loaders[tid], val_loaders[tid], epochs=ep, lr=lr)
        for et in range(tid+1):
            M[tid,et]=evaluate_split(model,val_loaders[et],'cuda')
    acc=cl_avg(M)
    out=Path(results_dir)/method/cfg_name(kwargs)/f'seed{seed}'
    out.mkdir(parents=True,exist_ok=True)
    json.dump({'method':method,'kwargs':kwargs,'seed':seed,
               'val':{'avg_accuracy':acc,'accuracy_matrix':M.tolist()},
               'timestamp':datetime.now().isoformat()}, open(out/'result.json','w'), indent=2)
    return acc

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results_dir',type=str,default='/data/roqui/meld_grid_baselines')
    p.add_argument('--epochs_task0',type=int,default=30)
    p.add_argument('--epochs_per_task',type=int,default=15)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--methods',type=str,default='EWC,LwF,ER,DER++,AGEM')
    args=p.parse_args()

    methods=args.methods.split(',')
    summary={}
    for method in methods:
        cls,grid=GRIDS[method]
        print(f"\n{'='*70}\nGRID {method} — {len(grid)} configs x {len(SEEDS)} seeds\n{'='*70}")
        best=(-1,None)
        for kwargs in grid:
            accs=[]
            for seed in SEEDS:
                a=run_one(method,cls,kwargs,seed,args.results_dir,
                          args.epochs_task0,args.epochs_per_task,args.lr)
                accs.append(a)
                print(f"  {cfg_name(kwargs):<30} seed{seed}: {a*100:.2f}%")
            mean=float(np.mean(accs))
            print(f"  {cfg_name(kwargs):<30} MEAN: {mean*100:.2f}%")
            if mean>best[0]: best=(mean,kwargs)
        summary[method]={'best_mean_acc':best[0],'best_kwargs':best[1]}
        print(f"  >>> BEST {method}: {best[1]} -> {best[0]*100:.2f}%")

    print(f"\n{'='*70}\nRECAP — meilleur HP par methode (val, moyenne 3 seeds)\n{'='*70}")
    for m,r in summary.items():
        print(f"  {m:<8} {r['best_kwargs']} -> {r['best_mean_acc']*100:.2f}%")
    json.dump(summary, open(Path(args.results_dir)/'best_hp_summary.json','w'), indent=2)

if __name__=='__main__':
    main()

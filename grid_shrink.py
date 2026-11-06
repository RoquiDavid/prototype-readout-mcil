# Shrinkage grid.
import sys, json, os, argparse
sys.path.insert(0,'/data/roqui')
import numpy as np, torch
from pathlib import Path
from continual_learning_v3_clffix import ComplementarityMoE_V3_ClfFix
from models_meld import PerceiverIO_MoE
from meld_dataset import load_meld_class_incremental
from fecam_classifier import FeCAMMemory, evaluate_task_fecam

SHRINKS=[0.01,0.05,0.1,0.2,0.4,0.6,0.8,0.9,0.95]

def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    os.environ['PYTHONHASHSEED']=str(s)

def cl(M,T):
    avg=M[-1,:].mean()
    fgt=np.mean([M[-1,j]-max(M[t,j] for t in range(j,T-1)) for j in range(T-1)]) if T>1 else 0.
    bwt=np.mean([M[-1,j]-M[j,j] for j in range(T-1)]) if T>1 else 0.
    return float(avg),float(fgt),float(bwt)

p=argparse.ArgumentParser()
p.add_argument('--seed',type=int,default=11)
p.add_argument('--d_model',type=int,default=64)
a=p.parse_args()
dev='cuda'

set_seed(a.seed)
tr,va,te=load_meld_class_incremental(data_dir='/data/roqui/meld_data',batch_size=32,seed=a.seed)
T=len(tr)
set_seed(a.seed)
model=PerceiverIO_MoE(num_classes=7,d_model=a.d_model,num_experts=5,r=8,alpha=16,
    tau=0.2,num_latents=4,lora_up_init_std=0.01,text_dim=600,audio_dim=300,video_dim=35,
    use_text=True,use_audio=True,use_video=False).to(dev)
learner=ComplementarityMoE_V3_ClfFix(model,dev,lambda_barlow=0.0,tau=0.2,
    epochs_task0=30,primary_loss_weight=0.1,entropy_weight=0.1,classifier_lr_mult=0.05)

mems={s:FeCAMMemory(shrink=s) for s in SHRINKS}
Mv={s:np.zeros((T,T)) for s in SHRINKS}
Mt={s:np.zeros((T,T)) for s in SHRINKS}

for tid in range(T):
    print(f"\n--- Task {tid}/{T-1} ---",flush=True)
    if tid>0: model.router.reset_routing_head()
    learner.train_task(tr[tid],va[tid],epochs=15,lr=1e-3)
    for s in SHRINKS:
        mems[s].update(model,tr[tid],dev)
        for et in range(tid+1):
            Mv[s][tid,et]=evaluate_task_fecam(model,va[et],mems[s],dev)['accuracy']
            Mt[s][tid,et]=evaluate_task_fecam(model,te[et],mems[s],dev)['accuracy']

print(f"\n{'='*58}\nseed={a.seed} d_model={a.d_model}\nshrink   val      test     fgt")
res={}
for s in SHRINKS:
    v,f,_=cl(Mv[s],T); t,_,_=cl(Mt[s],T)
    res[str(s)]={'val':v,'test':t,'fgt':f}
    print(f"{s:<9}{v*100:6.2f}%  {t*100:6.2f}%  {f*100:7.2f}%")
print(f"{'='*58}\nref shrink=0.2 d64 : val 42.48")
o=Path(f'/data/roqui/meld_grid/shrink/seed{a.seed}_d{a.d_model}'); o.mkdir(parents=True,exist_ok=True)
json.dump(res,open(o/'result.json','w'),indent=2)

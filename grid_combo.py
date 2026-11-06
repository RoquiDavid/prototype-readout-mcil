# Rank, init and experts grid.
import sys, json, os, itertools, argparse
sys.path.insert(0,'/data/roqui')
import numpy as np, torch
from pathlib import Path
from continual_learning_v3_clffix import ComplementarityMoE_V3_ClfFix
from models_meld import PerceiverIO_MoE
from meld_dataset import load_meld_class_incremental
from fecam_classifier import FeCAMMemory, evaluate_task_fecam
from ncm_classifier import PrototypeMemory, evaluate_task_ncm

COMBOS=[]
for r in [12,16,24]:
    for std in [0.0,0.01]:
        for ne in [3,5]:
            for ew in [0.1,0.3]:
                COMBOS.append(dict(r=r,lora_up_init_std=std,num_experts=ne,
                    entropy_weight=ew,primary_loss_weight=0.5,lr=1e-3,
                    classifier_lr_mult=0.05,epochs_per_task=15))

def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    os.environ['PYTHONHASHSEED']=str(s)

def run(cfg,seed):
    set_seed(seed)
    tr,va,te=load_meld_class_incremental(data_dir='/data/roqui/meld_data',batch_size=32,seed=seed)
    T=len(tr); set_seed(seed)
    m=PerceiverIO_MoE(num_classes=7,d_model=64,num_experts=cfg['num_experts'],
        r=cfg['r'],alpha=2*cfg['r'],tau=0.2,num_latents=4,
        lora_up_init_std=cfg['lora_up_init_std'],text_dim=600,audio_dim=300,video_dim=35,
        use_text=True,use_audio=True,use_video=False).to('cuda')
    L=ComplementarityMoE_V3_ClfFix(m,'cuda',lambda_barlow=0.0,tau=0.2,epochs_task0=30,
        primary_loss_weight=cfg['primary_loss_weight'],entropy_weight=cfg['entropy_weight'],
        classifier_lr_mult=cfg['classifier_lr_mult'])
    fe=FeCAMMemory(shrink=0.05); nc=PrototypeMemory(num_classes=7,space='frozen')
    Mf=np.zeros((T,T)); Mn=np.zeros((T,T))
    for tid in range(T):
        if tid>0: m.router.reset_routing_head()
        L.train_task(tr[tid],va[tid],epochs=cfg['epochs_per_task'],lr=cfg['lr'])
        fe.update(m,tr[tid],'cuda'); nc.update(m,tr[tid],'cuda')
        for et in range(tid+1):
            Mf[tid,et]=evaluate_task_fecam(m,va[et],fe,'cuda')['accuracy']
            Mn[tid,et]=evaluate_task_ncm(m,va[et],nc,'cuda')['accuracy']
    return float(Mf[-1,:].mean()),float(Mn[-1,:].mean())

p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,default=11); a=p.parse_args()
out=Path('/data/roqui/meld_grid/combo'); out.mkdir(parents=True,exist_ok=True)
res=[]
print(f"{'r':>3} {'std':>5} {'ne':>3} {'ew':>4}  {'FeCAM':>8}{'NCM':>8}",flush=True)
for c in COMBOS:
    f,n=run(c,a.seed)
    print(f"{c['r']:>3} {c['lora_up_init_std']:>5} {c['num_experts']:>3} "
          f"{c['entropy_weight']:>4}  {f*100:7.2f}%{n*100:7.2f}%",flush=True)
    res.append({**c,'fecam':f,'ncm':n})
    json.dump(res,open(out/f'seed{a.seed}.json','w'),indent=2)
b=max(res,key=lambda x:x['fecam'])
print(f"\nMEILLEUR : r={b['r']} std={b['lora_up_init_std']} ne={b['num_experts']} "
      f"ew={b['entropy_weight']} -> FeCAM {b['fecam']*100:.2f}%")
print("ref : 42.48 actuelle | 45.70 (r=16 seul) | 45.98 NCM brut")

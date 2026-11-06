# Latents and temperature grid.
import sys, json, os
sys.path.insert(0,'/data/roqui')
import numpy as np, torch
from pathlib import Path
from continual_learning_v3_clffix import ComplementarityMoE_V3_ClfFix
from models_meld import PerceiverIO_MoE
from meld_dataset import load_meld_class_incremental
from fecam_classifier import FeCAMMemory, evaluate_task_fecam

COMBOS=[(nl,tau) for nl in [4,8,16,32,64] for tau in [0.1,0.2,0.5]]
SEEDS=[11,50,99]
CFG=dict(r=16,lora_up_init_std=0.0,num_experts=3,entropy_weight=0.1,
         primary_loss_weight=0.5,lr=1e-3,classifier_lr_mult=0.05,epochs_per_task=15)

def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    os.environ['PYTHONHASHSEED']=str(s)

def run(nl,tau,seed):
    set_seed(seed)
    tr,va,te=load_meld_class_incremental(data_dir='/data/roqui/meld_data',batch_size=32,seed=seed)
    T=len(tr); set_seed(seed)
    m=PerceiverIO_MoE(num_classes=7,d_model=64,num_experts=CFG['num_experts'],
        r=CFG['r'],alpha=2*CFG['r'],tau=tau,num_latents=nl,
        lora_up_init_std=CFG['lora_up_init_std'],text_dim=600,audio_dim=300,video_dim=35,
        use_text=True,use_audio=True,use_video=False).to('cuda')
    L=ComplementarityMoE_V3_ClfFix(m,'cuda',lambda_barlow=0.0,tau=tau,epochs_task0=30,
        primary_loss_weight=CFG['primary_loss_weight'],entropy_weight=CFG['entropy_weight'],
        classifier_lr_mult=CFG['classifier_lr_mult'])
    fe=FeCAMMemory(shrink=0.05)
    M=np.zeros((T,T))
    for tid in range(T):
        if tid>0: m.router.reset_routing_head()
        L.train_task(tr[tid],va[tid],epochs=CFG['epochs_per_task'],lr=CFG['lr'])
        fe.update(m,tr[tid],'cuda')
        for et in range(tid+1):
            M[tid,et]=evaluate_task_fecam(m,va[et],fe,'cuda')['accuracy']
    return float(M[-1,:].mean())

out=Path('/data/roqui/meld_grid/latents'); out.mkdir(parents=True,exist_ok=True)
R={}
print(f"{'nl':>4}{'tau':>6}{'moyenne':>10}{'ecart':>8}   seeds",flush=True)
for nl,tau in COMBOS:
    a=[run(nl,tau,s) for s in SEEDS]
    R[f"{nl}_{tau}"]=a
    print(f"{nl:>4}{tau:>6}{np.mean(a)*100:9.2f}%{np.std(a)*100:7.2f}%   "
          + " ".join(f"{x*100:.2f}" for x in a),flush=True)
    json.dump(R,open(out/'results.json','w'),indent=2)
b=max(R,key=lambda k:np.mean(R[k]))
print(f"\nMEILLEUR : num_latents/tau = {b} -> {np.mean(R[b])*100:.2f}%")
print("ref : 42.54")

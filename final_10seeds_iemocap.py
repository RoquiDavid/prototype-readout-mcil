# Method on IEMOCAP, ten seeds.
import sys, json, os, argparse
sys.path.insert(0,'/data/roqui')
import numpy as np, torch
from pathlib import Path
from continual_learning_v3_clffix import ComplementarityMoE_V3_ClfFix
from models_meld import PerceiverIO_MoE
from iemocap_dataset import load_iemocap_class_incremental as load_meld_class_incremental
from fecam_classifier import FeCAMMemory, evaluate_task_fecam
from ncm_classifier import PrototypeMemory, evaluate_task_ncm

CFG=dict(r=16,lora_up_init_std=0.0,num_experts=3,entropy_weight=0.1,
         primary_loss_weight=0.5,lr=1e-3,classifier_lr_mult=0.05,epochs_per_task=15)
SEEDS=[10,20,30,40,50,60,70,80,90,100]

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

def run(seed):
    set_seed(seed)
    tr,va,te=load_meld_class_incremental(data_dir='/data/roqui/iemocap_data',batch_size=32,seed=seed)
    T=len(tr); set_seed(seed)
    m=PerceiverIO_MoE(num_classes=7,d_model=64,num_experts=CFG['num_experts'],
        r=CFG['r'],alpha=2*CFG['r'],tau=0.2,num_latents=4,
        lora_up_init_std=CFG['lora_up_init_std'],text_dim=100,audio_dim=100,video_dim=35,
        use_text=True,use_audio=True,use_video=False).to('cuda')
    L=ComplementarityMoE_V3_ClfFix(m,'cuda',lambda_barlow=0.0,tau=0.2,epochs_task0=30,
        primary_loss_weight=CFG['primary_loss_weight'],entropy_weight=CFG['entropy_weight'],
        classifier_lr_mult=CFG['classifier_lr_mult'])
    fe=FeCAMMemory(shrink=0.05); nc=PrototypeMemory(num_classes=7,space='frozen')
    Mvf=np.zeros((T,T)); Mtf=np.zeros((T,T)); Mvn=np.zeros((T,T)); Mtn=np.zeros((T,T))
    for tid in range(T):
        if tid>0: m.router.reset_routing_head()
        L.train_task(tr[tid],va[tid],epochs=CFG['epochs_per_task'],lr=CFG['lr'])
        fe.update(m,tr[tid],'cuda'); nc.update(m,tr[tid],'cuda')
        for et in range(tid+1):
            Mvf[tid,et]=evaluate_task_fecam(m,va[et],fe,'cuda')['accuracy']
            Mtf[tid,et]=evaluate_task_fecam(m,te[et],fe,'cuda')['accuracy']
            Mvn[tid,et]=evaluate_task_ncm(m,va[et],nc,'cuda')['accuracy']
            Mtn[tid,et]=evaluate_task_ncm(m,te[et],nc,'cuda')['accuracy']
    f1v=np.mean([evaluate_task_fecam(m,va[t],fe,'cuda')['f1_weighted'] for t in range(T)])
    f1t=np.mean([evaluate_task_fecam(m,te[t],fe,'cuda')['f1_weighted'] for t in range(T)])
    vf,fgt,bwt=cl(Mvf,T); tf,tfgt,tbwt=cl(Mtf,T)
    vn,_,_=cl(Mvn,T); tn,_,_=cl(Mtn,T)
    return {'seed':seed,'fecam_val':vf,'fecam_test':tf,'ncm_val':vn,'ncm_test':tn,
            'f1_val':float(f1v),'f1_test':float(f1t),'fgt_test':tfgt,'bwt_test':tbwt,
            'matrix_test':Mtf.tolist()}

p=argparse.ArgumentParser(); a=p.parse_args()
out=Path('/data/roqui/iemocap_grid/final10'); out.mkdir(parents=True,exist_ok=True)
R=[]
print(f"CONFIG FIGEE : {CFG}\n",flush=True)
print(f"{'seed':>5}{'FeCAM val':>11}{'FeCAM test':>12}{'NCM test':>10}{'f1 test':>9}",flush=True)
for s in SEEDS:
    r=run(s); R.append(r)
    print(f"{s:>5}{r['fecam_val']*100:10.2f}%{r['fecam_test']*100:11.2f}%"
          f"{r['ncm_test']*100:9.2f}%{r['f1_test']*100:8.2f}%",flush=True)
    json.dump(R,open(out/'results.json','w'),indent=2)
g=lambda k:np.array([r[k] for r in R])*100
print(f"\n{'='*60}")
for k,lbl in [('fecam_val','FeCAM val'),('fecam_test','FeCAM test'),
              ('ncm_test','NCM test'),('f1_test','f1w test'),
              ('fgt_test','forgetting test'),('bwt_test','BWT test')]:
    print(f"{lbl:<18}{g(k).mean():7.2f} +/- {g(k).std():.2f}")
print(f"{'='*60}\nref : 42.48 avant tuning | DER++ tune 41.41")

# Replay baselines with modality dropout.
import sys, json, os, argparse
sys.path.insert(0,'/data/roqui')
import numpy as np, torch, types
from pathlib import Path
from sklearn.metrics import f1_score
from models_meld import PerceiverIO_MoE
from meld_baselines import Naive, EWC, LwF, ER, DERpp, AGEM, DMoLE, CLMoE

p=argparse.ArgumentParser()
p.add_argument('--pdrop',type=float,default=0.0)
p.add_argument('--dataset',required=True,choices=['meld','iemocap'])
p.add_argument('--method',required=True)
a=p.parse_args()
if a.dataset=='meld':
    from meld_dataset import load_meld_class_incremental as load
    DD,TD,AD='/data/roqui/meld_data',600,300
else:
    from iemocap_dataset import load_iemocap_class_incremental as load
    DD,TD,AD='/data/roqui/iemocap_data',100,100

CLS={'Naive':Naive,'EWC':EWC,'LwF':LwF,'ER':ER,'DER++':DERpp,'AGEM':AGEM,'D-MoLE':DMoLE,'CL-MoE':CLMoE}
FALLBACK={'Naive':{},'EWC':{'lambda_ewc':1000.0},'LwF':{'lambda_kd':1.0,'temperature':2.0},
 'ER':{'buffer_size':500},'DER++':{'buffer_size':500,'alpha':0.5,'beta':0.8},
 'AGEM':{'buffer_size':500},'D-MoLE':{},'CL-MoE':{'momentum':0.9}}
kw,src=FALLBACK[a.method],'fallback'
g=Path('/data/roqui/meld_grid_baselines/best_hp_summary.json')
if g.exists():
    s=json.load(open(g))
    if a.method in s: kw,src=s[a.method]['best_kwargs'],'grid'
print(f"{a.dataset} {a.method} kwargs={kw} (source: {src})",flush=True)

MC=dict(num_classes=7,d_model=64,num_experts=5,r=8,alpha=16,tau=0.2,num_latents=4,
        text_dim=TD,audio_dim=AD,video_dim=35,use_text=True,use_audio=True,use_video=False)
SEEDS=[10,20,30,40,50,60,70,80,90,100]
REG=[('full',True,True),('text_only',True,False),('audio_only',False,True)]

def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    os.environ['PYTHONHASHSEED']=str(s)

def ev(m,L,d,kt=True,ka=True):
    m.eval(); c=n=0; P=[]; Y=[]
    with torch.no_grad():
        for b in L:
            t=b['text'].to(d); au=b['audio'].to(d)
            if not kt: t=torch.zeros_like(t)
            if not ka: au=torch.zeros_like(au)
            y=torch.clamp(b['labels'].to(d)[:,0],-3,3).round().long()+3
            pr=m(t,au,b['video'].to(d))['logits'].argmax(1)
            c+=(pr==y).sum().item(); n+=y.size(0); P+=pr.tolist(); Y+=y.tolist()
    return (c/n if n else 0.), f1_score(Y,P,average='weighted',zero_division=0)

out=Path(f'/data/roqui/sealed_baselines_drop{a.pdrop}/{a.dataset}'); out.mkdir(parents=True,exist_ok=True)
R=[]
for seed in SEEDS:
    set_seed(seed)
    tr,va,te=load(data_dir=DD,batch_size=32,seed=seed)
    T=len(tr); set_seed(seed)
    m=PerceiverIO_MoE(**MC).to('cuda')
    def fwd(self,text,audio,video):
        B=text.size(0)
        inp=torch.cat([self.text_proj(text.mean(1,keepdim=True)),
                       self.audio_proj(audio.mean(1,keepdim=True))],1)
        msk=torch.zeros(B,2,dtype=torch.bool,device=text.device)
        if self.training and self.p_drop>0:
            r=torch.rand(B,device=text.device)
            msk[:,0]=r<self.p_drop/2; msk[:,1]=(r>=self.p_drop/2)&(r<self.p_drop)
        lat=self.latent.expand(B,-1,-1)
        at,_=self.cross_attention(lat,inp,inp,key_padding_mask=msk)
        lat=self.norm1(lat+at); lat=self.norm2(lat+self.ffn(lat))
        return lat.mean(1)
    enc=m.perceiver_encoder; enc.p_drop=a.pdrop
    enc.forward=types.MethodType(fwd,enc)
    L=CLS[a.method](m,'cuda',**kw)
    Mv=np.zeros((T,T)); Mt=np.zeros((T,T))
    for tid in range(T):
        L.train_task(tr[tid],va[tid],epochs=30 if tid==0 else 15,lr=1e-3)
        for et in range(tid+1):
            Mv[tid,et]=ev(m,va[et],'cuda')[0]; Mt[tid,et]=ev(m,te[et],'cuda')[0]
    r={'seed':seed,'val':float(Mv[-1,:].mean()),'test':float(Mt[-1,:].mean()),
       'f1_test':float(np.mean([ev(m,te[t],'cuda')[1] for t in range(T)])),
       'fgt_test':float(np.mean([Mt[-1,j]-max(Mt[x,j] for x in range(j,T-1)) for j in range(T-1)])),
       'matrix_test':Mt.tolist(),'kwargs':kw,'kwargs_source':src}
    for name,kt,ka in REG:
        r[f'test_{name}']=float(np.mean([ev(m,te[t],'cuda',kt,ka)[0] for t in range(T)]))
    R.append(r)
    print(f"seed {seed}: test {r['test']*100:.2f}  text_only {r['test_text_only']*100:.2f}  "
          f"audio_only {r['test_audio_only']*100:.2f}",flush=True)
    json.dump(R,open(out/f'{a.method}.json','w'),indent=2)
print("FINI",flush=True)

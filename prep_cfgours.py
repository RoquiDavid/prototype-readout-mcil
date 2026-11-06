# Creates baselines_sealed_cfgours.py.
import sys
from pathlib import Path
R = Path('/data/roqui')
src = R / 'baselines_sealed.py'; dst = R / 'baselines_sealed_cfgours.py'
s = src.read_text(encoding='utf-8')
def rep(s, a, b, nom):
    k = s.count(a)
    if k != 1: sys.exit(f"ARRET : '{nom}' trouve {k} fois au lieu de 1.")
    return s.replace(a, b)
s = rep(s, "MC=dict(num_classes=7,d_model=64,num_experts=5,r=8,alpha=16,tau=0.2,num_latents=4,",
           "MC=dict(num_classes=7,d_model=64,num_experts=3,r=16,alpha=32,tau=0.2,num_latents=4,",
        "configuration des experts")
s = rep(s, "out=Path(f'/data/roqui/sealed_baselines/{a.dataset}')",
           "out=Path(f'/data/roqui/sealed_baselines_cfgours/{a.dataset}')",
        "dossier de sortie")
if dst.exists(): sys.exit(f"ARRET : {dst} existe deja.")
dst.write_text(s, encoding='utf-8')
print("OK : baselines_sealed_cfgours.py cree")

# Raw features and random projection.
import sys, json
sys.path.insert(0, '/data/roqui')
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from scipy.stats import wilcoxon
from runs_nuit import load, set_seed, acc_tasks, DATASETS, SEEDS, DEVICE, DATA
from fecam_classifier import FeCAMMemory

OUT = Path('/data/roqui/nuit')

class Feat(nn.Module):
    def __init__(self, d_in, d_out=None):
        super().__init__()
        self.W = None if d_out is None else nn.Parameter(
            torch.randn(d_in, d_out) / d_in ** 0.5, requires_grad=False)
    def forward(self, text, audio):
        x = torch.cat([text.mean(1), audio.mean(1)], 1)
        return x if self.W is None else x @ self.W

class FeatFeCAM(FeCAMMemory):
    @torch.no_grad()
    def _features(self, model, batch, device):
        return F.normalize(model(batch['text'].to(device), batch['audio'].to(device)), dim=1)

def run(ds, seed, d_out):
    set_seed(seed); tr, va, te = load(ds, seed); set_seed(seed)
    m = Feat(DATA[ds]['text_dim'] + DATA[ds]['audio_dim'], d_out).to(DEVICE).eval()
    fe = FeatFeCAM(shrink=0.05)
    for tid in range(len(tr)):
        fe.update(m, tr[tid], DEVICE)
    return acc_tasks(m, lambda b: fe.predict(m, b, DEVICE), te)

for ds in DATASETS:
    R = []
    for seed in SEEDS:
        R.append({'seed': seed, 'raw_concat': run(ds, seed, None), 'rand_proj_64': run(ds, seed, 64)})
        json.dump(R, open(OUT / f'P_{ds}.json', 'w'), indent=2)
        print(f"[P] {ds} seed {seed} fait", flush=True)
    U = [r['untrained_encoder'] for r in json.load(open(OUT / f'U_{ds}.json'))]
    for k in ('raw_concat', 'rand_proj_64'):
        a = [r[k] for r in R]
        print(f"===== P | {ds} | {k} : {100*np.mean(a):.2f} +/- {100*np.std(a):.2f}"
              f" | encodeur jamais entraine - {k} : {100*(np.mean(U)-np.mean(a)):+.2f} pts"
              f" | p = {wilcoxon(U, a).pvalue:.4f}", flush=True)
print("RUN P TERMINE", flush=True)

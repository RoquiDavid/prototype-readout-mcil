# Readout on an untrained encoder.
import sys, json
sys.path.insert(0, '/data/roqui')
import numpy as np
from pathlib import Path
from runs_nuit import load, build, set_seed, acc_tasks, DATASETS, SEEDS, DEVICE
from fecam_classifier import FeCAMMemory

OUT = Path('/data/roqui/nuit'); OUT.mkdir(parents=True, exist_ok=True)

for ds in DATASETS:
    R = []
    for seed in SEEDS:
        set_seed(seed); tr, va, te = load(ds, seed); T = len(tr); set_seed(seed)
        m = build(ds); m.eval()
        fe = FeCAMMemory(shrink=0.05)
        for tid in range(T):
            fe.update(m, tr[tid], DEVICE)
        acc = acc_tasks(m, lambda b: fe.predict(m, b, DEVICE), te)
        R.append({'seed': seed, 'untrained_encoder': acc})
        json.dump(R, open(OUT / f'U_{ds}.json', 'w'), indent=2)
        print(f"[U] {ds} seed {seed} : {100*acc:.2f}", flush=True)
    a = np.array([r['untrained_encoder'] for r in R]) * 100
    print(f"===== U | {ds} | {len(R)} graines : {a.mean():.2f} +/- {a.std():.2f}", flush=True)
print("RUN U TERMINE", flush=True)

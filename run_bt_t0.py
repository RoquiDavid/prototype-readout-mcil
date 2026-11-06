# Barlow Twins on the first task.
import sys, os, json, types
sys.path.insert(0, '/data/roqui')
import numpy as np, torch
from pathlib import Path
from scipy.stats import wilcoxon
import continual_learning_v3_clffix as clf
from runs_nuit import (load, build, learner, train_task, acc_tasks, fwd_mask, set_seed,
                       DATASETS, SEEDS, DEVICE, SMOKE)
from fecam_classifier import FeCAMMemory

OUT = Path('/data/roqui/nuit_bt_smoke' if SMOKE else '/data/roqui/nuit_bt')
OUT.mkdir(parents=True, exist_ok=True)
GRID = [(0.1, 0.3), (0.1, 0.5), (1.0, 0.3), (1.0, 0.5)]
VAL_SEEDS = SEEDS[:3]


def bt_views(views, tau, lam):
    zt, za = views[:, 0], views[:, 1]
    B = zt.size(0)
    if B < 2:
        return torch.zeros((), device=zt.device)
    zt = (zt - zt.mean(0)) / (zt.std(0) + 1e-8)
    za = (za - za.mean(0)) / (za.std(0) + 1e-8)
    c = zt.T @ za / B
    d = torch.diagonal(c)
    return ((d - tau) ** 2).sum() + 0.01 * ((c ** 2).sum() - (d ** 2).sum())


clf._pairwise_barlow_twins = bt_views


def get_views(self, perceiver_features):
    text, audio, video = self.model._last
    enc = self.model.perceiver_encoder
    enc.force = 'no_audio'; zt = enc(text, audio, video)
    enc.force = 'no_text';  za = enc(text, audio, video)
    enc.force = None
    return torch.stack([zt, za], 1)


def run(ds, seed, w, tau, split):
    set_seed(seed); tr, va, te = load(ds, seed); T = len(tr); set_seed(seed)
    m = build(ds); enc = m.perceiver_encoder
    enc.p_drop = 0.0; enc.force = None; enc.forward = types.MethodType(fwd_mask, enc)
    fwd0 = m.forward

    def fwd_stash(text, audio, video, *a, **k):
        m._last = (text, audio, video)
        return fwd0(text, audio, video, *a, **k)
    m.forward = fwd_stash

    L = learner(m)
    L.lambda_barlow = w; L.tau = tau
    L._get_expert_outputs = types.MethodType(get_views, L)
    fe = FeCAMMemory(shrink=0.05)
    for tid in range(T):
        train_task(m, L, tr, va, tid)
        if tid == 0:
            L.lambda_barlow = 0.0
        fe.update(m, tr[tid], DEVICE)
    return acc_tasks(m, lambda b: fe.predict(m, b, DEVICE), va if split == 'val' else te)


if __name__ == '__main__':
    V = {}
    for w, tau in [(0.0, 0.2)] + GRID:
        key = f'w{w}_tau{tau}'
        for ds in DATASETS:
            V.setdefault(key, {})[ds] = [run(ds, s, w, tau, 'val') for s in VAL_SEEDS]
            print(f"[VAL] {key} {ds} : {100*np.mean(V[key][ds]):.2f}", flush=True)
            json.dump(V, open(OUT / 'val.json', 'w'), indent=2)
    score = {k: np.mean([np.mean(v[ds]) for ds in DATASETS]) for k, v in V.items() if k != 'w0.0_tau0.2'}
    ref = np.mean([np.mean(V['w0.0_tau0.2'][ds]) for ds in DATASETS])
    best = max(score, key=score.get)
    w, tau = [(a, b) for a, b in GRID if f'w{a}_tau{b}' == best][0]
    print(f"===== VAL : reglage retenu {best} ({100*score[best]:.2f}) | modele actuel {100*ref:.2f}", flush=True)
    for ds in DATASETS:
        R = []
        for s in SEEDS:
            R.append(run(ds, s, w, tau, 'test'))
            json.dump({'config': best, 'seeds': SEEDS[:len(R)], 'test': R},
                      open(OUT / f'test_{ds}.json', 'w'), indent=2)
            print(f"[TEST] {ds} graine {s} : {100*R[-1]:.2f}", flush=True)
        line = f"===== TEST {ds} {best} : {100*np.mean(R):.2f} +/- {100*np.std(R):.2f}"
        try:
            cur = [r['0.0']['matched_full'] for r in json.load(open(f'/data/roqui/nuit/A_{ds}.json'))]
            unt = [r['untrained_encoder'] for r in json.load(open(f'/data/roqui/nuit/U_{ds}.json'))]
            if len(cur) == len(R):
                line += f" | vs actuel {100*(np.mean(R)-np.mean(cur)):+.2f} p={wilcoxon(R, cur).pvalue:.4f}"
            if len(unt) == len(R):
                line += f" | vs non entraine {100*(np.mean(R)-np.mean(unt)):+.2f} p={wilcoxon(R, unt).pvalue:.4f}"
        except Exception as e:
            line += f" | comparaison impossible ({e})"
        print(line, flush=True)
    print("RUN BT T0 TERMINE", flush=True)

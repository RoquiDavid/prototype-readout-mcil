# NCM, LDA and FeCAM readouts.
import sys, json
sys.path.insert(0, '/data/roqui')
import numpy as np, torch
from pathlib import Path
from scipy.stats import wilcoxon
from runs_nuit import load, build, learner, train_task, set_seed, DATASETS, SEEDS, DEVICE, SMOKE
from fecam_classifier import FeCAMMemory

OUT = Path('/data/roqui/nuit_smoke' if SMOKE else '/data/roqui/nuit'); OUT.mkdir(parents=True, exist_ok=True)
SHRINKS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
FE = FeCAMMemory()


@torch.no_grad()
def feats(m, loader):
    m.eval(); Z, Y = [], []
    for b in loader:
        Z.append(FE._features(m, b, DEVICE)); Y.append(FeCAMMemory._labels(b, DEVICE))
    return torch.cat(Z), torch.cat(Y)


def stats(Z, Y):
    st = {}
    for c in Y.unique().tolist():
        X = Z[Y == c]; mu = X.mean(0); Xc = X - mu
        st[c] = (mu, (Xc.T @ Xc) / max(X.size(0) - 1, 1), X.size(0))
    return st


def prec(cov, s):
    d = cov.size(0); I = torch.eye(d, device=cov.device)
    return torch.linalg.inv((1 - s) * cov + s * (cov.trace() / d) * I + 1e-5 * I)


def predict(Z, st, mode, s=0.05):
    cls = sorted(st); M = torch.stack([st[c][0] for c in cls])
    if mode == 'ncm':
        D = torch.cdist(Z, M) ** 2
    elif mode == 'lda':
        n = sum(st[c][2] for c in cls)
        S = sum((st[c][2] - 1) * st[c][1] for c in cls) / max(n - len(cls), 1)
        P = prec(S, s); diff = Z[:, None, :] - M[None]
        D = torch.einsum('bcd,de,bce->bc', diff, P, diff)
    else:
        D = torch.stack([(((Z - st[c][0]) @ prec(st[c][1], s)) * (Z - st[c][0])).sum(1) for c in cls], 1)
    return torch.tensor(cls, device=Z.device)[D.argmin(1)]


def run(ds, seed):
    set_seed(seed); tr, va, te = load(ds, seed); T = len(tr); set_seed(seed)
    m = build(ds); L = learner(m)
    train_task(m, L, tr, va, 0)
    st = {}
    for tid in range(T):
        st.update(stats(*feats(m, tr[tid])))
    test = [feats(m, te[tid]) for tid in range(T)]
    modes = [('ncm', None), ('lda', None)] + [('fecam', s) for s in SHRINKS]
    res = {}
    for mode, s in modes:
        key = mode if s is None else f'fecam_{s}'
        res[key] = float(np.mean([(predict(Z, st, mode, s if s is not None else 0.05) == Y).float().mean().item()
                                  for Z, Y in test]))
    return res


if __name__ == '__main__':
    for ds in DATASETS:
        R = []
        try:
            ref = {r['seed']: r['0.0']['matched_full'] for r in json.load(open(f'/data/roqui/nuit/A_{ds}.json'))}
        except Exception:
            ref = {}
        for seed in SEEDS:
            r = run(ds, seed); r['seed'] = seed; R.append(r)
            json.dump(R, open(OUT / f'R_{ds}.json', 'w'), indent=2)
            chk = f" | controle FeCAM 0.05 vs article : {100*(r['fecam_0.05']-ref[seed]):+.2f} pts" if seed in ref else ''
            print(f"[R] {ds} graine {seed} : " + ', '.join(f"{k} {100*v:.2f}" for k, v in r.items() if k != 'seed') + chk, flush=True)
        base = [r['fecam_0.05'] for r in R]
        for k in [k for k in R[0] if k != 'seed']:
            a = [r[k] for r in R]
            p = ''
            if k != 'fecam_0.05' and len(R) >= 2:
                try:
                    p = f" | vs FeCAM 0.05 : {100*(np.mean(a)-np.mean(base)):+.2f} p={wilcoxon(a, base).pvalue:.4f}"
                except ValueError:
                    p = " | identique a FeCAM 0.05"
            print(f"===== R | {ds} | {k:12s} : {100*np.mean(a):.2f} +/- {100*np.std(a):.2f}{p}", flush=True)
    print("RUN R TERMINE", flush=True)

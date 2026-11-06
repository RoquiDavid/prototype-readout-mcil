# First task with several classes.
import sys, json
sys.path.insert(0, '/data/roqui')
import numpy as np
from scipy.stats import wilcoxon
import iemocap_dataset
from meld_dataset import load_meld_class_incremental, DEFAULT_CI_TASK_EMOTIONS
from runs_nuit import build, learner, train_task, set_seed, acc_tasks, DATASETS, SEEDS, DEVICE, DATA, OUT
from fecam_classifier import FeCAMMemory

KS = [1, 2, 3]
BASE = {'meld': [list(t) for t in DEFAULT_CI_TASK_EMOTIONS],
        'iemocap': [list(t) for t in iemocap_dataset.TASK_SPLIT]}


def split(ds, k):
    s = BASE[ds]
    return [sum(s[:k], [])] + s[k:]


def load(ds, seed, k):
    if ds == 'meld':
        return load_meld_class_incremental(data_dir=DATA[ds]['data_dir'], batch_size=32, seed=seed,
                                           task_emotions=split(ds, k))
    iemocap_dataset.TASK_SPLIT = split(ds, k)
    return iemocap_dataset.load_iemocap_class_incremental(data_dir=DATA[ds]['data_dir'], batch_size=32, seed=seed)


def run(ds, seed, k, trained):
    set_seed(seed); tr, va, te = load(ds, seed, k); set_seed(seed)
    m = build(ds)
    if trained:
        train_task(m, learner(m), tr, va, 0)
    m.eval()
    fe = FeCAMMemory(shrink=0.05)
    for tid in range(len(tr)):
        fe.update(m, tr[tid], DEVICE)
    return acc_tasks(m, lambda b: fe.predict(m, b, DEVICE), te)


def ref(ds):
    try:
        A = {r['seed']: r['0.0']['matched_full'] for r in json.load(open(OUT / f'A_{ds}.json'))}
        U = {r['seed']: r['untrained_encoder'] for r in json.load(open(OUT / f'U_{ds}.json'))}
        return A, U
    except Exception:
        return {}, {}


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        f = OUT / f'T_{ds}.json'
        R = json.load(open(f)) if f.exists() else []
        done = {(r['k'], r['seed']) for r in R}
        A, U = ref(ds)
        for k in KS:
            for seed in SEEDS:
                if (k, seed) in done:
                    continue
                r = {'k': k, 'seed': seed, 't0': split(ds, k)[0],
                     'trained': run(ds, seed, k, True), 'untrained': run(ds, seed, k, False)}
                R.append(r)
                json.dump(R, open(f, 'w'), indent=2)
                chk = ''
                if k == 1 and seed in A and seed in U:
                    chk = f" | ecart avec A {100*(r['trained']-A[seed]):+.2f}, avec U {100*(r['untrained']-U[seed]):+.2f}"
                print(f"[T] {ds} K={k} graine {seed} : entraine {100*r['trained']:.2f}, "
                      f"non entraine {100*r['untrained']:.2f}{chk}", flush=True)
        for k in KS:
            a = np.array([r['trained'] for r in R if r['k'] == k]) * 100
            b = np.array([r['untrained'] for r in R if r['k'] == k]) * 100
            try:
                p = f"p={wilcoxon(a, b).pvalue:.4f}"
            except ValueError:
                p = "test impossible"
            print(f"===== T | {ds} | K={k}, T0 = {split(ds, k)[0]} | entraine {a.mean():.2f} +/- {a.std():.2f}"
                  f" | non entraine {b.mean():.2f} +/- {b.std():.2f} | ecart {(a - b).mean():+.2f} {p}", flush=True)
    print("RUN T TERMINE", flush=True)

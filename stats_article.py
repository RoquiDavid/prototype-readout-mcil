# Paired Wilcoxon tests, saved results.
import json, glob, os
import numpy as np
from scipy.stats import wilcoxon

ROOT = os.environ.get('ROOT', '/data/roqui')


def ours(ds):
    R = json.load(open(f'{ROOT}/nuit/A_{ds}.json'))
    return {r.get('seed', i): r['0.0']['matched_full'] for i, r in enumerate(R)}


def base(path):
    R = json.load(open(path))
    return {r.get('seed', i): r['test'] for i, r in enumerate(R)}


def compare(o, b, label):
    seeds = sorted(set(o) & set(b))
    x = np.array([o[s] for s in seeds]); y = np.array([b[s] for s in seeds])
    p = wilcoxon(x, y).pvalue
    print(f"{label:24s} n={len(seeds):2d}  {100*y.mean():6.2f} +/- {100*y.std():.2f}"
          f"  methode - baseline {100*(x.mean()-y.mean()):+6.2f}  p={p:.4f}")


for ds in ['meld', 'iemocap']:
    o = ours(ds)
    print(f"== {ds} : methode {100*np.mean(list(o.values())):.2f} +/- {100*np.std(list(o.values())):.2f}")
    print("-- Tableau principal (memoire 500, 200 pour A-GEM)")
    for f in sorted(glob.glob(f'{ROOT}/sealed_baselines/{ds}/*.json')):
        compare(o, base(f), os.path.basename(f)[:-5])
    print("-- Tableau des memoires (1000 et 2000 exemples)")
    for M in ['ER', 'DER++']:
        for B in [1000, 2000]:
            f = f'{ROOT}/sealed_baselines_buf{B}/{ds}/{M}.json'
            if os.path.exists(f):
                compare(o, base(f), f'{M} {B}')
    print("-- ER et DER++ avec 3 experts de rang 16")
    for M in ['ER', 'DER++']:
        f = f'{ROOT}/sealed_baselines_cfgours/{ds}/{M}.json'
        if os.path.exists(f):
            compare(o, base(f), f'{M} (3 experts, rang 16)')

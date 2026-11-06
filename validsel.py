# Selection on validation (ch5: CL-MoE).
import json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path

R = Path(os.environ.get('VALIDSEL_R', '/data/roqui'))
V = R / 'validsel'
PY = sys.executable
SEL_SEEDS = [11, 50, 99]
TEST_SEEDS = list(range(10, 101, 10))
CONFIGS = [('b30', 'ch4_t0all30'), ('b15', 'ch4_equity'), ('ch3', 'ch3_10seeds')]
GRIDS = {
    'ComplementarityMoE_Ours': [dict(tau=t, lambda_barlow=lb, router_lr_mult=rl, primary_loss_weight=pw)
                                for t in (0.2, 0.3, 0.5) for lb in (0.01, 0.1)
                                for rl in (1.0, 3.0) for pw in (0.5, 1.0)],
    'EWC': [dict(lambda_ewc=l) for l in (1.0, 10.0, 100.0, 500.0, 1000.0)],
    'DMoLE_Adapted': [dict(dmole_alpha=a) for a in (0.1, 0.5, 1.0, 2.0)],
    'CLMoE_Adapted': [dict(clmoe_task_w=w) for w in (0.1, 0.5, 1.0)],
}
CH5_W = (0.1, 0.5, 1.0)


def log(m):
    print(f"=== {datetime.now():%F %T} [{os.uname().nodename} {os.getpid()}] {m}", flush=True)


def tag(d):
    return '_'.join(f"{k}{v:g}" for k, v in d.items())


def load(n):
    return (R / n).read_text(encoding='utf-8', errors='surrogateescape')


def rep(s, a, b, nom):
    k = s.count(a)
    if k != 1:
        sys.exit(f"ARRET : '{nom}' trouve {k} fois au lieu de 1.")
    return s.replace(a, b)


def prep():
    new = {}
    s = load('continual_learning.py')
    s = rep(s, "    def __init__(self, model, device, lambda_barlow=0.01, tau=0.2, epochs_task0=None):\n",
            "    def __init__(self, model, device, lambda_barlow=0.01, tau=0.2, epochs_task0=None,\n"
            "                 router_lr_mult=3.0, primary_loss_weight=1.0):\n", "signature de la methode")
    s = rep(s, "        self.router_lr_mult = 3.0\n        self.primary_loss_weight = 1.0\n",
            "        self.router_lr_mult = router_lr_mult\n        self.primary_loss_weight = primary_loss_weight\n",
            "reglages de la methode")
    new['continual_learning_vs.py'] = s

    s = load('continual_learning_sota_dmolefix.py')
    s = rep(s, "                momentum: float = 0.9):\n        super().__init__(model, device)\n        self.momentum = momentum\n",
            "                momentum: float = 0.9, task_loss_weight: float = 0.5):\n        super().__init__(model, device)\n"
            "        self.momentum = momentum\n        self.task_loss_weight = task_loss_weight\n", "signature de CL-MoE")
    s = rep(s, "        return ce_loss + 0.5 * task_loss - 0.1 * entropy\n",
            "        return ce_loss + self.task_loss_weight * task_loss - 0.1 * entropy\n", "perte de CL-MoE")
    new['continual_learning_sota_vs.py'] = s

    s = load('ComprehensiveBenchmark_t0all.py')
    s = rep(s, "from continual_learning import ComplementarityMoE_Simple , EWC, Naive\n"
               "from continual_learning_sota import DMoLE_Adapted, CLMoE_Adapted, ProgLoRA_Adapted\n",
            "from continual_learning_vs import ComplementarityMoE_Simple , EWC, Naive\n"
            "from continual_learning_sota_vs import DMoLE_Adapted, CLMoE_Adapted, ProgLoRA_Adapted  # D-MoLE corrige (22/10)\n",
            "imports")
    s = rep(s, "learner_class(model, self.device, curriculum_alpha=0.5)",
            "learner_class(model, self.device, curriculum_alpha=self.args.dmole_alpha)", "reglage de D-MoLE")
    s = rep(s, "learner_class(model, self.device, momentum=0.9)",
            "learner_class(model, self.device, momentum=0.9, task_loss_weight=self.args.clmoe_task_w)", "reglage de CL-MoE")
    s = rep(s, "                                   epochs_task0=getattr(self.args, 'epochs_task0', None))\n",
            "                                   epochs_task0=getattr(self.args, 'epochs_task0', None),\n"
            "                                   router_lr_mult=self.args.router_lr_mult,\n"
            "                                   primary_loss_weight=self.args.primary_loss_weight)\n", "reglages de la methode")
    s = rep(s, "            task_results = self.evaluate_all_tasks(model, test_loaders, task_id)\n",
            "            task_results = self.evaluate_all_tasks(model, val_loaders if self.args.eval_split == 'val' else test_loaders, task_id)\n",
            "evaluation")
    s = rep(s, "        # Run all experiments\n",
            "        if self.args.only:\n"
            "            experiments = [e for e in experiments if e[0] == self.args.only]\n"
            "            if len(experiments) != 1:\n"
            "                sys.exit(f'ARRET : methode inconnue {self.args.only}')\n"
            "        \n        # Run all experiments\n", "liste des methodes")
    s = rep(s, "                       help='EWC regularization weight')\n",
            "                       help='EWC regularization weight')\n"
            "    # Choix sur la validation (24/10)\n"
            "    parser.add_argument('--eval_split', type=str, default='test', choices=['test', 'val'])\n"
            "    parser.add_argument('--only', type=str, default=None)\n"
            "    parser.add_argument('--router_lr_mult', type=float, default=3.0)\n"
            "    parser.add_argument('--primary_loss_weight', type=float, default=1.0)\n"
            "    parser.add_argument('--dmole_alpha', type=float, default=0.5)\n"
            "    parser.add_argument('--clmoe_task_w', type=float, default=0.5)\n", "arguments")
    new['ComprehensiveBenchmark_vs.py'] = s

    s = load('meld_baselines.py')
    s = rep(s, "    def __init__(self, model, device='cuda', momentum=0.9):\n        self.model = model\n"
               "        self.device = device\n        self.momentum = momentum\n",
            "    def __init__(self, model, device='cuda', momentum=0.9, task_loss_weight=0.5):\n        self.model = model\n"
            "        self.device = device\n        self.momentum = momentum\n        self.task_loss_weight = task_loss_weight\n",
            "signature de CL-MoE (Ch5)")
    s = rep(s, "                loss = ce + 0.5 * task_loss - 0.1 * entropy\n",
            "                loss = ce + self.task_loss_weight * task_loss - 0.1 * entropy\n", "perte de CL-MoE (Ch5)")
    new['meld_baselines_vs.py'] = s

    s = load('baselines_sealed.py')
    s = rep(s, "from meld_baselines import ", "from meld_baselines_vs import ", "import (Ch5)")
    s = rep(s, "p.add_argument('--method',required=True)\n",
            "p.add_argument('--method',required=True)\n"
            "p.add_argument('--kw',required=True)     # reglages (JSON), choix sur la validation 24/10\n"
            "p.add_argument('--seeds',required=True)  # ex. 11,50,99\n"
            "p.add_argument('--out',required=True)    # dossier NOUVEAU\n", "arguments (Ch5)")
    s = rep(s, "kw,src=FALLBACK[a.method],'fallback'\n", "kw,src=json.loads(a.kw),'validsel'\n", "reglages (Ch5)")
    s = rep(s, "    if a.method in s: kw,src=s[a.method]['best_kwargs'],'grid'\n",
            "    pass  # validsel : les reglages viennent de --kw\n", "grille d'origine (Ch5)")
    s = rep(s, "SEEDS=[10,20,30,40,50,60,70,80,90,100]\n", "SEEDS=[int(x) for x in a.seeds.split(',')]\n", "graines (Ch5)")
    s = rep(s, "out=Path(f'/data/roqui/sealed_baselines/{a.dataset}')", "out=Path(a.out)/a.dataset", "dossier (Ch5)")
    new['baselines_vs.py'] = s

    for n in new:
        if (R / n).exists():
            sys.exit(f"ARRET : {R / n} existe deja.")
    for n, s in new.items():
        (R / n).write_text(s, encoding='utf-8', errors='surrogateescape')
    print("OK : " + ", ".join(new) + " crees.")


def orig_args(src):
    A = []
    for sd in TEST_SEEDS:
        a = json.load(open(R / src / f'seed_{sd}' / 'summary.json'))['args']
        A.append({k: v for k, v in a.items() if k not in ('results_dir', 'seed')})
    if any(x != A[0] for x in A):
        sys.exit(f"ARRET : les arguments different d'une graine a l'autre dans {src}.")
    a = A[0]
    if src != 'ch4_t0all30' and a.get('epochs_task0') not in (None, a['epochs_per_task']):
        sys.exit(f"ARRET : {src} a epochs_task0={a.get('epochs_task0')} : budget non reproductible.")
    return a


def cli(a):
    c = []
    for k, v in a.items():
        if v is None or v is False:
            continue
        c += [f'--{k}'] if v is True else [f'--{k}', str(v)]
    return c


def current(cfg_args):
    return {'ComplementarityMoE_Ours': dict(tau=cfg_args['tau'], lambda_barlow=cfg_args['lambda_barlow'],
                                            router_lr_mult=3.0, primary_loss_weight=1.0),
            'EWC': dict(lambda_ewc=float(cfg_args['lambda_ewc'])),
            'DMoLE_Adapted': dict(dmole_alpha=0.5),
            'CLMoE_Adapted': dict(clmoe_task_w=0.5)}


class Job:
    def __init__(self, jid, cmd, out, check):
        self.jid, self.cmd, self.out, self.check = jid, cmd, out, check

    def done(self):
        try:
            return self.check()
        except Exception:
            return False

    def failed(self):
        return (V / 'echecs' / f'{self.jid}.txt').exists()


def mosei_job(cfg, a, method, d, seed, split, out):
    args = dict(a, seed=seed, results_dir=str(out), eval_split=split, only=method, **d)
    f = out / 'summary.json'
    return Job(f"{cfg}__{split}__{method}__{tag(d)}__s{seed}",
               [PY, '-u', str(R / 'ComprehensiveBenchmark_vs.py')] + cli(args), out,
               lambda: 'accuracy_matrix' in json.load(open(f))['results'][method])


def acc(M):
    return sum(M[-1]) / len(M[-1]) * 100


def select(scores, cur):
    best = max(scores.values())
    tied = [t for t, v in scores.items() if abs(v - best) < 1e-9]
    return cur if cur in tied else tied[0]


def plan():
    jobs, waiting, info = [], False, {}
    order = [CONFIGS[0], ('ch5', None)] + CONFIGS[1:]
    for cfg, src in order:
        if cfg == 'ch5':
            p1 = []
            for w in CH5_W:
                out = V / 'ch5' / 'sel' / f'w{w:g}'
                f = out / 'meld' / 'CL-MoE.json'
                p1.append(Job(f"ch5__val__CL-MoE__w{w:g}",
                              [PY, '-u', str(R / 'baselines_vs.py'), '--dataset', 'meld', '--method', 'CL-MoE',
                               '--kw', json.dumps({'momentum': 0.9, 'task_loss_weight': w}),
                               '--seeds', ','.join(map(str, SEL_SEEDS)), '--out', str(out)], out,
                              lambda f=f: len(json.load(open(f))) == len(SEL_SEEDS)))
            jobs += p1
            if not all(j.done() or j.failed() for j in p1):
                waiting = True
                continue
            if any(j.failed() for j in p1):
                info['ch5'] = 'ECHECS : pas de choix automatique'
                continue
            sc = {f'w{w:g}': sum(r['val'] for r in json.load(open(V / 'ch5' / 'sel' / f'w{w:g}' / 'meld' / 'CL-MoE.json'))) / len(SEL_SEEDS) * 100
                  for w in CH5_W}
            best = select(sc, 'w0.5')
            write_sel(V / 'ch5' / 'selection.json', {'CL-MoE': {'scores_val': sc, 'choix': best, 'actuel': 'w0.5'}})
            info['ch5'] = best
            if best != 'w0.5':
                w = float(best[1:])
                for ds in ('meld', 'iemocap'):
                    out = V / 'ch5' / 'final' / best
                    f = out / ds / 'CL-MoE.json'
                    jobs.append(Job(f"ch5__test__CL-MoE__{best}__{ds}",
                                    [PY, '-u', str(R / 'baselines_vs.py'), '--dataset', ds, '--method', 'CL-MoE',
                                     '--kw', json.dumps({'momentum': 0.9, 'task_loss_weight': w}),
                                     '--seeds', ','.join(map(str, TEST_SEEDS)), '--out', str(out)], out,
                                    lambda f=f: len(json.load(open(f))) == len(TEST_SEEDS)))
            continue
        a = orig_args(src)
        cur = current(a)
        p1 = [mosei_job(cfg, a, m, d, s, 'val', V / cfg / 'sel' / m / tag(d) / f'seed_{s}')
              for m, G in GRIDS.items() for d in G for s in SEL_SEEDS]
        jobs += p1
        if not all(j.done() or j.failed() for j in p1):
            waiting = True
            continue
        if any(j.failed() for j in p1):
            info[cfg] = 'ECHECS : pas de choix automatique'
            continue
        sel = {}
        for m, G in GRIDS.items():
            sc = {tag(d): sum(acc(json.load(open(V / cfg / 'sel' / m / tag(d) / f'seed_{s}' / 'summary.json'))['results'][m]['accuracy_matrix'])
                              for s in SEL_SEEDS) / len(SEL_SEEDS) for d in G}
            best = select(sc, tag(cur[m]))
            sel[m] = {'scores_val': sc, 'choix': best, 'actuel': tag(cur[m])}
            if best != tag(cur[m]):
                d = next(d for d in G if tag(d) == best)
                jobs += [mosei_job(cfg, a, m, d, s, 'test', V / cfg / 'final' / m / f'seed_{s}') for s in TEST_SEEDS]
        write_sel(V / cfg / 'selection.json', sel)
        info[cfg] = {m: (x['choix'], 'inchange' if x['choix'] == x['actuel'] else 'RELANCE') for m, x in sel.items()}
    return jobs, waiting, info


def write_sel(p, d):
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        t = p.with_suffix(f'.tmp{os.getpid()}')
        t.write_text(json.dumps(d, indent=2))
        os.replace(t, p)


def run(cmd, logf):
    logf.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(logf, 'w') as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(R)).returncode
    return rc, time.time() - t0


def equiv():
    a = orig_args('ch4_t0all30')
    m = 'ComplementarityMoE_Ours'
    ref = json.load(open(R / 'ch4_t0all30' / 'seed_10' / 'summary.json'))['results'][m]['accuracy_matrix']
    j = mosei_job('equiv', a, m, current(a)[m], 10, 'test', V / 'equiv' / 'b30_test_seed10')
    rc, dt = run(j.cmd, V / 'logs' / 'equiv_b30.log')
    print(f"1) Ch4 budget 30, graine 10, methode seule, reglages d'origine : {dt / 60:.1f} min, rc={rc}")
    if not j.done():
        (V / 'STOP').touch()
        sys.exit("ARRET : ce run n'a pas abouti. Voir validsel/logs/equiv_b30.log.")
    got = json.load(open(j.out / 'summary.json'))['results'][m]['accuracy_matrix']
    if got != ref:
        (V / 'STOP').touch()
        sys.exit("ARRET : le resultat n'est PAS identique au run d'origine. Rien ne sera lance.")
    print("   IDENTIQUE au run d'origine (matrice complete).")
    c = orig_args('ch3_10seeds')
    M = {}
    for sp in ('val', 'test'):
        j = mosei_job('equiv', c, m, current(c)[m], 11, sp, V / 'equiv' / f'ch3_{sp}_seed11')
        rc, dt = run(j.cmd, V / 'logs' / f'equiv_ch3_{sp}.log')
        if not j.done():
            (V / 'STOP').touch()
            sys.exit(f"ARRET : le run Ch3 sur '{sp}' n'a pas abouti (rc={rc}). Voir validsel/logs/equiv_ch3_{sp}.log")
        M[sp] = json.load(open(j.out / 'summary.json'))['results'][m]['accuracy_matrix']
        print(f"2) Ch3, graine 11, evaluation sur '{sp}' : {dt / 60:.1f} min, ACC {acc(M[sp]):.2f}")
    if M['val'] == M['test']:
        (V / 'STOP').touch()
        sys.exit("ARRET : validation et test donnent la meme matrice : l'evaluation sur la validation ne marche pas.")
    print("   La validation et le test donnent des matrices differentes : l'evaluation sur la validation marche.")
    (V / 'EQUIV_OK').touch()
    print("EQUIV_OK : on peut lancer les travailleurs.")


def lance():
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    if not (V / 'EQUIV_OK').exists():
        equiv()
    if (V / 'STOP').exists():
        sys.exit("ARRET : STOP present.")
    (R / 'run').mkdir(exist_ok=True)
    host = os.uname().nodename
    for i in range(1, n + 1):
        f = open(R / 'run' / f'validsel_{host}_{i}.log', 'a')
        subprocess.Popen([PY, '-u', str(Path(__file__).resolve()), 'work'], stdout=f, stderr=subprocess.STDOUT,
                         cwd=str(R), start_new_session=True)
        print(f"travailleur {i} lance : run/validsel_{host}_{i}.log", flush=True)


def work():
    if (V / 'STOP').exists() or not (V / 'EQUIV_OK').exists():
        sys.exit("ARRET : lance d'abord 'python validsel.py equiv' (et il doit finir par EQUIV_OK).")
    log("travailleur demarre")
    while True:
        if (V / 'STOP').exists():
            log("STOP trouve : arret"); return
        jobs, waiting, info = plan()
        todo = [j for j in jobs if not j.done() and not j.failed()]
        if not todo and not waiting:
            log(f"TRAVAILLEUR TERMINE {json.dumps(info)}"); return
        took = False
        for j in todo:
            lock = V / 'verrous' / j.jid
            try:
                lock.mkdir(parents=True)
            except FileExistsError:
                continue
            took = True
            log(f"debut {j.jid}")
            rc, dt = run(j.cmd, V / 'logs' / f'{j.jid}.log')
            if j.done():
                log(f"fin {j.jid} ({dt / 60:.1f} min)")
            else:
                (V / 'echecs').mkdir(parents=True, exist_ok=True)
                (V / 'echecs' / f'{j.jid}.txt').write_text(f"rc={rc}\n")
                log(f"ECHEC {j.jid} rc={rc} : voir validsel/logs/{j.jid}.log")
            break
        if not took:
            time.sleep(float(os.environ.get('VALIDSEL_SLEEP', '60')))


def etat():
    jobs, waiting, info = plan()
    by = {}
    for j in jobs:
        k = j.jid.split('__')[0] + ' ' + j.jid.split('__')[1]
        d = by.setdefault(k, [0, 0, 0])
        d[0] += j.done(); d[1] += j.failed(); d[2] += 1
    for k, (dn, fl, tot) in by.items():
        print(f"{k:12s} fini {dn:4d} / {tot:4d}   echecs {fl}")
    enc = [p.name for p in (V / 'verrous').glob('*') if not any(j.jid == p.name and (j.done() or j.failed()) for j in jobs)] if (V / 'verrous').exists() else []
    print(f"en cours (verrou sans resultat) : {len(enc)}")
    for k, v in info.items():
        print(f"choix {k} : {v}")


if __name__ == '__main__':
    {'prep': prep, 'equiv': equiv, 'lance': lance, 'work': work, 'etat': etat}[sys.argv[1]]()

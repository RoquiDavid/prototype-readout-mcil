# Runs A, E and C.
import sys, os, json, types
sys.path.insert(0, '/data/roqui')
import numpy as np, torch
import torch.nn.functional as F
from pathlib import Path
from continual_learning_v3_clffix import ComplementarityMoE_V3_ClfFix
from models_meld import PerceiverIO_MoE
from meld_dataset import load_meld_class_incremental
from iemocap_dataset import load_iemocap_class_incremental
from fecam_classifier import FeCAMMemory

SMOKE = os.environ.get('SMOKE') == '1'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS = [10] if SMOKE else [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
CFG = dict(r=16, lora_up_init_std=0.0, num_experts=3, entropy_weight=0.1,
           primary_loss_weight=0.5, lr=1e-3, classifier_lr_mult=0.05,
           epochs_per_task=1 if SMOKE else 15, epochs_task0=1 if SMOKE else 30)
DATA = {
    'meld':    dict(loader=load_meld_class_incremental, data_dir='/data/roqui/meld_data',
                    text_dim=600, audio_dim=300),
    'iemocap': dict(loader=load_iemocap_class_incremental, data_dir='/data/roqui/iemocap_data',
                    text_dim=100, audio_dim=100),
}
DATASETS = ['meld'] if SMOKE else ['meld', 'iemocap']
OUT = Path('/data/roqui/nuit_smoke' if SMOKE else '/data/roqui/nuit')
REGIMES = ['full', 'no_audio', 'no_text']


def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(s)


def load(ds, seed):
    return DATA[ds]['loader'](data_dir=DATA[ds]['data_dir'], batch_size=32, seed=seed)


def build(ds, num_experts=None, use_text=True, use_audio=True):
    return PerceiverIO_MoE(num_classes=7, d_model=64,
        num_experts=num_experts or CFG['num_experts'], r=CFG['r'], alpha=2 * CFG['r'],
        tau=0.2, num_latents=4, lora_up_init_std=CFG['lora_up_init_std'],
        text_dim=DATA[ds]['text_dim'], audio_dim=DATA[ds]['audio_dim'], video_dim=35,
        use_text=use_text, use_audio=use_audio, use_video=False).to(DEVICE)


def learner(m, old_expert_lr_mult=0.01):
    return ComplementarityMoE_V3_ClfFix(m, DEVICE, lambda_barlow=0.0, tau=0.2,
        epochs_task0=CFG['epochs_task0'], primary_loss_weight=CFG['primary_loss_weight'],
        entropy_weight=CFG['entropy_weight'], classifier_lr_mult=CFG['classifier_lr_mult'],
        old_expert_lr_mult=old_expert_lr_mult)


def train_task(m, L, tr, va, tid):
    if tid > 0:
        m.router.reset_routing_head()
    L.train_task(tr[tid], va[tid], epochs=CFG['epochs_per_task'], lr=CFG['lr'])


@torch.no_grad()
def acc_tasks(m, predict, te):
    m.eval(); accs = []
    for loader in te:
        c = n = 0
        for b in loader:
            y = FeCAMMemory._labels(b, DEVICE); p = predict(b)
            c += (p == y).sum().item(); n += y.size(0)
        accs.append(c / n if n else 0.)
    return float(np.mean(accs))


def fwd_mask(self, text, audio, video):
    B = text.size(0)
    inp = torch.cat([self.text_proj(text.mean(1, keepdim=True)),
                     self.audio_proj(audio.mean(1, keepdim=True))], 1)
    mask = torch.zeros(B, 2, dtype=torch.bool, device=text.device)
    if self.training and self.p_drop > 0:
        r = torch.rand(B, device=text.device)
        mask[:, 0] = r < self.p_drop / 2
        mask[:, 1] = (r >= self.p_drop / 2) & (r < self.p_drop)
    if self.force == 'no_audio': mask[:, 1] = True
    if self.force == 'no_text':  mask[:, 0] = True
    lat = self.latent.expand(B, -1, -1)
    a, _ = self.cross_attention(lat, inp, inp, key_padding_mask=mask)
    lat = self.norm1(lat + a); lat = self.norm2(lat + self.ffn(lat))
    return lat.mean(1)


def run_A(ds, seed, p):
    set_seed(seed); tr, va, te = load(ds, seed); T = len(tr); set_seed(seed)
    m = build(ds); enc = m.perceiver_encoder
    enc.p_drop = p; enc.force = None; enc.forward = types.MethodType(fwd_mask, enc)
    L = learner(m)
    banks = {r: FeCAMMemory(shrink=0.05) for r in REGIMES}
    for tid in range(T):
        train_task(m, L, tr, va, tid)
        for r in REGIMES:
            enc.force = None if r == 'full' else r
            banks[r].update(m, tr[tid], DEVICE)
        enc.force = None
    res = {}
    for r in REGIMES:
        enc.force = None if r == 'full' else r
        res['single_' + r] = acc_tasks(m, lambda b: banks['full'].predict(m, b, DEVICE), te)
        res['matched_' + r] = acc_tasks(m, lambda b, r=r: banks[r].predict(m, b, DEVICE), te)
    enc.force = None
    return res


class ExpertFeCAM(FeCAMMemory):
    def __init__(self, shrink=0.05):
        super().__init__(shrink=shrink); self.cls_expert = {}

    @torch.no_grad()
    def _expert_feat(self, model, batch, device, k):
        h = model.perceiver_encoder(batch['text'].to(device), batch['audio'].to(device),
                                    batch['video'].to(device))
        return F.normalize(model.experts[k](h), dim=1)

    def update_expert(self, model, loader, device, k):
        before = set(self.mean)
        self._features = lambda mdl, b, dev: self._expert_feat(mdl, b, dev, k)
        try:
            self.update(model, loader, device)
        finally:
            del self._features
        for c in set(self.mean) - before:
            self.cls_expert[c] = k

    @torch.no_grad()
    def predict(self, model, batch, device):
        h = model.perceiver_encoder(batch['text'].to(device), batch['audio'].to(device),
                                    batch['video'].to(device))
        classes = sorted(self.mean); feats = {}; dists = []
        for c in classes:
            k = self.cls_expert[c]
            if k not in feats:
                feats[k] = F.normalize(model.experts[k](h), dim=1)
            diff = feats[k] - self.mean[c]
            dists.append(((diff @ self.prec[c]) * diff).sum(1))
        idx = torch.stack(dists, 1).argmin(1)
        return torch.tensor(classes, device=device)[idx]


def run_E(ds, seed):
    set_seed(seed); tr, va, te = load(ds, seed); T = len(tr); set_seed(seed)
    m = build(ds, num_experts=T)
    L = learner(m, old_expert_lr_mult=0.0)
    enc_bank = FeCAMMemory(shrink=0.05); exp_bank = ExpertFeCAM(shrink=0.05)
    for tid in range(T):
        train_task(m, L, tr, va, tid)
        enc_bank.update(m, tr[tid], DEVICE)
        exp_bank.update_expert(m, tr[tid], DEVICE, tid)
    return {'encoder_space': acc_tasks(m, lambda b: enc_bank.predict(m, b, DEVICE), te),
            'expert_space':  acc_tasks(m, lambda b: exp_bank.predict(m, b, DEVICE), te)}


def run_C(ds, seed, mod):
    set_seed(seed); tr, va, te = load(ds, seed); T = len(tr); set_seed(seed)
    m = build(ds, use_text=(mod == 'text'), use_audio=(mod == 'audio'))
    L = learner(m); fe = FeCAMMemory(shrink=0.05)
    for tid in range(T):
        train_task(m, L, tr, va, tid)
        fe.update(m, tr[tid], DEVICE)
    return acc_tasks(m, lambda b: fe.predict(m, b, DEVICE), te)


def flat(d, pre=''):
    out = {}
    for k, v in d.items():
        if k == 'seed': continue
        if isinstance(v, dict): out.update(flat(v, pre + k + '/'))
        else: out[pre + k] = v
    return out


def summarize(run, ds, R):
    from scipy.stats import wilcoxon
    F_ = [flat(r) for r in R]
    print(f"\n===== RUN {run} | {ds} | {len(R)} graines (moyenne des taches, test) =====", flush=True)
    for k in F_[0]:
        a = np.array([f[k] for f in F_]) * 100
        print(f"{k:<26}{a.mean():7.2f} +/- {a.std():.2f}", flush=True)
    pairs = []
    if run == 'A':
        pairs = [(f'{p}/matched_{r}', f'{p}/single_{r}') for p in ['0.0', '0.3'] for r in REGIMES[1:]]
    if run == 'E':
        pairs = [('expert_space', 'encoder_space')]
    for a_k, b_k in pairs:
        a = np.array([f[a_k] for f in F_]); b = np.array([f[b_k] for f in F_])
        try: pv = f"p={wilcoxon(a, b).pvalue:.4f}"
        except Exception as e: pv = f"test impossible ({e.__class__.__name__})"
        print(f"  {a_k} - {b_k} : {100*(a-b).mean():+.2f} pts | Wilcoxon {pv}", flush=True)


if __name__ == '__main__':
    run = sys.argv[1]
    assert run in ('A', 'E', 'C'), 'argument attendu : A, E ou C'
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"RUN {run} | device {DEVICE} | SMOKE={SMOKE} | CFG {CFG}", flush=True)
    for ds in DATASETS:
        R = []
        for seed in SEEDS:
            if run == 'A': r = {str(p): run_A(ds, seed, p) for p in (0.0, 0.3)}
            elif run == 'E': r = run_E(ds, seed)
            else: r = {mod: run_C(ds, seed, mod) for mod in ('text', 'audio')}
            r['seed'] = seed; R.append(r)
            json.dump(R, open(OUT / f'{run}_{ds}.json', 'w'), indent=2)
            print(f"[{run}] {ds} seed {seed} fait", flush=True)
        summarize(run, ds, R)
    print(f"\nRUN {run} TERMINE" + (" -- SMOKE OK" if SMOKE else ""), flush=True)

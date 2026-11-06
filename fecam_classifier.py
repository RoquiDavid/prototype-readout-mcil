# FeCAM prototype readout.
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


class FeCAMMemory:
    def __init__(self, shrink: float = 0.2):
        self.shrink = shrink
        self.mean = {}
        self.prec = {}

    @staticmethod
    def _labels(batch, device):
        return torch.clamp(batch['labels'].to(device)[:, 0], -3, 3).round().long() + 3

    @torch.no_grad()
    def _features(self, model, batch, device):
        h = model.perceiver_encoder(
            batch['text'].to(device), batch['audio'].to(device), batch['video'].to(device))
        return F.normalize(h, dim=1)

    @torch.no_grad()
    def update(self, model, train_loader, device):
        was_training = model.training
        model.eval()
        feats_by_class = {}
        for batch in train_loader:
            h = self._features(model, batch, device)
            y = self._labels(batch, device)
            for c in torch.unique(y):
                c = int(c)
                feats_by_class.setdefault(c, []).append(h[y == c])
        for c, chunks in feats_by_class.items():
            X = torch.cat(chunks, 0)
            mu = X.mean(0)
            Xc = X - mu
            cov = (Xc.T @ Xc) / max(X.size(0) - 1, 1)
            d = cov.size(0)
            scaled_I = (cov.trace() / d) * torch.eye(d, device=device)
            cov_reg = (1 - self.shrink) * cov + self.shrink * scaled_I
            cov_reg = cov_reg + 1e-5 * torch.eye(d, device=device)
            self.mean[c] = mu.detach()
            self.prec[c] = torch.linalg.inv(cov_reg).detach()
        if was_training:
            model.train()
        print(f"  [FeCAM] classes mises a jour : {sorted(feats_by_class)}")

    @torch.no_grad()
    def predict(self, model, batch, device):
        h = self._features(model, batch, device)
        classes = sorted(self.mean)
        dists = []
        for c in classes:
            diff = h - self.mean[c]
            dists.append(((diff @ self.prec[c]) * diff).sum(1))
        idx = torch.stack(dists, dim=1).argmin(1)
        return torch.tensor(classes, device=device)[idx]


@torch.no_grad()
def evaluate_task_fecam(model, test_loader, fecam_mem, device):
    model.eval()
    all_preds, all_labels = [], []
    correct = total = 0
    for batch in test_loader:
        preds = fecam_mem.predict(model, batch, device)
        y = FeCAMMemory._labels(batch, device)
        correct += (preds == y).sum().item(); total += y.size(0)
        all_preds.extend(preds.cpu().numpy()); all_labels.extend(y.cpu().numpy())
    return {
        'accuracy': correct / total,
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
    }

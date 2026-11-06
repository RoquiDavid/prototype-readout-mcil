# Nearest class mean readout.
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


class PrototypeMemory:
    def __init__(self, num_classes: int = 7, space: str = 'frozen'):
        self.num_classes = num_classes
        self.space = space
        self.prototypes = {}

    @torch.no_grad()
    def _features(self, model, batch, device):
        t = batch['text'].to(device)
        a = batch['audio'].to(device)
        v = batch['video'].to(device)
        if self.space == 'frozen':
            return model.perceiver_encoder(t, a, v)
        return model(t, a, v)['features']

    @staticmethod
    def _labels(batch, device):
        return torch.clamp(batch['labels'].to(device)[:, 0], -3, 3).round().long() + 3

    @torch.no_grad()
    def update(self, model, train_loader, device):
        was_training = model.training
        model.eval()
        sums, counts = {}, {}
        for batch in train_loader:
            h = self._features(model, batch, device)
            y = self._labels(batch, device)
            for c in torch.unique(y):
                c = int(c); m = (y == c)
                sums[c] = sums.get(c, 0) + h[m].sum(0)
                counts[c] = counts.get(c, 0) + int(m.sum())
        for c in sums:
            self.prototypes[c] = (sums[c] / counts[c]).detach()
        if was_training:
            model.train()
        print(f"  [NCM] prototypes mis a jour : classes {sorted(self.prototypes)}")

    @torch.no_grad()
    def predict(self, features, device):
        classes = sorted(self.prototypes)
        P = F.normalize(torch.stack([self.prototypes[c] for c in classes]).to(device), dim=1)
        h = F.normalize(features, dim=1)
        idx = (h @ P.T).argmax(1)
        return torch.tensor(classes, device=device)[idx]


@torch.no_grad()
def evaluate_task_ncm(model, test_loader, proto_memory, device):
    model.eval()
    all_preds, all_labels = [], []
    correct = total = 0
    for batch in test_loader:
        h = proto_memory._features(model, batch, device)
        preds = proto_memory.predict(h, device)
        y = proto_memory._labels(batch, device)
        correct += (preds == y).sum().item(); total += y.size(0)
        all_preds.extend(preds.cpu().numpy()); all_labels.extend(y.cpu().numpy())
    return {
        'accuracy': correct / total,
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
    }

# IEMOCAP loading and task split.
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

TASK_SPLIT = [
    [5],
    [2],
    [0, 4],
    [3],
    [1],
]


class _UttDataset(Dataset):
    def __init__(self, text, audio, labels):
        self.text = torch.tensor(np.array(text), dtype=torch.float)
        self.audio = torch.tensor(np.array(audio), dtype=torch.float)
        self.labels = torch.tensor(np.array(labels), dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        c = self.labels[i]
        return {
            'text': self.text[i].unsqueeze(0),
            'audio': self.audio[i].unsqueeze(0),
            'video': torch.zeros(1, 35),
            'labels': (c - 3).unsqueeze(0),
        }


def load_iemocap_class_incremental(data_dir='/data/roqui/iemocap_data',
                                   batch_size=256, seed=11,
                                   pkl_name='IEMOCAP_features_raw.pkl'):
    path = f'{data_dir}/{pkl_name}'
    d = pickle.load(open(path, 'rb'), encoding='latin1')
    videoLabels, videoText, videoAudio = d[2], d[3], d[4]
    trainVid, testVid = list(d[7]), list(d[8])

    def flatten(vids):
        T, A, Y = [], [], []
        for v in sorted(vids):
            for i, lab in enumerate(videoLabels[v]):
                T.append(np.asarray(videoText[v][i], dtype=np.float32))
                A.append(np.asarray(videoAudio[v][i], dtype=np.float32))
                Y.append(int(lab))
        return T, A, Y

    _tv = sorted(trainVid)
    val_vids = [v for k, v in enumerate(_tv) if k % 10 == 0]
    tr_vids  = [v for k, v in enumerate(_tv) if k % 10 != 0]
    trT2, trA2, trY2 = flatten(tr_vids)
    vaT, vaA, vaY    = flatten(val_vids)
    teT, teA, teY    = flatten(testVid)
    trT, trA, trY    = trT2, trA2, trY2
    assert set(range(6)) <= set(vaY), 'une classe est absente du val'
    print(f"[IEMOCAP] dialogues: train={len(tr_vids)} val={len(val_vids)} test={len(testVid)}")

    def task_loaders(T, A, Y, shuffle):
        loaders = []
        for classes in TASK_SPLIT:
            sel = [i for i, c in enumerate(Y) if c in classes]
            ds = _UttDataset([T[i] for i in sel], [A[i] for i in sel], [Y[i] for i in sel])
            loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False))
        return loaders

    print(f"[IEMOCAP] train={len(trY2)} val={len(vaY)} test={len(teY)} "
          f"| text {np.array(trT[0]).shape[0]}D audio {np.array(trA[0]).shape[0]}D | {len(TASK_SPLIT)} taches")
    return (task_loaders(trT2, trA2, trY2, True),
            task_loaders(vaT, vaA, vaY, False),
            task_loaders(teT, teA, teY, False))


if __name__ == '__main__':
    E = {0:'happy',1:'sad',2:'neutral',3:'angry',4:'excited',5:'frustrated'}
    tr, va, te = load_iemocap_class_incremental()
    b0 = next(iter(tr[0]))
    print("text", tuple(b0['text'].shape), "| audio", tuple(b0['audio'].shape),
          "| labels", tuple(b0['labels'].shape))
    ok = True
    for t, cls in enumerate(TASK_SPLIT):
        seen, n = set(), {}
        for nm, L in (('train',tr),('val',va),('test',te)):
            c = 0
            for batch in L[t]:
                y = torch.clamp(batch['labels'][:,0],-3,3).round().long()+3
                if nm == 'train': seen |= set(y.tolist())
                c += y.size(0)
            n[nm] = c
        good = (seen == set(cls)); ok = ok and good
        print(f"[{'OK   ' if good else 'ECHEC'}] T{t} {'+'.join(E[i] for i in cls):<20}"
              f" train={n['train']:>5} val={n['val']:>4} test={n['test']:>4}")
    print("RESULTAT :", "tout est bon" if ok else "PROBLEME, ne pas lancer")

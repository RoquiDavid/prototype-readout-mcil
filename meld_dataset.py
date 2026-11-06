# MELD loading and task split.
import pickle
import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


EMOTION_TO_IDX = {
    'anger': 0,
    'disgust': 1,
    'fear': 2,
    'joy': 3,
    'neutral': 4,
    'sadness': 5,
    'surprise': 6,
}


def _load_labels(csv_path):
    labels_by_dialogue = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            did = str(row['Dialogue_ID'])
            uid = int(row['Utterance_ID'])
            emotion = row['Emotion'].strip().lower()
            if emotion not in EMOTION_TO_IDX:
                continue
            eidx = EMOTION_TO_IDX[emotion]
            if did not in labels_by_dialogue:
                labels_by_dialogue[did] = []
            labels_by_dialogue[did].append((uid, eidx))
    for did in labels_by_dialogue:
        labels_by_dialogue[did].sort(key=lambda x: x[0])
    return labels_by_dialogue


def _flatten_split(text_features, audio_features, labels_by_dialogue):
    texts = []
    audios = []
    labels = []

    for did in text_features:
        if did not in labels_by_dialogue:
            continue
        text_arr = text_features[did]
        audio_arr = audio_features[did]
        utt_labels = labels_by_dialogue[did]

        n_feat = text_arr.shape[0]
        n_lab = len(utt_labels)
        n = min(n_feat, n_lab)

        for i in range(n):
            texts.append(text_arr[i])
            audios.append(audio_arr[i])
            labels.append(utt_labels[i][1])

    return np.array(texts), np.array(audios), np.array(labels)


class MELDDataset(Dataset):

    def __init__(self, text_features, audio_features, labels):
        self.text = torch.FloatTensor(text_features)
        self.audio = torch.FloatTensor(audio_features)
        self.labels = torch.FloatTensor(labels.astype(np.float32) - 3.0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'text': self.text[idx].unsqueeze(0),
            'audio': self.audio[idx].unsqueeze(0),
            'video': torch.zeros(1, 35),
            'labels': self.labels[idx].unsqueeze(0),
        }


def load_meld_continual(data_dir: str = '/data/roqui/meld_data',
                        num_tasks: int = 5,
                        batch_size: int = 32,
                        seed: int = 42):
    data_dir = Path(data_dir)
    features_dir = data_dir / 'features' / 'features'

    print("Loading MELD dataset...")

    with open(features_dir / os.environ.get('MELD_TEXT_PKL', 'text_emotion.pkl'), 'rb') as f:
        text_data = pickle.load(f, encoding='latin1')
    with open(features_dir / 'audio_emotion.pkl', 'rb') as f:
        audio_data = pickle.load(f, encoding='latin1')


    csv_dir = data_dir / 'MELD' / 'data' / 'MELD'
    train_labels = _load_labels(csv_dir / 'train_sent_emo.csv')
    val_labels = _load_labels(csv_dir / 'dev_sent_emo.csv')
    test_labels = _load_labels(csv_dir / 'test_sent_emo.csv')

    train_text, train_audio, train_y = _flatten_split(
        text_data[0], audio_data[0], train_labels)
    val_text, val_audio, val_y = _flatten_split(
        text_data[1], audio_data[1], val_labels)
    test_text, test_audio, test_y = _flatten_split(
        text_data[2], audio_data[2], test_labels)

    print(f"Dataset loaded: Train={len(train_y)}, Val={len(val_y)}, Test={len(test_y)}")
    print(f"  Text dim={train_text.shape[1]}, Audio dim={train_audio.shape[1]}")
    print(f"  Emotions: {sorted(EMOTION_TO_IDX.keys())}")

    for split_name, split_y in [('Train', train_y), ('Val', val_y), ('Test', test_y)]:
        unique, counts = np.unique(split_y, return_counts=True)
        dist = {EMOTION_TO_IDX: c for EMOTION_TO_IDX, c in zip(unique, counts)}
        print(f"  {split_name} class dist: {dist}")

    rng = np.random.RandomState(seed)

    train_perm = rng.permutation(len(train_y))
    train_text = train_text[train_perm]
    train_audio = train_audio[train_perm]
    train_y = train_y[train_perm]

    val_perm = rng.permutation(len(val_y))
    val_text = val_text[val_perm]
    val_audio = val_audio[val_perm]
    val_y = val_y[val_perm]

    test_perm = rng.permutation(len(test_y))
    test_text = test_text[test_perm]
    test_audio = test_audio[test_perm]
    test_y = test_y[test_perm]

    print(f"\nCreating {num_tasks} task splits...")

    def _split_into_tasks(text_arr, audio_arr, labels_arr, num_tasks):
        n = len(labels_arr)
        task_sizes = [n // num_tasks] * num_tasks
        for i in range(n % num_tasks):
            task_sizes[i] += 1

        datasets = []
        idx = 0
        for t, size in enumerate(task_sizes):
            t_text = text_arr[idx:idx+size]
            t_audio = audio_arr[idx:idx+size]
            t_labels = labels_arr[idx:idx+size]
            datasets.append(MELDDataset(t_text, t_audio, t_labels))
            print(f"  Task {t}: {size} samples")
            idx += size
        return datasets

    train_datasets = _split_into_tasks(train_text, train_audio, train_y, num_tasks)
    val_datasets = _split_into_tasks(val_text, val_audio, val_y, num_tasks)
    test_datasets = _split_into_tasks(test_text, test_audio, test_y, num_tasks)

    train_loaders = [
        DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
        for ds in train_datasets
    ]
    val_loaders = [
        DataLoader(ds, batch_size=batch_size, shuffle=False)
        for ds in val_datasets
    ]
    test_loaders = [
        DataLoader(ds, batch_size=batch_size, shuffle=False)
        for ds in test_datasets
    ]

    return train_loaders, val_loaders, test_loaders


DEFAULT_CI_TASK_EMOTIONS = [
    ['neutral'],
    ['joy'],
    ['surprise', 'disgust'],
    ['anger', 'fear'],
    ['sadness'],
]


def load_meld_class_incremental(data_dir: str = '/data/roqui/meld_data',
                                batch_size: int = 32,
                                seed: int = 42,
                                task_emotions: list = None):
    if task_emotions is None:
        task_emotions = DEFAULT_CI_TASK_EMOTIONS

    num_tasks = len(task_emotions)

    all_emotions = [e for task in task_emotions for e in task]
    assert len(all_emotions) == len(set(all_emotions)), \
        'Une émotion apparaît dans plusieurs tâches'
    for emo in all_emotions:
        assert emo in EMOTION_TO_IDX, f'Émotion inconnue : {emo}'

    data_dir = Path(data_dir)
    features_dir = data_dir / 'features' / 'features'

    print("Loading MELD dataset (CLASS-INCREMENTAL mode)...")
    print("Task split:")
    for t, emotions in enumerate(task_emotions):
        print(f"  T{t}: {emotions}")

    with open(features_dir / os.environ.get('MELD_TEXT_PKL', 'text_emotion.pkl'), 'rb') as f:
        text_data = pickle.load(f, encoding='latin1')
    with open(features_dir / 'audio_emotion.pkl', 'rb') as f:
        audio_data = pickle.load(f, encoding='latin1')

    csv_dir = data_dir / 'MELD' / 'data' / 'MELD'
    train_labels_by_dlg = _load_labels(csv_dir / 'train_sent_emo.csv')
    val_labels_by_dlg = _load_labels(csv_dir / 'dev_sent_emo.csv')
    test_labels_by_dlg = _load_labels(csv_dir / 'test_sent_emo.csv')

    train_text, train_audio, train_y = _flatten_split(
        text_data[0], audio_data[0], train_labels_by_dlg)
    val_text, val_audio, val_y = _flatten_split(
        text_data[1], audio_data[1], val_labels_by_dlg)
    test_text, test_audio, test_y = _flatten_split(
        text_data[2], audio_data[2], test_labels_by_dlg)

    rng = np.random.RandomState(seed)

    def _filter_and_build(text_arr, audio_arr, labels_arr, split_name):
        datasets = []
        for t, emotions in enumerate(task_emotions):
            class_ids = [EMOTION_TO_IDX[e] for e in emotions]
            mask = np.isin(labels_arr, class_ids)
            t_text = text_arr[mask]
            t_audio = audio_arr[mask]
            t_labels = labels_arr[mask]

            perm = rng.permutation(len(t_labels))
            t_text = t_text[perm]
            t_audio = t_audio[perm]
            t_labels = t_labels[perm]

            datasets.append(MELDDataset(t_text, t_audio, t_labels))
            print(f"  {split_name:5s} T{t}: {len(t_labels):5d} samples")
        return datasets

    print("\nBuilding task datasets:")
    train_datasets = _filter_and_build(train_text, train_audio, train_y, 'Train')
    val_datasets = _filter_and_build(val_text, val_audio, val_y, 'Val')
    test_datasets = _filter_and_build(test_text, test_audio, test_y, 'Test')

    train_loaders = [
        DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
        for ds in train_datasets
    ]
    val_loaders = [
        DataLoader(ds, batch_size=batch_size, shuffle=False)
        for ds in val_datasets
    ]
    test_loaders = [
        DataLoader(ds, batch_size=batch_size, shuffle=False)
        for ds in test_datasets
    ]

    return train_loaders, val_loaders, test_loaders

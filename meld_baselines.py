# Replay and replay-free baselines.
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from copy import deepcopy
import numpy as np
import random


def _compute_class_weights(train_loader, device):
    class_counts = torch.zeros(7, device=device)
    for batch in train_loader:
        labels = torch.clamp(batch['labels'].to(device)[:, 0], -3, 3).round().long() + 3
        for label in labels:
            class_counts[label] += 1
    total = class_counts.sum()
    weights = total / (class_counts + 1.0)
    weights = torch.clamp(weights, max=10.0)
    weights = weights / weights.sum() * 7
    return weights


class ReplayBuffer:
    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self.buffer = []
        self.seen = 0

    def add(self, text, audio, video, labels, logits=None):
        batch_size = text.size(0)
        for i in range(batch_size):
            sample = {
                'text': text[i].cpu(),
                'audio': audio[i].cpu(),
                'video': video[i].cpu(),
                'labels': labels[i].cpu(),
            }
            if logits is not None:
                sample['logits'] = logits[i].cpu()

            self.seen += 1
            if len(self.buffer) < self.max_size:
                self.buffer.append(sample)
            else:
                idx = random.randint(0, self.seen - 1)
                if idx < self.max_size:
                    self.buffer[idx] = sample

    def sample(self, n: int, device: str = 'cuda'):
        n = min(n, len(self.buffer))
        if n == 0:
            return None
        indices = random.sample(range(len(self.buffer)), n)
        samples = [self.buffer[i] for i in indices]
        batch = {
            'text': torch.stack([s['text'] for s in samples]).to(device),
            'audio': torch.stack([s['audio'] for s in samples]).to(device),
            'video': torch.stack([s['video'] for s in samples]).to(device),
            'labels': torch.stack([s['labels'] for s in samples]).to(device),
        }
        if 'logits' in samples[0]:
            batch['logits'] = torch.stack([s['logits'] for s in samples]).to(device)
        return batch

    def __len__(self):
        return len(self.buffer)


class Naive:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.task_id = 0

    def train_task(self, train_loader, val_loader, epochs, lr):
        class_weights = _compute_class_weights(train_loader, self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            n = 0
            for batch in train_loader:
                text = batch['text'].to(self.device)
                audio = batch['audio'].to(self.device)
                video = batch['video'].to(self.device)
                labels = torch.clamp(batch['labels'].to(self.device)[:, 0], -3, 3).round().long() + 3

                optimizer.zero_grad()
                loss = nn.CrossEntropyLoss(weight=class_weights)(
                    self.model(text, audio, video)['logits'], labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n += 1
            if epoch % 5 == 0:
                print(f"    Epoch {epoch+1}/{epochs}: Loss={total_loss/n:.4f}")
        self.task_id += 1


class EWC:
    def __init__(self, model, device='cuda', lambda_ewc=1000.0):
        self.model = model
        self.device = device
        self.lambda_ewc = lambda_ewc
        self.task_id = 0
        self.fisher = {}
        self.optpar = {}

    def _compute_fisher(self, train_loader):
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        for batch in train_loader:
            text = batch['text'].to(self.device)
            audio = batch['audio'].to(self.device)
            video = batch['video'].to(self.device)
            labels = torch.clamp(batch['labels'].to(self.device)[:, 0], -3, 3).round().long() + 3
            self.model.zero_grad()
            loss = nn.CrossEntropyLoss()(self.model(text, audio, video)['logits'], labels)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data ** 2
        for n in fisher:
            fisher[n] /= len(train_loader)
        return fisher

    def train_task(self, train_loader, val_loader, epochs, lr):
        class_weights = _compute_class_weights(train_loader, self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            n = 0
            for batch in train_loader:
                text = batch['text'].to(self.device)
                audio = batch['audio'].to(self.device)
                video = batch['video'].to(self.device)
                labels = torch.clamp(batch['labels'].to(self.device)[:, 0], -3, 3).round().long() + 3

                optimizer.zero_grad()
                ce = nn.CrossEntropyLoss(weight=class_weights)(
                    self.model(text, audio, video)['logits'], labels)
                ewc_loss = 0.0
                if self.task_id > 0:
                    for nn_name, param in self.model.named_parameters():
                        if nn_name in self.fisher:
                            ewc_loss += (self.fisher[nn_name] * (param - self.optpar[nn_name]) ** 2).sum()
                loss = ce + self.lambda_ewc * ewc_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n += 1
            if epoch % 5 == 0:
                print(f"    Epoch {epoch+1}/{epochs}: Loss={total_loss/n:.4f}")

        self.fisher = self._compute_fisher(train_loader)
        self.optpar = {n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad}
        self.task_id += 1


class LwF:
    def __init__(self, model, device='cuda', lambda_kd=1.0, temperature=2.0):
        self.model = model
        self.device = device
        self.lambda_kd = lambda_kd
        self.temperature = temperature
        self.task_id = 0
        self.teacher = None

    def train_task(self, train_loader, val_loader, epochs, lr):
        class_weights = _compute_class_weights(train_loader, self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            n = 0
            for batch in train_loader:
                text = batch['text'].to(self.device)
                audio = batch['audio'].to(self.device)
                video = batch['video'].to(self.device)
                labels = torch.clamp(batch['labels'].to(self.device)[:, 0], -3, 3).round().long() + 3

                optimizer.zero_grad()
                outputs = self.model(text, audio, video)
                ce = nn.CrossEntropyLoss(weight=class_weights)(outputs['logits'], labels)

                kd_loss = 0.0
                if self.teacher is not None:
                    with torch.no_grad():
                        teacher_out = self.teacher(text, audio, video)
                    kd_loss = F.kl_div(
                        F.log_softmax(outputs['logits'] / self.temperature, dim=-1),
                        F.softmax(teacher_out['logits'] / self.temperature, dim=-1),
                        reduction='batchmean'
                    ) * (self.temperature ** 2)

                loss = ce + self.lambda_kd * kd_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n += 1
            if epoch % 5 == 0:
                print(f"    Epoch {epoch+1}/{epochs}: Loss={total_loss/n:.4f}")

        self.teacher = deepcopy(self.model)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()
        self.task_id += 1


class ER:
    def __init__(self, model, device='cuda', buffer_size=200):
        self.model = model
        self.device = device
        self.buffer = ReplayBuffer(buffer_size)
        self.task_id = 0

    def train_task(self, train_loader, val_loader, epochs, lr):
        class_weights = _compute_class_weights(train_loader, self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            n = 0
            for batch in train_loader:
                text = batch['text'].to(self.device)
                audio = batch['audio'].to(self.device)
                video = batch['video'].to(self.device)
                labels = torch.clamp(batch['labels'].to(self.device)[:, 0], -3, 3).round().long() + 3

                optimizer.zero_grad()
                loss = nn.CrossEntropyLoss(weight=class_weights)(
                    self.model(text, audio, video)['logits'], labels)

                if len(self.buffer) > 0:
                    replay = self.buffer.sample(min(32, len(self.buffer)), self.device)
                    r_labels = torch.clamp(replay['labels'][:, 0], -3, 3).round().long() + 3
                    r_loss = nn.CrossEntropyLoss()(
                        self.model(replay['text'], replay['audio'], replay['video'])['logits'],
                        r_labels)
                    loss = loss + r_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n += 1

                self.buffer.add(text, audio, video, batch['labels'].to(self.device))

            if epoch % 5 == 0:
                print(f"    Epoch {epoch+1}/{epochs}: Loss={total_loss/n:.4f}")
        self.task_id += 1


class DERpp:
    def __init__(self, model, device='cuda', buffer_size=200, alpha=0.5, beta=0.5):
        self.model = model
        self.device = device
        self.buffer = ReplayBuffer(buffer_size)
        self.alpha = alpha
        self.beta = beta
        self.task_id = 0

    def train_task(self, train_loader, val_loader, epochs, lr):
        class_weights = _compute_class_weights(train_loader, self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            n = 0
            for batch in train_loader:
                text = batch['text'].to(self.device)
                audio = batch['audio'].to(self.device)
                video = batch['video'].to(self.device)
                labels = torch.clamp(batch['labels'].to(self.device)[:, 0], -3, 3).round().long() + 3

                optimizer.zero_grad()
                outputs = self.model(text, audio, video)
                loss = nn.CrossEntropyLoss(weight=class_weights)(outputs['logits'], labels)

                if len(self.buffer) > 0:
                    replay = self.buffer.sample(min(32, len(self.buffer)), self.device)
                    r_out = self.model(replay['text'], replay['audio'], replay['video'])

                    if 'logits' in replay:
                        logit_loss = F.mse_loss(r_out['logits'], replay['logits'])
                        loss = loss + self.alpha * logit_loss

                    r_labels = torch.clamp(replay['labels'][:, 0], -3, 3).round().long() + 3
                    label_loss = nn.CrossEntropyLoss()(r_out['logits'], r_labels)
                    loss = loss + self.beta * label_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n += 1

                with torch.no_grad():
                    stored_logits = outputs['logits'].detach()
                self.buffer.add(text, audio, video, batch['labels'].to(self.device),
                                logits=stored_logits)

            if epoch % 5 == 0:
                print(f"    Epoch {epoch+1}/{epochs}: Loss={total_loss/n:.4f}")
        self.task_id += 1


class AGEM:
    def __init__(self, model, device='cuda', buffer_size=200):
        self.model = model
        self.device = device
        self.buffer = ReplayBuffer(buffer_size)
        self.task_id = 0

    def _project_gradient(self, grad_current, grad_ref):
        dot = (grad_current * grad_ref).sum()
        if dot < 0:
            proj = grad_current - (dot / (grad_ref.norm() ** 2 + 1e-8)) * grad_ref
            return proj
        return grad_current

    def train_task(self, train_loader, val_loader, epochs, lr):
        class_weights = _compute_class_weights(train_loader, self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            n = 0
            for batch in train_loader:
                text = batch['text'].to(self.device)
                audio = batch['audio'].to(self.device)
                video = batch['video'].to(self.device)
                labels = torch.clamp(batch['labels'].to(self.device)[:, 0], -3, 3).round().long() + 3

                optimizer.zero_grad()
                loss = nn.CrossEntropyLoss(weight=class_weights)(
                    self.model(text, audio, video)['logits'], labels)
                loss.backward()

                if len(self.buffer) > 0:
                    replay = self.buffer.sample(min(32, len(self.buffer)), self.device)
                    r_labels = torch.clamp(replay['labels'][:, 0], -3, 3).round().long() + 3

                    current_grads = {n: p.grad.clone() for n, p in self.model.named_parameters()
                                     if p.grad is not None}

                    optimizer.zero_grad()
                    r_loss = nn.CrossEntropyLoss()(
                        self.model(replay['text'], replay['audio'], replay['video'])['logits'],
                        r_labels)
                    r_loss.backward()

                    for nn_name, param in self.model.named_parameters():
                        if param.grad is not None and nn_name in current_grads:
                            param.grad.data = self._project_gradient(
                                current_grads[nn_name], param.grad.data)

                optimizer.step()
                total_loss += loss.item()
                n += 1

                self.buffer.add(text, audio, video, batch['labels'].to(self.device))

            if epoch % 5 == 0:
                print(f"    Epoch {epoch+1}/{epochs}: Loss={total_loss/n:.4f}")
        self.task_id += 1


class DMoLE:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.task_id = 0

    def train_task(self, train_loader, val_loader, epochs, lr):
        class_weights = _compute_class_weights(train_loader, self.device)
        num_experts = len(self.model.experts)
        primary = self.task_id % num_experts

        for i, expert in enumerate(self.model.experts):
            for p in expert.parameters():
                p.requires_grad = (i >= self.task_id)
        for p in self.model.perceiver_encoder.parameters():
            p.requires_grad = True
        for p in self.model.router.parameters():
            p.requires_grad = True
        for p in self.model.classifier.parameters():
            p.requires_grad = True
        for p in self.model.experts[primary].parameters():
            p.requires_grad = True

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=1e-4)
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            n = 0
            for batch in train_loader:
                text = batch['text'].to(self.device)
                audio = batch['audio'].to(self.device)
                video = batch['video'].to(self.device)
                labels = torch.clamp(batch['labels'].to(self.device)[:, 0], -3, 3).round().long() + 3

                optimizer.zero_grad()
                loss = nn.CrossEntropyLoss(weight=class_weights)(
                    self.model(text, audio, video)['logits'], labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n += 1
            if epoch % 5 == 0:
                print(f"    Epoch {epoch+1}/{epochs}: Loss={total_loss/n:.4f}")
        self.task_id += 1


class CLMoE:
    def __init__(self, model, device='cuda', momentum=0.9):
        self.model = model
        self.device = device
        self.momentum = momentum
        self.task_id = 0
        self.ema_experts = None

    def _init_ema(self):
        self.ema_experts = [
            {k: v.clone() for k, v in expert.state_dict().items()}
            for expert in self.model.experts
        ]

    def _update_ema(self):
        if self.ema_experts is None:
            self._init_ema()
        else:
            for i, expert in enumerate(self.model.experts):
                for key in self.ema_experts[i]:
                    self.ema_experts[i][key] = (
                        self.momentum * self.ema_experts[i][key] +
                        (1 - self.momentum) * expert.state_dict()[key]
                    )

    def train_task(self, train_loader, val_loader, epochs, lr):
        class_weights = _compute_class_weights(train_loader, self.device)
        num_experts = len(self.model.experts)
        expert_id = self.task_id % num_experts

        for i, expert in enumerate(self.model.experts):
            for p in expert.parameters():
                p.requires_grad = (i == expert_id)
        for p in self.model.router.parameters():
            p.requires_grad = True
        for p in self.model.classifier.parameters():
            p.requires_grad = True
        for p in self.model.perceiver_encoder.parameters():
            p.requires_grad = True

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=1e-4)
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            n = 0
            for batch in train_loader:
                text = batch['text'].to(self.device)
                audio = batch['audio'].to(self.device)
                video = batch['video'].to(self.device)
                labels = torch.clamp(batch['labels'].to(self.device)[:, 0], -3, 3).round().long() + 3

                optimizer.zero_grad()
                outputs = self.model(text, audio, video)
                ce = nn.CrossEntropyLoss(weight=class_weights)(outputs['logits'], labels)

                w = outputs['expert_weights']
                task_loss = -torch.log(w[:, expert_id] + 1e-8).mean()
                entropy = -(w * torch.log(w + 1e-8)).sum(-1).mean()

                loss = ce + 0.5 * task_loss - 0.1 * entropy
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n += 1

            self._update_ema()
            if epoch % 5 == 0:
                print(f"    Epoch {epoch+1}/{epochs}: Loss={total_loss/n:.4f}")
        self.task_id += 1


class ProgLoRA:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.task_id = 0
        self.task_expert_map = {}

    def train_task(self, train_loader, val_loader, epochs, lr):
        class_weights = _compute_class_weights(train_loader, self.device)
        num_experts = len(self.model.experts)
        expert_id = self.task_id % num_experts
        self.task_expert_map[self.task_id] = expert_id
        print(f"  Task {self.task_id} -> new expert {expert_id}")

        for i, expert in enumerate(self.model.experts):
            for p in expert.parameters():
                p.requires_grad = (i == expert_id)
        for p in self.model.router.parameters():
            p.requires_grad = True
        for p in self.model.classifier.parameters():
            p.requires_grad = True
        for p in self.model.perceiver_encoder.parameters():
            p.requires_grad = True

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=1e-4)
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            n = 0
            for batch in train_loader:
                text = batch['text'].to(self.device)
                audio = batch['audio'].to(self.device)
                video = batch['video'].to(self.device)
                labels = torch.clamp(batch['labels'].to(self.device)[:, 0], -3, 3).round().long() + 3

                optimizer.zero_grad()
                loss = nn.CrossEntropyLoss(weight=class_weights)(
                    self.model(text, audio, video)['logits'], labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n += 1
            if epoch % 5 == 0:
                print(f"    Epoch {epoch+1}/{epochs}: Loss={total_loss/n:.4f}")
        self.task_id += 1

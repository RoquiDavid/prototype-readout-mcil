# Training of the method.
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from copy import deepcopy
import numpy as np


class ContinualLearner:
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.task_id = 0
        self.class_weights = None

    def set_class_weights(self, train_loader):
        class_counts = torch.zeros(7, device=self.device)
        for batch in train_loader:
            labels_continuous = batch['labels'].to(self.device)
            labels = torch.clamp(labels_continuous[:, 0], -3, 3).round().long() + 3
            for label in labels:
                class_counts[label] += 1

        present_mask = class_counts > 0
        n_present = present_mask.sum().item()

        weights = torch.ones(7, device=self.device)

        if n_present > 1:
            w = torch.ones(7, device=self.device)
            present_counts = class_counts[present_mask]
            present_w = present_counts.sum() / (present_counts + 1.0)
            present_w = torch.clamp(present_w, max=10.0)
            present_w = present_w / present_w.mean()
            w[present_mask] = present_w
            weights = w

        self.class_weights = weights
        print(f"  Class weights: {weights.cpu().numpy()} (present={n_present}/7)")

    def train_task(self, train_loader, val_loader, epochs: int, lr: float):
        raise NotImplementedError


def _pairwise_barlow_twins(expert_outputs: torch.Tensor, tau: float,
                           lambda_bt: float) -> torch.Tensor:
    B, N, D = expert_outputs.shape
    if B < 2:
        return torch.tensor(0.0, device=expert_outputs.device)
    bt_loss = torch.tensor(0.0, device=expert_outputs.device)
    num_pairs = 0

    for i in range(N):
        for j in range(i + 1, N):
            ei = expert_outputs[:, i, :]
            ej = expert_outputs[:, j, :]

            ei_norm = (ei - ei.mean(0)) / (ei.std(0) + 1e-8)
            ej_norm = (ej - ej.mean(0)) / (ej.std(0) + 1e-8)

            c = (ei_norm.T @ ej_norm) / B

            on_diag  = ((torch.diagonal(c) - tau) ** 2).sum()
            off_diag = (c ** 2).sum() - (torch.diagonal(c) ** 2).sum()

            bt_loss += on_diag + lambda_bt * off_diag
            num_pairs += 1

    return bt_loss / max(num_pairs, 1)


class ComplementarityMoE_V3_ClfFix:

    def __init__(self, model: nn.Module, device: str = 'cuda',
                 lambda_barlow: float = 0.01, tau: float = 0.2,
                 epochs_task0: int = None,
                 primary_loss_weight: float = 1.0,
                 entropy_weight: float = 0.1,
                 classifier_lr_mult: float = 0.0,
                 old_expert_lr_mult: float = 0.01):
        self.model = model
        self.device = device
        self.lambda_barlow = lambda_barlow
        self.tau = tau
        self.task_id = 0
        self.router_lr_mult = 3.0
        self.primary_loss_weight = primary_loss_weight
        self.entropy_weight = entropy_weight
        self.classifier_lr_mult = classifier_lr_mult
        self.old_expert_lr_mult = old_expert_lr_mult
        self.epochs_task0 = epochs_task0
        self.routing_history = []
        self.class_weights = None

    def set_class_weights(self, train_loader):
        class_counts = torch.zeros(7, device=self.device)
        for batch in train_loader:
            labels_continuous = batch['labels'].to(self.device)
            labels = torch.clamp(labels_continuous[:, 0], -3, 3).round().long() + 3
            for label in labels:
                class_counts[label] += 1

        present_mask = class_counts > 0
        n_present = present_mask.sum().item()

        weights = torch.ones(7, device=self.device)

        if n_present > 1:
            w = torch.ones(7, device=self.device)
            present_counts = class_counts[present_mask]
            present_w = present_counts.sum() / (present_counts + 1.0)
            present_w = torch.clamp(present_w, max=10.0)
            present_w = present_w / present_w.mean()
            w[present_mask] = present_w
            weights = w

        self.class_weights = weights
        print(f"  Class weights: {weights.cpu().numpy()} (present={n_present}/7)")

    def _get_expert_outputs(self, perceiver_features: torch.Tensor) -> torch.Tensor:
        expert_outs = []
        for expert in self.model.experts:
            expert_outs.append(expert(perceiver_features))
        return torch.stack(expert_outs, dim=1)

    def train_task(self, train_loader, val_loader, epochs: int, lr: float):
        self.set_class_weights(train_loader)

        num_experts = len(self.model.experts)
        primary_expert = self.task_id % num_experts
        print(f"  Primary expert for Task {self.task_id}: Expert {primary_expert}")

        actual_epochs = epochs
        if self.task_id == 0 and self.epochs_task0 is not None:
            actual_epochs = self.epochs_task0
            print(f"  [Task 0] Using {actual_epochs} epochs")

        classifier_trainable = self.classifier_lr_mult > 0

        if self.task_id > 0:
            for param in self.model.perceiver_encoder.parameters():
                param.requires_grad = False

            for param in self.model.classifier.parameters():
                param.requires_grad = classifier_trainable

            for i, expert in enumerate(self.model.experts):
                if i == primary_expert:
                    for p in expert.parameters():
                        p.requires_grad = True
                elif i < self.task_id:
                    for p in expert.parameters():
                        p.requires_grad = True
                else:
                    for p in expert.parameters():
                        p.requires_grad = False

            print(f"  [FROZEN] Perceiver frozen")
            if classifier_trainable:
                print(f"  [CLASSIFIER] Trainable at {self.classifier_lr_mult}×LR")
            else:
                print(f"  [CLASSIFIER] Frozen")
            print(f"  [EXPERTS] Primary={primary_expert} (1×LR), "
                  f"Old={[i for i in range(num_experts) if i < self.task_id and i != primary_expert]} (0.01×LR), "
                  f"Future={list(range(max(self.task_id, primary_expert + 1), num_experts))} (frozen)")
        else:
            for param in self.model.parameters():
                param.requires_grad = True

        trainable_params = []

        trainable_params.append({
            'params': self.model.router.parameters(),
            'lr': lr * self.router_lr_mult
        })

        trainable_params.append({
            'params': self.model.experts[primary_expert].parameters(),
            'lr': lr
        })

        if self.task_id > 0:
            for i in range(self.task_id):
                if i != primary_expert and i < num_experts:
                    trainable_params.append({
                        'params': self.model.experts[i].parameters(),
                        'lr': lr * self.old_expert_lr_mult
                    })
            if classifier_trainable:
                trainable_params.append({
                    'params': self.model.classifier.parameters(),
                    'lr': lr * self.classifier_lr_mult
                })
        else:
            trainable_params.insert(0, {
                'params': self.model.perceiver_encoder.parameters(),
                'lr': lr
            })
            trainable_params.append({
                'params': self.model.classifier.parameters(),
                'lr': lr
            })

        optimizer = torch.optim.AdamW(trainable_params, weight_decay=1e-4)
        print(f"  [DEBUG] Optimizer groups for task {self.task_id}:")
        for i, group in enumerate(optimizer.param_groups):
            n_params = sum(p.numel() for p in group['params'])
            print(f"    Group {i}: lr={group['lr']:.6e}, n_params={n_params}")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=actual_epochs, eta_min=lr * 0.01
        )

        self.model.train()
        epoch_routing_data = []

        clf_w_before = self.model.classifier[4].weight.detach().clone()

        for epoch in range(actual_epochs):
            total_loss = 0
            num_batches = 0

            for batch in train_loader:
                text  = batch['text'].to(self.device)
                audio = batch['audio'].to(self.device)
                video = batch['video'].to(self.device)
                labels_continuous = batch['labels'].to(self.device)
                labels = torch.clamp(labels_continuous[:, 0], -3, 3).round().long() + 3

                optimizer.zero_grad()
                outputs = self.model(text, audio, video)

                ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)(
                    outputs['logits'], labels
                )

                expert_weights = outputs['expert_weights']
                primary_loss = -torch.log(
                    expert_weights[:, primary_expert] + 1e-8
                ).mean()
                entropy = -(
                    expert_weights * torch.log(expert_weights + 1e-8)
                ).sum(dim=-1).mean()

                total_loss_batch = (
                    ce_loss
                    + self.primary_loss_weight * primary_loss
                    - self.entropy_weight * entropy
                )

                if self.lambda_barlow > 0:
                    perceiver_features = outputs['perceiver_features']
                    expert_outputs_stack = self._get_expert_outputs(perceiver_features)
                    bt_loss = _pairwise_barlow_twins(
                        expert_outputs_stack, self.tau, self.lambda_barlow
                    )
                    total_loss_batch = total_loss_batch + self.lambda_barlow * bt_loss

                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += total_loss_batch.item()
                num_batches += 1

                if epoch == actual_epochs - 1:
                    epoch_routing_data.append(
                        expert_weights.detach().cpu().numpy()
                    )

            scheduler.step()

            avg_loss = total_loss / num_batches
            if epoch % 2 == 0:
                print(f"    Epoch {epoch+1}/{actual_epochs}: Loss={avg_loss:.4f}")

        clf_w_after = self.model.classifier[4].weight.detach()
        clf_delta = (clf_w_after - clf_w_before).abs().sum().item()
        print(f"  [DEBUG] Classifier final Linear weight delta sum: {clf_delta:.6e}")

        if epoch_routing_data:
            all_weights = np.concatenate(epoch_routing_data, axis=0)
            self.routing_history.append({
                'task_id': self.task_id,
                'mean_weights': all_weights.mean(axis=0).tolist(),
                'std_weights': all_weights.std(axis=0).tolist(),
                'primary_expert': primary_expert,
                'primary_weight_mean': float(
                    all_weights[:, primary_expert].mean()
                ),
            })
            print(f"  [Routing] Mean weights: "
                  f"{[f'{w:.3f}' for w in all_weights.mean(axis=0)]}")
            print(f"  [Routing] Primary expert {primary_expert} weight: "
                  f"{all_weights[:, primary_expert].mean():.3f}")

        self.task_id += 1


ComplementarityMoE_ClfFix = ComplementarityMoE_V3_ClfFix

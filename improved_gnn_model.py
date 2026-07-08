#!/usr/bin/env python3
"""
改进的GNN模型。
关键修正：
1. evaluate 支持验证集搜索阈值、测试集固定阈值；
2. 阈值默认按验证集 MCC 选择；
3. 测试集评估时不会再用测试集标签反选阈值，避免数据泄露。
"""
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
)
import numpy as np


class ImprovedResidualBlock(nn.Module):
    """改进的残差块。"""
    def __init__(self, in_dim, out_dim, dropout=0.5):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(dropout * 0.5),
        )
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return F.relu(self.linear(x) + self.shortcut(x))


class ImprovedBindingSiteGNN(nn.Module):
    """蛋白质-DNA结合位点预测GNN。"""

    def __init__(self, input_dim=1280, hidden_dim=256, dropout=0.5,
                 use_focal_loss=True, focal_alpha=0.25, focal_gamma=2.0, pos_weight=1.5):
        super().__init__()

        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.best_threshold = 0.5

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
        )

        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout, concat=False)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)

        self.res_blocks = nn.ModuleList([
            ImprovedResidualBlock(hidden_dim, hidden_dim, dropout) for _ in range(1)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

        if use_focal_loss:
            self.loss_fn = self.focal_loss
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

        # 温度参数用于概率校准
        self.temperature = nn.Parameter(torch.ones(1))

    def focal_loss(self, pred, target):
        """Focal Loss。"""
        ce_loss = F.binary_cross_entropy_with_logits(pred, target.float(), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.input_proj(x)
        identity = x

        x1 = F.elu(self.conv1(x, edge_index))
        x2 = F.elu(self.conv2(x, edge_index))
        x3 = F.elu(self.conv3(x, edge_index))
        x = x1 + x2 + x3 + identity

        for block in self.res_blocks:
            x = block(x)

        logits = self.classifier(x).squeeze()

        if self.training:
            return logits
        return logits / self.temperature.clamp_min(1e-6)

    def train_model(self, train_data, val_data, epochs=30, lr=5e-4, device='cpu', patience=5):
        """普通训练接口：验证集按 MCC 选阈值和早停。"""
        self.to(device)
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=1e-3,
            betas=(0.9, 0.999),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.7, patience=3
        )

        best_val_mcc = -1e9
        best_val_auc = 0.0
        best_state = None
        no_improve = 0

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            batch_count = 0

            for data in train_data:
                if data.x.size(0) == 0:
                    continue

                data = data.to(device)
                if (data.y == 1).sum().item() == 0:
                    continue

                optimizer.zero_grad()
                out = self(data)
                loss = self.loss_fn(out, data.y.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

            avg_loss = total_loss / batch_count if batch_count > 0 else 0.0

            val_metrics = self.evaluate(
                val_data,
                device=device,
                search_threshold=True,
                threshold_metric='mcc',
            )
            scheduler.step(val_metrics['mcc'])

            print(
                f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | "
                f"Val MCC: {val_metrics['mcc']:.4f} | Val F1: {val_metrics['f1']:.4f} | "
                f"Val AUC-PR: {val_metrics['auc_pr']:.4f} | "
                f"Thr: {val_metrics['threshold']:.2f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

            if val_metrics['mcc'] > best_val_mcc:
                best_val_mcc = val_metrics['mcc']
                best_val_auc = val_metrics['auc_pr']
                self.best_threshold = val_metrics['threshold']
                best_state = copy.deepcopy(self.state_dict())
                torch.save(best_state, "best_improved_gnn_model.pt")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        if best_state is not None:
            self.load_state_dict(best_state)
        elif os.path.exists("best_improved_gnn_model.pt"):
            self.load_state_dict(torch.load("best_improved_gnn_model.pt", map_location=device))

        print(
            f"Training complete. Best Val MCC: {best_val_mcc:.4f}, "
            f"Best Val AUC-PR: {best_val_auc:.4f}, "
            f"Fixed Threshold: {self.best_threshold:.2f}"
        )
        return best_val_auc, best_val_mcc

    def evaluate(self, dataset, device='cpu', threshold=None, search_threshold=False, threshold_metric='mcc'):
        """
        评估函数。
        - search_threshold=True：只允许用于验证集，按验证集标签搜索最优阈值；
        - search_threshold=False：用于测试集，必须使用固定阈值，不再扫描测试集阈值。
        """
        empty_result = {
            'f1': 0.0,
            'mcc': 0.0,
            'auc_pr': 0.0,
            'accuracy': 0.0,
            'balanced_accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'sensitivity': 0.0,
            'specificity': 0.0,
            'auc_roc': 0.0,
            'threshold': float(threshold if threshold is not None else getattr(self, 'best_threshold', 0.5)),
            'confusion_matrix': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
            'total_samples': 0,
            'positive_samples': 0,
            'negative_samples': 0,
            'positive_ratio': 0.0,
        }
        if not dataset:
            return empty_result

        self.eval()
        self.to(device)

        all_probs = []
        all_labels = []

        with torch.no_grad():
            for data in dataset:
                if data.x.size(0) == 0:
                    continue

                data = data.to(device)
                out = self(data)
                probs = torch.sigmoid(out)

                all_probs.extend(probs.detach().cpu().tolist())
                all_labels.extend(data.y.detach().cpu().tolist())

        if len(all_labels) == 0:
            return empty_result

        all_labels = [int(label) for label in all_labels]
        all_probs = [float(prob) for prob in all_probs]

        def compute_metrics(th):
            preds = [1 if p > th else 0 for p in all_probs]

            accuracy = accuracy_score(all_labels, preds)
            balanced_acc = balanced_accuracy_score(all_labels, preds)
            precision = precision_score(all_labels, preds, zero_division=0)
            recall = recall_score(all_labels, preds, zero_division=0)
            f1 = f1_score(all_labels, preds, zero_division=0)
            mcc = matthews_corrcoef(all_labels, preds)

            tn, fp, fn, tp = confusion_matrix(all_labels, preds, labels=[0, 1]).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            auc_pr = 0.0
            auc_roc = 0.0
            if len(set(all_labels)) > 1:
                try:
                    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
                    auc_pr = auc(recall_curve, precision_curve)
                    auc_roc = roc_auc_score(all_labels, all_probs)
                except Exception:
                    auc_pr = 0.0
                    auc_roc = 0.0

            return {
                'f1': float(f1),
                'mcc': float(mcc),
                'auc_pr': float(auc_pr),
                'accuracy': float(accuracy),
                'balanced_accuracy': float(balanced_acc),
                'precision': float(precision),
                'recall': float(recall),
                'sensitivity': float(recall),
                'specificity': float(specificity),
                'auc_roc': float(auc_roc),
                'threshold': float(th),
                'confusion_matrix': {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)},
                'total_samples': int(len(all_labels)),
                'positive_samples': int(sum(all_labels)),
                'negative_samples': int(len(all_labels) - sum(all_labels)),
                'positive_ratio': float(sum(all_labels) / len(all_labels)) if len(all_labels) > 0 else 0.0,
            }

        if search_threshold:
            thresholds = np.arange(0.05, 0.96, 0.01)
            best_threshold = 0.5
            best_value = -1e9

            for th in thresholds:
                metrics = compute_metrics(float(th))
                if threshold_metric == 'mcc':
                    value = metrics['mcc']
                elif threshold_metric == 'f1':
                    value = metrics['f1']
                elif threshold_metric == 'balanced_accuracy':
                    value = metrics['balanced_accuracy']
                else:
                    value = metrics['mcc']

                if value > best_value:
                    best_value = value
                    best_threshold = float(th)

            self.best_threshold = best_threshold
            return compute_metrics(best_threshold)

        if threshold is None:
            threshold = getattr(self, 'best_threshold', 0.5)

        return compute_metrics(float(threshold))

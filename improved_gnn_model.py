import torch
from sklearn.metrics import precision_recall_curve, auc
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score,
                             roc_auc_score, confusion_matrix, balanced_accuracy_score,
                             matthews_corrcoef)
import numpy as np
import os

class ImprovedResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.5):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(dropout * 0.5)
        )
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return F.relu(self.linear(x) + self.shortcut(x))

class ImprovedBindingSiteGNN(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=256, dropout=0.5,
                 use_focal_loss=True, focal_alpha=0.25, focal_gamma=2.0, pos_weight=1.5):
        super().__init__()

        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7)
        )

        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout, concat=False)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)

        self.res_blocks = nn.ModuleList([ImprovedResidualBlock(hidden_dim, hidden_dim, dropout)])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1)
        )

        if use_focal_loss:
            self.loss_fn = self.focal_loss
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

        self.temperature = nn.Parameter(torch.ones(1))

    def focal_loss(self, pred, target):
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
        else:
            return logits / self.temperature

    def train_model(self, train_data, val_data, epochs=30, lr=5e-4, device='cpu', patience=5):
        self.to(device)

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=1e-3,
            betas=(0.9, 0.999)
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.7, patience=3
        )

        best_val_auc = 0
        best_val_f1 = 0
        no_improve = 0

        for epoch in range(epochs):
            self.train()
            total_loss = 0
            batch_count = 0

            for data in train_data:
                if data.x.size(0) == 0:
                    continue

                data = data.to(device)
                optimizer.zero_grad()
                out = self(data)

                if (data.y == 1).sum().item() == 0:
                    continue

                loss = self.loss_fn(out, data.y.float())
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)

                optimizer.step()
                total_loss += loss.item()
                batch_count += 1

            avg_loss = total_loss / batch_count if batch_count > 0 else 0

            val_metrics = self.evaluate(val_data, device)
            val_f1 = val_metrics['f1']
            val_auc_pr = val_metrics['auc_pr']

            print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Val F1: {val_f1:.4f} | Val AUC-PR: {val_auc_pr:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

            scheduler.step(val_auc_pr)

            if val_auc_pr > best_val_auc:
                best_val_auc = val_auc_pr
                best_val_f1 = val_f1
                torch.save(self.state_dict(), "best_improved_gnn_model.pt")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        if os.path.exists("best_improved_gnn_model.pt"):
            self.load_state_dict(torch.load("best_improved_gnn_model.pt"))
        print(f"Training complete. Best Val AUC-PR: {best_val_auc:.4f}, Best Val F1: {best_val_f1:.4f}")
        return best_val_auc, best_val_f1

    def evaluate(self, dataset, device='cpu'):
        if not dataset:
            return {'f1': 0, 'mcc': 0, 'auc_pr': 0}

        self.eval()
        self.to(device)
        all_preds = []  # 存储所有预测标签
        all_probs = []  # 存储所有预测概率
        all_labels = []  # 存储所有真实标签

        with torch.no_grad():
            for data in dataset:
                if data.x.size(0) == 0:
                    continue

                data = data.to(device)
                out = self(data)
                probs = torch.sigmoid(out)  # 得到预测概率
                preds = torch.round(probs)  # 将概率值转换为0或1的预测标签

                all_preds.extend(preds.cpu().tolist())  # 保存预测标签
                all_probs.extend(probs.cpu().tolist())  # 保存预测概率
                all_labels.extend(data.y.cpu().tolist())  # 保存真实标签

        all_labels = [int(label) for label in all_labels]

        auc_pr = float('nan')
        auc_roc = float('nan')

        if any(label == 1 for label in all_labels):
            try:
                precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
                auc_pr = auc(recall_curve, precision_curve)

                auc_roc = roc_auc_score(all_labels, all_probs)
            except:
                pass

        # 计算混淆矩阵相关指标
        try:
            tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        except:
            specificity = 0.0

        # 计算MCC
        try:
            mcc = matthews_corrcoef(all_labels, all_preds)
        except:
            mcc = 0.0

        return {
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'specificity': specificity,
            'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
            'mcc': mcc,
            'auc_pr': auc_pr,
            'auc_roc': auc_roc
        }

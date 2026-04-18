import torch
from sklearn.metrics import precision_recall_curve, auc
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score,
                             roc_auc_score, confusion_matrix, balanced_accuracy_score)
import numpy as np
import os
import torch
import random
import numpy as np
def set_seed(seed):
    """设置随机种子确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 设置Python哈希种子
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
class ResidualBlock(nn.Module):
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

    def forward(self, x):
        return F.relu(self.linear(x) + self.shortcut(x))

class BindingSiteGNN(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=512, dropout=0.3):
        super().__init__()
        # 输入投影层
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 图卷积层（混合GAT和GCN）
        self.conv1 = GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)

        # 残差块
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dim, hidden_dim, dropout) for _ in range(2)])

        # 输出层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(256),
            nn.Linear(256, 1)
        )

        # 使用加权交叉熵损失
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # 输入投影
        x = self.input_proj(x)

        # 多类型图卷积
        x1 = F.elu(self.conv1(x, edge_index))
        x2 = F.elu(self.conv2(x, edge_index))
        x3 = F.elu(self.conv3(x, edge_index))
        x = x1 + x2 + x3  # 合并图卷积结果

        # 残差块
        for block in self.res_blocks:
            x = block(x)

        return self.classifier(x).squeeze()

    def train_model(self, train_data, val_data, epochs=100, lr=1e-3, device='cpu', patience=10):
        self.to(device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_auc = 0
        best_val_f1 = 0
        no_improve = 0

        for epoch in range(epochs):
            self.train()
            total_loss = 0
            batch_count = 0

            for data in train_data:
                # 跳过空图
                if data.x.size(0) == 0:
                    continue

                data = data.to(device)
                optimizer.zero_grad()
                out = self(data)

                # 跳过全负样本的图
                if (data.y == 1).sum().item() == 0:
                    continue

                loss = self.loss_fn(out, data.y.float())
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

                optimizer.step()
                total_loss += loss.item()
                batch_count += 1

            if batch_count == 0:
                avg_loss = 0
            else:
                avg_loss = total_loss / batch_count

            # 更新学习率
            scheduler.step()

            # 验证
            val_metrics = self.evaluate(val_data, device)
            val_f1 = val_metrics['f1']
            val_auc_pr = val_metrics['auc_pr']

            print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Val F1: {val_f1:.4f} | Val AUC-PR: {val_auc_pr:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

            # 保存最佳模型
            if val_auc_pr > best_val_auc:
                best_val_auc = val_auc_pr
                best_val_f1 = val_f1
                torch.save(self.state_dict(), "best_gnn_model.pt")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # 加载最佳模型
        if os.path.exists("best_gnn_model.pt"):
            try:
                self.load_state_dict(torch.load("best_gnn_model.pt"))
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    print(f"模型尺寸不匹配，跳过加载旧模型: {e}")
                    os.remove("best_gnn_model.pt")
                else:
                    raise e
        print(f"Training complete. Best Val AUC-PR: {best_val_auc:.4f}, Best Val F1: {best_val_f1:.4f}")
        return best_val_auc, best_val_f1

    def evaluate(self, dataset, device='cpu'):
        if not dataset:
            return {'f1': 0, 'mcc': 0, 'auc_pr': 0}

        self.eval()
        self.to(device)
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for data in dataset:
                # 跳过空图
                if data.x.size(0) == 0:
                    continue

                data = data.to(device)
                out = self(data)
                probs = torch.sigmoid(out)

                all_probs.extend(probs.cpu().tolist())
                all_labels.extend(data.y.cpu().tolist())

        if len(all_labels) == 0:
            return {'f1': 0, 'mcc': 0, 'auc_pr': 0}

        # 确保标签是整数
        all_labels = [int(label) for label in all_labels]

        # 计算各种指标
        auc_pr = float('nan')
        auc_roc = float('nan')

        if any(label == 1 for label in all_labels):
            try:
                # AUC-PR
                precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
                auc_pr = auc(recall_curve, precision_curve)

                # AUC-ROC
                auc_roc = roc_auc_score(all_labels, all_probs)
            except:
                pass

        return {
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'accuracy': accuracy_score(all_labels, all_preds),
            'auc_pr': auc_pr,
            'auc_roc': auc_roc
        }

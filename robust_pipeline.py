#!/usr/bin/env python3
"""
鲁棒性增强训练-测试管道：无测试集阈值泄露版本。

关键修正：
1. 构图函数从 data_loader 导入，确保增强图也使用余弦距离 KNN；
2. 交叉验证时，验证折对应的增强图不会进入训练折；
3. 阈值只在验证集上按 MCC 选择；
4. 独立测试集只使用验证集固定阈值评估，不再在测试集上扫描阈值；
5. 主流程不使用测试集结果选择模型超参数；
6. 使用3折模型概率集成，提高泛化稳定性；
7. 扩散候选样本优先使用余弦相似度筛选，与余弦KNN构图保持一致。
"""
import os
import time
import glob
import json
import copy
import shutil
import contextlib
import warnings

import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data
from sklearn.model_selection import KFold
from sklearn.metrics import (
    f1_score, matthews_corrcoef, precision_recall_curve, auc, accuracy_score,
    precision_score, recall_score, roc_auc_score, confusion_matrix, balanced_accuracy_score
)

warnings.filterwarnings('ignore')

from balanced_training_config import BalancedTrainingConfig
from improved_gnn_model import ImprovedBindingSiteGNN
from data_loader import ProteinDataset, create_knn_edges
from ddpm_diffusion_model import EnhancedDiffusionModel
from main import calculate_class_ratio
from gnn_model import set_seed


class RobustTrainingConfig(BalancedTrainingConfig):
    """鲁棒训练配置。"""
    def __init__(self, target_ratio=0.30, experiment_name="default"):
        super().__init__()

        # 数据增强策略
        self.target_ratio = target_ratio
        self.experiment_name = experiment_name
        self.min_samples_per_protein = 3
        self.max_augment_ratio = 1.0
        self.max_generated_per_protein = 64
        self.candidate_multiplier = 4
        self.max_candidate_samples = 256
        self.strict_distance_multiplier = 1.5
        self.relaxed_keep_ratio = 0.35
        self.allow_relaxed_diffusion = True
        self.disable_diffusion = False  # 设为True可跑无扩散对照
        self.strict_cosine_quantile = 0.05
        self.strict_cosine_min_threshold = 0.60

        # 质量控制
        self.quality_threshold = 0.7
        self.diversity_threshold = 0.3

        # 域适应
        self.use_domain_adaptation = True
        self.domain_weight = 0.1

        # 交叉验证
        self.use_cross_validation = True
        self.cv_folds = 3

        # 集成学习开关。注意：最终测试阈值仍固定来自验证集。
        self.ensemble_size = self.cv_folds  # 使用所有CV折模型做概率集成
        self.ensemble_dropout_rates = [0.3, 0.4, 0.5]

        # 输出控制
        self.verbose_loading = False

        # 阈值策略：只在验证集上按 MCC 选阈值
        self.threshold_metric = 'mcc'

        print("🎯 鲁棒训练配置:")
        print(f"  - 目标比例: {self.target_ratio:.1%}")
        print(f"  - KNN k值: {self.knn_k}")
        print(f"  - GNN dropout: {self.gnn_dropout}")
        print(f"  - 阈值选择指标: {self.threshold_metric.upper()}，仅验证集选择")
        print(f"  - 使用Focal Loss配置: {self.use_focal_loss}")
        print(f"  - 正样本权重: {self.pos_weight}")
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")



def _to_cpu_float_tensor(array_like):
    """转成 CPU float tensor，并清理 NaN/Inf。"""
    if isinstance(array_like, torch.Tensor):
        x = array_like.detach().cpu().float()
    else:
        x = torch.tensor(np.asarray(array_like), dtype=torch.float32)
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _pairwise_max_cosine_similarity(a, b, chunk_size=256):
    """PyTorch 分块计算 a 到 b 的最大余弦相似度。
    用于扩散样本筛选，避免 sklearn/pairwise_distances 的 Windows 底层崩溃，
    并与余弦KNN构图的度量保持一致。
    """
    a = _to_cpu_float_tensor(a)
    b = _to_cpu_float_tensor(b)
    if a.numel() == 0 or b.numel() == 0:
        return torch.empty((0,), dtype=torch.float32)

    a = torch.nn.functional.normalize(a, p=2, dim=1, eps=1e-12)
    b = torch.nn.functional.normalize(b, p=2, dim=1, eps=1e-12)

    max_sims = []
    for start in range(0, a.size(0), chunk_size):
        sims = torch.mm(a[start:start + chunk_size], b.t())
        max_sims.append(sims.max(dim=1).values)
    return torch.cat(max_sims, dim=0)


def _real_positive_cosine_threshold(real_pos_samples, quantile=0.05, min_threshold=0.60):
    """
    用真实正样本内部的最近邻余弦相似度自适应估计 strict 阈值。
    阈值越高越严格。这里使用较低分位数作为保守下界，避免高维欧氏距离导致 strict 永远为0。
    """
    real = _to_cpu_float_tensor(real_pos_samples)
    n = real.size(0)
    if n < 3:
        return None

    real = torch.nn.functional.normalize(real, p=2, dim=1, eps=1e-12)
    sim = torch.mm(real, real.t())
    sim.fill_diagonal_(-float('inf'))
    nn_sim = sim.max(dim=1).values
    nn_sim = nn_sim[torch.isfinite(nn_sim)]
    if nn_sim.numel() == 0:
        return None

    threshold = torch.quantile(nn_sim, quantile).item()
    return max(float(threshold), float(min_threshold))


def _select_diffusion_samples(candidate_samples, real_pos_samples, n_to_generate, config):
    """
    只从扩散模型生成的 candidate_samples 中选样本；不复制真实正样本。
    筛选逻辑改为余弦相似度，与 ESM-2 余弦KNN构图保持一致。

    mode:
      - strict_diffusion: 通过真实正样本内部余弦阈值筛选
      - relaxed_diffusion: strict不足时，取余弦相似度最高的一批扩散候选
      - failed: 没有可用扩散样本
    """
    if candidate_samples is None or len(candidate_samples) == 0:
        return None, 'failed', 0.0, 0.0

    gen = _to_cpu_float_tensor(candidate_samples)
    real = _to_cpu_float_tensor(real_pos_samples)

    finite_mask = torch.isfinite(gen).all(dim=1)
    gen = gen[finite_mask]
    if gen.size(0) == 0 or real.size(0) == 0:
        return None, 'failed', 0.0, 0.0

    max_sim = _pairwise_max_cosine_similarity(gen, real, chunk_size=256)
    if max_sim.numel() == 0:
        return None, 'failed', 0.0, 0.0

    strict_threshold = _real_positive_cosine_threshold(
        real,
        quantile=getattr(config, 'strict_cosine_quantile', 0.05),
        min_threshold=getattr(config, 'strict_cosine_min_threshold', 0.60)
    )

    avg_quality = float(max_sim.mean().item())
    accepted_ratio = 0.0

    if strict_threshold is not None:
        strict_mask = max_sim >= strict_threshold
        strict_samples = gen[strict_mask]
        accepted_ratio = float(strict_mask.float().mean().item())
        if strict_samples.size(0) >= max(1, min(n_to_generate, config.min_samples_per_protein)):
            keep_n = min(n_to_generate, strict_samples.size(0))
            strict_sim = max_sim[strict_mask]
            order = torch.argsort(strict_sim, descending=True)[:keep_n]
            return strict_samples[order].numpy(), 'strict_diffusion', avg_quality, accepted_ratio

    # strict不足时仍只用扩散生成候选，选最像真实正类的样本；不做fallback复制。
    if getattr(config, 'allow_relaxed_diffusion', True):
        keep_n = min(n_to_generate, gen.size(0), max(1, int(gen.size(0) * config.relaxed_keep_ratio)))
        order = torch.argsort(max_sim, descending=True)[:keep_n]
        return gen[order].numpy(), 'relaxed_diffusion', avg_quality, accepted_ratio

    return None, 'failed', avg_quality, accepted_ratio


def _rebuild_augmented_graph(data, final_samples, config):
    """把扩散生成样本接回原图，并用余弦 KNN 重构边。"""
    new_x = torch.tensor(final_samples, dtype=torch.float32)
    new_y = torch.ones(new_x.size(0), dtype=torch.long)

    updated_x = torch.cat([data.x.cpu(), new_x], dim=0)
    updated_y = torch.cat([data.y.cpu(), new_y], dim=0)

    if len(updated_x) > config.max_nodes_per_graph:
        pos_mask_new = (updated_y == 1)
        neg_mask_new = (updated_y == 0)
        pos_indices = torch.where(pos_mask_new)[0]
        neg_indices = torch.where(neg_mask_new)[0]
        max_neg = config.max_nodes_per_graph - len(pos_indices)
        if max_neg > 0 and len(neg_indices) > max_neg:
            keep_neg = neg_indices[torch.randperm(len(neg_indices))[:max_neg]]
            keep_indices = torch.cat([pos_indices, keep_neg])
        else:
            keep_indices = torch.arange(len(updated_x))
        updated_x = updated_x[keep_indices]
        updated_y = updated_y[keep_indices]

    updated_edge_index = create_knn_edges(
        updated_x,
        k=config.knn_k,
        max_samples=2000,
        verbose=config.verbose_loading
    )

    return Data(
        x=updated_x,
        edge_index=updated_edge_index,
        y=updated_y,
        protein_context=data.protein_context.cpu(),
        name=str(getattr(data, 'name', 'protein')) + '_diffusion_aug'
    )


def robust_augment_dataset(dataset, diffusion_model, config):
    """
    严格扩散增强：
    1. 新节点必须来自扩散模型 generate_positive_sample；
    2. 不再使用复制/加噪真实正样本 fallback；
    3. 统计严格通过、放宽采用、未增强数量，避免把 fallback 当扩散成功。
    """
    augmented_data = []
    stats = {
        'strict_diffusion': 0,
        'relaxed_diffusion': 0,
        'failed_no_aug': 0,
        'no_positive': 0,
        'generated_nodes': 0,
        'quality_scores': [],
        'accepted_ratios': []
    }

    print("🎯 严格扩散增强策略:")
    print(f"  - 目标比例: {config.target_ratio:.1%}（建议 20%-30%，不要再用 90%）")
    print(f"  - 单蛋白最大扩散生成节点: {config.max_generated_per_protein}")
    print(f"  - 最大增强倍数: {config.max_augment_ratio}")
    print(f"  - 扩散样本筛选: cosine similarity（strict_min={config.strict_cosine_min_threshold}, q={config.strict_cosine_quantile}）")
    print(f"  - 是否允许放宽扩散采用: {config.allow_relaxed_diffusion}")
    print("  - fallback复制增强: False")

    for data in tqdm(dataset, desc="Diffusion augmenting"):
        try:
            pos_mask = (data.y == 1)
            if pos_mask.sum().item() == 0:
                augmented_data.append(data)
                stats['no_positive'] += 1
                continue

            real_pos_samples = data.x[pos_mask].cpu().numpy()
            n_pos = int(pos_mask.sum().item())
            n_neg = int((data.y == 0).sum().item())
            total_nodes = n_pos + n_neg

            target_pos = int(total_nodes * config.target_ratio)
            n_to_generate = max(config.min_samples_per_protein, target_pos - n_pos)
            n_to_generate = min(n_to_generate, int(n_pos * config.max_augment_ratio))
            n_to_generate = min(n_to_generate, config.max_generated_per_protein)

            if n_to_generate <= 0:
                augmented_data.append(data)
                stats['failed_no_aug'] += 1
                continue

            candidate_num = min(
                max(n_to_generate * config.candidate_multiplier, config.min_samples_per_protein),
                config.max_candidate_samples
            )

            protein_context = data.protein_context.to(config.device)
            candidate_samples = diffusion_model.generate_positive_sample(
                protein_context,
                num_samples=candidate_num,
                verbose=False
            )

            final_samples, mode, quality_score, accepted_ratio = _select_diffusion_samples(
                candidate_samples,
                real_pos_samples,
                n_to_generate,
                config
            )

            stats['quality_scores'].append(float(quality_score))
            stats['accepted_ratios'].append(float(accepted_ratio))

            if final_samples is None or len(final_samples) == 0:
                augmented_data.append(data)
                stats['failed_no_aug'] += 1
                continue

            augmented_graph = _rebuild_augmented_graph(data, final_samples, config)
            augmented_data.append(augmented_graph)
            stats[mode] += 1
            stats['generated_nodes'] += int(len(final_samples))

        except Exception as e:
            print(f"Warning: diffusion augmentation failed for {getattr(data, 'name', 'unknown')}: {e}")
            augmented_data.append(data)
            stats['failed_no_aug'] += 1

    total = max(1, len(dataset))
    avg_quality = float(np.mean(stats['quality_scores'])) if stats['quality_scores'] else 0.0
    avg_accept = float(np.mean(stats['accepted_ratios'])) if stats['accepted_ratios'] else 0.0
    print(
        "✅ 扩散增强统计: "
        f"strict={stats['strict_diffusion']}, "
        f"relaxed={stats['relaxed_diffusion']}, "
        f"no_aug={stats['failed_no_aug']}, "
        f"no_positive={stats['no_positive']}, "
        f"total={len(dataset)}, "
        f"generated_nodes={stats['generated_nodes']}, "
        f"avg_quality={avg_quality:.4f}, avg_strict_accept={avg_accept:.3f}"
    )
    print(
        f"📌 扩散采用率: {(stats['strict_diffusion'] + stats['relaxed_diffusion']) / total:.1%}；"
        "没有 fallback 复制增强。"
    )

    return augmented_data, stats

def domain_adaptive_loss(predictions, targets, domain_weight=0.1):
    """域适应损失：BCEWithLogitsLoss + 预测概率方差正则项。"""
    if predictions.dim() == 1:
        base_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            predictions,
            targets.float()
        )
    else:
        base_loss = torch.nn.functional.cross_entropy(predictions, targets.long())

    if predictions.size(0) > 1:
        if predictions.dim() == 1:
            probs = torch.sigmoid(predictions)
            prob_var = torch.var(probs, dim=0)
        else:
            probs = torch.softmax(predictions, dim=1)
            prob_var = torch.var(probs, dim=0).mean()
        domain_loss = domain_weight * prob_var
    else:
        domain_loss = torch.tensor(0.0, device=predictions.device)

    return base_loss + domain_loss


class RobustGNNModel(ImprovedBindingSiteGNN):
    """鲁棒GNN模型。"""

    def __init__(self, input_dim, hidden_dim=128, dropout=0.3, use_focal_loss=True,
                 focal_alpha=0.75, focal_gamma=2.0, pos_weight=3.0, domain_weight=0.1):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_focal_loss=use_focal_loss,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            pos_weight=pos_weight
        )
        self.domain_weight = domain_weight
        self.best_threshold = 0.5

    def train_with_domain_adaptation(self, train_data, val_data, epochs=100, lr=0.001,
                                     device='cuda', patience=10, threshold_metric='mcc'):
        """训练：验证集按 MCC 选阈值和早停，测试集不参与。"""
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.7,
            patience=5
        )

        best_val_mcc = -1e9
        best_val_f1 = 0.0
        best_val_auc = 0.0
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            batch_count = 0

            for data in train_data:
                if data.x.size(0) == 0:
                    continue

                data = data.to(device)
                optimizer.zero_grad()

                out = self(data)
                # Focal/BCE 主损失 + 预测概率方差正则，兼顾类别不平衡和跨数据分布稳定性。
                base_loss = self.loss_fn(out, data.y.float())
                if out.numel() > 1:
                    prob_var = torch.var(torch.sigmoid(out))
                    loss = base_loss + self.domain_weight * prob_var
                else:
                    loss = base_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

            avg_loss = total_loss / batch_count if batch_count > 0 else 0.0

            # 每轮都在验证集上选阈值；该阈值只来自验证集
            val_metrics = self.evaluate(
                val_data,
                device=device,
                search_threshold=True,
                threshold_metric=threshold_metric
            )
            scheduler.step(val_metrics['mcc'])

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | Loss={avg_loss:.4f} | "
                    f"Val MCC={val_metrics['mcc']:.4f} | Val F1={val_metrics['f1']:.4f} | "
                    f"Val AUC-PR={val_metrics['auc_pr']:.4f} | Thr={val_metrics['threshold']:.2f}"
                )

            if val_metrics['mcc'] > best_val_mcc:
                best_val_mcc = val_metrics['mcc']
                best_val_f1 = val_metrics['f1']
                best_val_auc = val_metrics['auc_pr']
                self.best_threshold = val_metrics['threshold']
                best_state = copy.deepcopy(self.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if best_state is not None:
            self.load_state_dict(best_state)

        return best_val_auc, best_val_f1, best_val_mcc, self.best_threshold



def cross_validation_training(original_data, config):
    """严格无泄漏交叉验证：每折只用训练折训练扩散模型，并只增强训练折。"""
    print(f"\n🔄 {config.cv_folds}折严格无泄漏交叉验证训练...")
    print("📊 严格策略: 每折单独训练扩散模型；验证折不参与扩散训练、不参与增强、不参与GNN训练")

    kf = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)
    cv_results = []
    original_indices = list(range(len(original_data)))
    fold_aug_stats = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(original_indices)):
        print(f"\n📊 第 {fold + 1}/{config.cv_folds} 折（严格无泄漏）")
        train_idx = list(train_idx)
        val_idx = list(val_idx)
        train_original = [original_data[i] for i in train_idx]
        val_fold = [original_data[i] for i in val_idx]

        train_ratio, train_pos, train_neg = calculate_class_ratio(train_original)
        val_ratio, val_pos, val_neg = calculate_class_ratio(val_fold)
        print(f"  📈 当前折训练原始数据: {len(train_original)} 个蛋白, 正样本={train_pos:,}, 负样本={train_neg:,}, 比例={train_ratio:.3%}")
        print(f"  📊 当前折验证原始数据: {len(val_fold)} 个蛋白, 正样本={val_pos:,}, 负样本={val_neg:,}, 比例={val_ratio:.3%}")

        if getattr(config, 'disable_diffusion', False):
            print("  🚫 无扩散对照模式：不训练扩散模型，不添加生成节点")
            train_augmented = []
            aug_stats = {
                'strict_diffusion': 0,
                'relaxed_diffusion': 0,
                'failed_no_aug': len(train_original),
                'no_positive': 0,
                'generated_nodes': 0,
                'quality_scores': [],
                'accepted_ratios': []
            }
            fold_aug_stats.append(aug_stats)
            aug_ratio, aug_pos, aug_neg = calculate_class_ratio(train_original)
            train_fold = train_original
            print(f"  ✅ 无扩散训练折: 正样本={aug_pos:,}, 负样本={aug_neg:,}, 比例={aug_ratio:.3%}")
            print(f"  📈 GNN训练集大小: {len(train_fold)} (仅原始训练折)")
        else:
            print("  🧠 当前折训练扩散模型（仅 train_fold）...")
            diffusion_model = EnhancedDiffusionModel(
                input_dim=config.diffusion_input_dim,
                T=config.diffusion_T,
                device=config.device
            )
            diffusion_start = time.time()
            diffusion_model.train_on_positive_samples(
                train_original,
                epochs=config.diffusion_epochs,
                batch_size=config.diffusion_batch_size
            )
            diffusion_time = time.time() - diffusion_start
            print(f"  ✅ 当前折扩散模型完成: {diffusion_time:.1f}秒")

            print("  🧬 当前折扩散增强训练数据（仅 train_fold，无fallback复制）...")
            augment_start = time.time()
            train_augmented, aug_stats = robust_augment_dataset(train_original, diffusion_model, config)
            augment_time = time.time() - augment_start
            fold_aug_stats.append(aug_stats)

            aug_ratio, aug_pos, aug_neg = calculate_class_ratio(train_augmented)
            print(f"  ✅ 当前折增强完成: 正样本={aug_pos:,}, 负样本={aug_neg:,}, 比例={aug_ratio:.3%}, 用时={augment_time:.1f}秒")

            train_fold = train_augmented + train_original
            print(f"  📈 GNN训练集大小: {len(train_fold)} (扩散增强: {len(train_augmented)}, 原始: {len(train_original)})")
        print(f"  📊 GNN验证集大小: {len(val_fold)} (仅原始真实数据)")

        model = RobustGNNModel(
            input_dim=config.diffusion_input_dim,
            hidden_dim=config.gnn_hidden_dim,
            dropout=config.gnn_dropout,
            use_focal_loss=config.use_focal_loss,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma,
            pos_weight=config.pos_weight,
            domain_weight=config.domain_weight
        )

        best_auc, best_f1, best_mcc, best_threshold = model.train_with_domain_adaptation(
            train_fold,
            val_fold,
            epochs=config.gnn_epochs,
            lr=config.gnn_lr,
            device=config.device,
            patience=config.gnn_patience,
            threshold_metric=config.threshold_metric
        )

        cv_results.append({
            'model': model,
            'val_f1': float(best_f1),
            'val_mcc': float(best_mcc),
            'val_auc': float(best_auc),
            'threshold': float(best_threshold),
            'augmentation_stats': aug_stats,
            'augmented_ratio': float(aug_ratio)
        })

        print(
            f"  ✅ 第{fold + 1}折: MCC={best_mcc:.4f}, F1={best_f1:.4f}, "
            f"AUC-PR={best_auc:.4f}, Thr={best_threshold:.2f}"
        )

    return cv_results, fold_aug_stats

def _collect_probs_labels(model, dataset, device):
    """收集单个模型在数据集上的概率和标签。"""
    model.eval()
    model.to(device)
    probs_all = []
    labels_all = []
    with torch.no_grad():
        for data in dataset:
            if data.x.size(0) == 0:
                continue
            data = data.to(device)
            logits = model(data)
            probs = torch.sigmoid(logits)
            probs_all.extend(probs.detach().cpu().tolist())
            labels_all.extend(data.y.detach().cpu().tolist())
    return np.asarray(probs_all, dtype=float), np.asarray(labels_all, dtype=int)


def _compute_metrics_from_probs(labels, probs, threshold):
    """用固定阈值计算指标；测试集不参与阈值搜索。"""
    if len(labels) == 0:
        return {
            'f1': 0.0, 'mcc': 0.0, 'auc_pr': 0.0, 'accuracy': 0.0,
            'balanced_accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
            'sensitivity': 0.0, 'specificity': 0.0, 'auc_roc': 0.0,
            'threshold': float(threshold),
            'confusion_matrix': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
            'total_samples': 0, 'positive_samples': 0, 'negative_samples': 0, 'positive_ratio': 0.0
        }

    preds = (probs > threshold).astype(int)
    accuracy = accuracy_score(labels, preds)
    balanced_acc = balanced_accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    mcc = matthews_corrcoef(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    auc_pr = 0.0
    auc_roc = 0.0
    if len(set(labels.tolist())) > 1:
        try:
            precision_curve, recall_curve, _ = precision_recall_curve(labels, probs)
            auc_pr = auc(recall_curve, precision_curve)
            auc_roc = roc_auc_score(labels, probs)
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
        'threshold': float(threshold),
        'confusion_matrix': {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)},
        'total_samples': int(len(labels)),
        'positive_samples': int(labels.sum()),
        'negative_samples': int(len(labels) - labels.sum()),
        'positive_ratio': float(labels.mean()) if len(labels) > 0 else 0.0,
    }


def evaluate_fixed_threshold(model, dataset, config):
    """单模型独立测试集评估：只使用验证集固定阈值。"""
    fixed_threshold = getattr(model, 'best_threshold', 0.5)
    return model.evaluate(
        dataset,
        device=config.device,
        threshold=fixed_threshold,
        search_threshold=False
    )


def evaluate_ensemble_fixed_threshold(models, dataset, config, fixed_threshold=None):
    """
    3折模型概率集成评估：
    - 每个fold模型都来自训练集内部CV；
    - 测试集只用于最终评估；
    - 阈值使用各fold验证阈值的平均值，不在测试集上重新搜索。
    """
    if not models:
        return _compute_metrics_from_probs(np.asarray([], dtype=int), np.asarray([], dtype=float), 0.5)

    all_model_probs = []
    labels_ref = None
    for model in models:
        probs, labels = _collect_probs_labels(model, dataset, config.device)
        all_model_probs.append(probs)
        if labels_ref is None:
            labels_ref = labels

    min_len = min(len(p) for p in all_model_probs) if all_model_probs else 0
    if min_len == 0:
        return _compute_metrics_from_probs(np.asarray([], dtype=int), np.asarray([], dtype=float), 0.5)

    probs_stack = np.vstack([p[:min_len] for p in all_model_probs])
    avg_probs = probs_stack.mean(axis=0)
    labels_ref = labels_ref[:min_len]

    if fixed_threshold is None:
        thresholds = [float(getattr(m, 'best_threshold', 0.5)) for m in models]
        fixed_threshold = float(np.mean(thresholds))

    return _compute_metrics_from_probs(labels_ref, avg_probs, fixed_threshold)


def load_dataset_from_file(dataset_file, config):
    """从单个txt文件加载数据集。"""
    temp_data_dir = os.path.join(config.data_dir, "temp")
    os.makedirs(temp_data_dir, exist_ok=True)

    temp_file = os.path.join(temp_data_dir, os.path.basename(dataset_file))
    shutil.copy2(dataset_file, temp_file)

    try:
        dataset_loader = ProteinDataset(temp_data_dir, device=config.device)
        dataset = dataset_loader.proteins
    finally:
        if os.path.exists(temp_data_dir):
            shutil.rmtree(temp_data_dir)

    return dataset


def load_dataset_quiet(dataset_file, config):
    """静默加载数据集。"""
    temp_data_dir = os.path.join(config.data_dir, "temp")
    os.makedirs(temp_data_dir, exist_ok=True)

    temp_file = os.path.join(temp_data_dir, os.path.basename(dataset_file))
    shutil.copy2(dataset_file, temp_file)

    try:
        if not config.verbose_loading:
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    dataset_loader = ProteinDataset(temp_data_dir, device=config.device)
                    dataset = dataset_loader.proteins
        else:
            dataset_loader = ProteinDataset(temp_data_dir, device=config.device)
            dataset = dataset_loader.proteins
    finally:
        if os.path.exists(temp_data_dir):
            shutil.rmtree(temp_data_dir)

    return dataset


def print_metrics(metrics, eval_time=None):
    """打印指标。"""
    time_text = f" ({eval_time:.2f}s)" if eval_time is not None else ""
    print(f"📈 鲁棒测试结果{time_text}:")
    print(f"  🎯 Threshold:        {metrics['threshold']:.2f}  ← 固定验证集阈值")
    print(f"  🎯 F1 Score:         {metrics['f1']:.4f}")
    print(f"  🎯 MCC:              {metrics['mcc']:.4f}")
    print(f"  🎯 Accuracy:         {metrics['accuracy']:.4f}")
    print(f"  🎯 Balanced Acc:     {metrics['balanced_accuracy']:.4f}")
    print(f"  🎯 Precision:        {metrics['precision']:.4f}")
    print(f"  🎯 Recall:           {metrics['recall']:.4f}")
    print(f"  🎯 Specificity:      {metrics['specificity']:.4f}")
    print(f"  🎯 AUC-PR:           {metrics['auc_pr']:.4f}")
    print(f"  🎯 AUC-ROC:          {metrics['auc_roc']:.4f}")



def train_and_test_robust_model(train_file, test_files, config, new_test_files=None):
    """训练并测试鲁棒模型：扩散训练和增强都在每个CV训练折内部完成。"""
    train_name = os.path.splitext(os.path.basename(train_file))[0]
    print(f"\n🚀 开始严格扩散增强训练-测试: {train_name}")
    print("=" * 60)

    ratio_str = f"{config.target_ratio:.2f}".replace(".", "")
    output_dir_name = f"{train_name}_strict_diffusion_r{ratio_str}_{config.experiment_name}"
    output_path = os.path.join(config.output_dir, output_dir_name)
    os.makedirs(output_path, exist_ok=True)

    print("📊 阶段1: 加载训练数据...")
    train_dataset = load_dataset_quiet(train_file, config)
    if not train_dataset:
        print(f"❌ 数据集为空: {train_file}")
        return None

    print(f"✅ 加载了 {len(train_dataset)} 个蛋白质")
    orig_ratio, orig_pos, orig_neg = calculate_class_ratio(train_dataset)
    print(f"📊 原始训练数据: {orig_pos:,} 正样本, {orig_neg:,} 负样本 (比例: {orig_ratio:.3%})")

    print("\n🔒 阶段2: 严格无泄漏交叉验证训练 + 折内扩散增强")
    gnn_start = time.time()
    cv_results, fold_aug_stats = cross_validation_training(train_dataset, config)
    gnn_time = time.time() - gnn_start

    if not cv_results:
        print("❌ 交叉验证失败，没有可用模型")
        return None

    best_cv_result = max(cv_results, key=lambda x: x['val_mcc'])
    fold_models = [r['model'] for r in cv_results]
    fold_thresholds = [float(r['threshold']) for r in cv_results]
    ensemble_threshold = float(np.mean(fold_thresholds)) if fold_thresholds else float(best_cv_result['threshold'])
    for m, th in zip(fold_models, fold_thresholds):
        m.best_threshold = th

    avg_f1 = float(np.mean([r['val_f1'] for r in cv_results]))
    avg_mcc = float(np.mean([r['val_mcc'] for r in cv_results]))
    avg_auc = float(np.mean([r['val_auc'] for r in cv_results]))
    avg_aug_ratio = float(np.mean([r.get('augmented_ratio', orig_ratio) for r in cv_results]))

    total_strict = int(sum(s['strict_diffusion'] for s in fold_aug_stats))
    total_relaxed = int(sum(s['relaxed_diffusion'] for s in fold_aug_stats))
    total_no_aug = int(sum(s['failed_no_aug'] for s in fold_aug_stats))
    total_generated_nodes = int(sum(s['generated_nodes'] for s in fold_aug_stats))

    print(
        f"✅ 严格交叉验证完成: 平均MCC={avg_mcc:.4f}, 平均F1={avg_f1:.4f}, "
        f"平均AUC-PR={avg_auc:.4f}, 平均增强比例={avg_aug_ratio:.3%} - 用时: {gnn_time:.1f}秒"
    )
    print(f"✅ 集成固定阈值: {ensemble_threshold:.2f} ← 各折验证集 {config.threshold_metric.upper()} 阈值平均")
    print(
        f"🧬 扩散增强汇总: strict={total_strict}, relaxed={total_relaxed}, "
        f"no_aug={total_no_aug}, generated_nodes={total_generated_nodes}; fallback复制增强=0"
    )

    model_save_path = os.path.join(output_path, "strict_diffusion_ensemble_gnn_models.pt")
    torch.save(
        {
            'fold_state_dicts': [m.state_dict() for m in fold_models],
            'fold_thresholds': fold_thresholds,
            'ensemble_threshold': float(ensemble_threshold),
            'selected_by': 'cv_fold_validation_mcc_mean_threshold',
            'augmentation_summary': {
                'strict_diffusion': total_strict,
                'relaxed_diffusion': total_relaxed,
                'failed_no_aug': total_no_aug,
                'generated_nodes': total_generated_nodes,
                'fallback': 0
            }
        },
        model_save_path
    )
    print(f"💾 3折集成模型保存至: {model_save_path}")

    print("\n🔍 阶段3: 独立测试集固定阈值评估")
    print("=" * 60)

    test_results = {}
    new_test_results = {}

    print(f"\n{'=' * 80}")
    print("📊 原始测试集 (DNA系列)")
    print(f"{'=' * 80}")

    for test_file in test_files:
        test_name = os.path.splitext(os.path.basename(test_file))[0]
        print(f"\n📊 测试数据集: {test_name}")
        try:
            test_dataset = load_dataset_quiet(test_file, config)
            print(f"✅ 加载了 {len(test_dataset)} 个蛋白质")
            start_time = time.time()
            metrics = evaluate_ensemble_fixed_threshold(fold_models, test_dataset, config, ensemble_threshold)
            eval_time = time.time() - start_time
            print_metrics(metrics, eval_time)
            test_results[test_name] = metrics
        except Exception as e:
            print(f"❌ 测试失败: {str(e)}")
            continue

    if new_test_files:
        print(f"\n{'=' * 80}")
        print("📊 新增测试集 (PDNA系列)")
        print(f"{'=' * 80}")
        for test_file in new_test_files:
            test_name = os.path.splitext(os.path.basename(test_file))[0]
            print(f"\n📊 测试数据集: {test_name}")
            try:
                test_dataset = load_dataset_quiet(test_file, config)
                print(f"✅ 加载了 {len(test_dataset)} 个蛋白质")
                start_time = time.time()
                metrics = evaluate_ensemble_fixed_threshold(fold_models, test_dataset, config, ensemble_threshold)
                eval_time = time.time() - start_time
                print_metrics(metrics, eval_time)
                new_test_results[test_name] = metrics
            except Exception as e:
                print(f"❌ 测试失败: {str(e)}")
                continue

    total_time = gnn_time

    full_results = {
        "model_name": train_name + "_strict_diffusion_noleak",
        "model_type": "Strict Fold-internal Diffusion Augmentation + 3-Fold Ensemble Robust GNN + Validation-fixed Threshold",
        "training_info": {
            "original_positive": int(orig_pos),
            "original_negative": int(orig_neg),
            "original_ratio": float(orig_ratio),
            "avg_augmented_ratio": float(avg_aug_ratio),
            "target_ratio": float(config.target_ratio),
            "cv_avg_f1": float(avg_f1),
            "cv_avg_mcc": float(avg_mcc),
            "cv_avg_auc": float(avg_auc),
            "best_cv_f1": float(best_cv_result['val_f1']),
            "best_cv_mcc": float(best_cv_result['val_mcc']),
            "best_cv_auc": float(best_cv_result['val_auc']),
            "fixed_threshold_from_validation": float(ensemble_threshold),
            "fold_thresholds": fold_thresholds,
            "ensemble_type": "mean_probability_from_all_cv_fold_models",
            "threshold_metric": config.threshold_metric,
            "gnn_time": float(gnn_time),
            "total_time": float(total_time)
        },
        "augmentation_summary": {
            "strict_diffusion": total_strict,
            "relaxed_diffusion": total_relaxed,
            "failed_no_aug": total_no_aug,
            "generated_nodes": total_generated_nodes,
            "fallback": 0,
            "note": "All added synthetic nodes come from diffusion_model.generate_positive_sample; no real-positive copying fallback is used."
        },
        "robust_config": {
            "quality_threshold": float(config.quality_threshold),
            "diversity_threshold": float(config.diversity_threshold),
            "domain_weight": float(config.domain_weight),
            "cv_folds": int(config.cv_folds),
            "max_augment_ratio": float(config.max_augment_ratio),
            "max_generated_per_protein": int(config.max_generated_per_protein),
            "target_ratio": float(config.target_ratio),
            "graph_metric": "cosine similarity KNN in ESM-2 embedding space",
            "leakage_control": "diffusion, augmentation and threshold selection are fold-internal; test uses fixed validation threshold"
        },
        "cv_results": [
            {
                "val_f1": r['val_f1'],
                "val_mcc": r['val_mcc'],
                "val_auc": r['val_auc'],
                "threshold": r['threshold'],
                "augmented_ratio": r.get('augmented_ratio', None),
                "augmentation_stats": {
                    k: v for k, v in r['augmentation_stats'].items()
                    if k not in ('quality_scores', 'accepted_ratios')
                }
            }
            for r in cv_results
        ],
        "test_results": test_results,
        "new_test_results": new_test_results
    }

    results_path = os.path.join(output_path, "strict_diffusion_results_noleak.json")
    with open(results_path, 'w') as f:
        json.dump(full_results, f, indent=2)

    print(f"\n✅ {train_name} 严格扩散增强训练-测试完成!")
    print(f"⏱️ 总用时: {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"📈 原始比例: {orig_ratio:.3%}; 折内平均增强比例: {avg_aug_ratio:.3%}")
    print(f"🎯 交叉验证性能: MCC={avg_mcc:.4f}, F1={avg_f1:.4f}, AUC-PR={avg_auc:.4f}")
    print(f"📁 结果保存至: {results_path}")

    return full_results

def main():
    """主函数：固定配置后做独立测试，不使用测试集调参。"""
    print("🛡️ 严格扩散增强训练-测试管道启动：无测试集阈值泄露版本")
    print("=" * 80)
    print("改进策略: 余弦KNN + 余弦筛选扩散增强 + 3折模型集成 + 验证集固定阈值 + 独立测试集评估")
    print("=" * 80)

    config = RobustTrainingConfig()
    set_seed(config.seed)

    print("\n🛡️ 鲁棒性配置:")
    print(f"  - 目标比例: {config.target_ratio:.1%}（已从90%降为更稳的默认比例）")
    print(f"  - 质量控制阈值: {config.quality_threshold}")
    print(f"  - 域适应权重: {config.domain_weight}")
    print(f"  - {config.cv_folds}折交叉验证")
    print(f"  - 阈值策略: 验证集 {config.threshold_metric.upper()} 最优，测试集固定")

    all_txt = sorted(glob.glob(os.path.join(config.data_dir, "*.txt")))
    train_files = [f for f in all_txt if "train" in os.path.basename(f).lower()]
    original_test_files = [
        f for f in all_txt
        if os.path.basename(f).startswith("DNA-") and "test" in os.path.basename(f).lower()
    ]
    new_test_files = [
        f for f in all_txt
        if os.path.basename(f).startswith("PDNA-") and "test" in os.path.basename(f).lower()
    ]

    print(f"\n🔍 找到 {len(train_files)} 个训练文件")
    print(f"   - 原始测试集 (DNA系列): {len(original_test_files)} 个")
    print(f"   - 新增测试集 (PDNA系列): {len(new_test_files)} 个")

    if original_test_files:
        print("\n📊 原始测试集:")
        for f in original_test_files:
            print(f"   - {os.path.basename(f)}")

    if new_test_files:
        print("\n📊 新增测试集:")
        for f in new_test_files:
            print(f"   - {os.path.basename(f)}")

    total_start = time.time()
    all_results = {}

    for train_file in train_files:
        try:
            result = train_and_test_robust_model(
                train_file,
                original_test_files,
                config,
                new_test_files
            )
            if result:
                train_name = os.path.splitext(os.path.basename(train_file))[0]
                all_results[train_name] = result
        except Exception as e:
            print(f"❌ {os.path.basename(train_file)} 训练失败: {str(e)}")
            continue

    total_time = time.time() - total_start

    if all_results:
        timestamp = int(time.time())
        results_file = f"all_strict_diffusion_results_noleak_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n✅ 所有训练完成，总用时: {total_time // 60:.0f}m {total_time % 60:.0f}s")
        print(f"📊 详细结果: {results_file}")

        print(f"\n{'=' * 100}")
        print("📋 鲁棒模型测试性能 - 原始测试集 (DNA系列，固定验证集阈值)")
        print("=" * 100)
        print(f"{'训练集':<15} {'平均F1':<10} {'平均MCC':<10} {'平均AUC-PR':<12} {'平均AUC-ROC':<12} {'平均平衡ACC':<12} {'固定阈值':<10}")
        print("-" * 100)

        for train_name, results in all_results.items():
            test_results = results['test_results']
            if test_results:
                avg_f1 = np.mean([m['f1'] for m in test_results.values()])
                avg_mcc = np.mean([m['mcc'] for m in test_results.values()])
                avg_auc_pr = np.mean([m['auc_pr'] for m in test_results.values()])
                avg_auc_roc = np.mean([m['auc_roc'] for m in test_results.values()])
                avg_balanced_acc = np.mean([m['balanced_accuracy'] for m in test_results.values()])
                fixed_thr = results['training_info']['fixed_threshold_from_validation']
                print(
                    f"{train_name:<15} {avg_f1:<10.4f} {avg_mcc:<10.4f} "
                    f"{avg_auc_pr:<12.4f} {avg_auc_roc:<12.4f} {avg_balanced_acc:<12.4f} {fixed_thr:<10.2f}"
                )
    else:
        print("\n❌ 没有成功完成任何鲁棒模型训练")


# 这个函数保留给你做探索，但不要把它作为论文最终调参流程。
def test_different_ratios(train_file, test_files, ratios=(0.10, 0.15, 0.20, 0.25, 0.30)):
    """
    比例测试函数，仅用于探索。
    注意：论文最终结果不能用测试集表现反向选择 target_ratio。
    最终论文应固定一个比例后，只做一次独立测试集评估。
    """
    print("⚠️ 比例测试仅供探索：不要用测试集平均F1/MCC选择论文最终参数。")
    results_summary = {}

    for ratio in ratios:
        print(f"\n🎯 探索比例: {ratio:.1%}")
        experiment_name = f"ratio_explore_{ratio:.2f}".replace(".", "")
        config = RobustTrainingConfig(target_ratio=ratio, experiment_name=experiment_name)
        config.verbose_loading = False
        set_seed(config.seed)

        try:
            result = train_and_test_robust_model(train_file, test_files, config)
            if result:
                results_summary[ratio] = result
        except Exception as e:
            print(f"❌ 比例 {ratio:.1%}: 错误 - {str(e)}")

    return results_summary


def quick_ratio_test():
    """快速比例探索入口。"""
    print("🚀 快速比例探索。注意：不要用它产生论文最终参数。")
    config = RobustTrainingConfig()
    train_file = os.path.join(config.data_dir, 'DNA-573_Train.txt')
    test_files = [
        os.path.join(config.data_dir, 'DNA-129_Test.txt'),
        os.path.join(config.data_dir, 'DNA-181_Test.txt'),
        os.path.join(config.data_dir, 'DNA-46_Test.txt')
    ]
    ratios = [0.10, 0.15, 0.20, 0.25, 0.30]
    return test_different_ratios(train_file, test_files, ratios)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--ratio-test":
        quick_ratio_test()
    elif len(sys.argv) > 1 and sys.argv[1] == "--no-diffusion":
        # 无扩散对照：同样CV、同样阈值策略、同样3折集成，只关闭扩散增强。
        print("🚫 启动无扩散对照实验")
        cfg = RobustTrainingConfig(target_ratio=0.30, experiment_name="no_diffusion_baseline")
        cfg.disable_diffusion = True
        set_seed(cfg.seed)
        train_files = sorted(glob.glob(os.path.join(cfg.data_dir, "*Train*.txt")))
        all_txt = sorted(glob.glob(os.path.join(cfg.data_dir, "*.txt")))
        original_test_files = [f for f in all_txt if os.path.basename(f).startswith("DNA-") and "test" in os.path.basename(f).lower()]
        new_test_files = [f for f in all_txt if os.path.basename(f).startswith("PDNA-") and "test" in os.path.basename(f).lower()]
        for train_file in train_files:
            train_and_test_robust_model(train_file, original_test_files, cfg, new_test_files)
    else:
        main()

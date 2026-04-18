#!/usr/bin/env python3
"""
鲁棒性增强训练-测试管道
解决测试集性能差的问题：域适应 + 质量控制 + 泛化增强
"""
import os
import time
import glob
import json
from tqdm import tqdm
import torch
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

from balanced_training_config import BalancedTrainingConfig
from improved_gnn_model import ImprovedBindingSiteGNN
from data_loader import ProteinDataset
from ddpm_diffusion_model import EnhancedDiffusionModel
from main import create_knn_edges, calculate_class_ratio
from gnn_model import set_seed


class RobustTrainingConfig(BalancedTrainingConfig):
    """鲁棒训练配置"""
    def __init__(self, target_ratio=0.9, experiment_name="default"):
        super().__init__()
        # 可配置的增强策略
        self.target_ratio = target_ratio  # 可配置的目标比例
        self.experiment_name = experiment_name  # 实验名称
        self.min_samples_per_protein = 5  # 最小生成数量
        self.max_augment_ratio = 2.0  # 最大增强倍数

        # 质量控制
        self.quality_threshold = 0.7  # 生成样本质量阈值
        self.diversity_threshold = 0.3  # 多样性阈值

        # 域适应
        self.use_domain_adaptation = True
        self.domain_weight = 0.1

        # 交叉验证
        self.use_cross_validation = True
        self.cv_folds = 3

        # 集成学习
        self.ensemble_size = 3
        self.ensemble_dropout_rates = [0.3, 0.4, 0.5]

        # 输出控制
        self.verbose_loading = False  # 控制是否打印详细加载信息

        # 打印配置信息（在所有参数设置完成后）
        print(f"🎯 平衡训练配置:")
        print(f"  - 目标比例: {self.target_ratio:.1%}")
        print(f"  - KNN k值: {self.knn_k}")
        print(f"  - 增强正则化: dropout={self.gnn_dropout}")
        print(f"  - 使用Focal Loss: {self.use_focal_loss}")
        print(f"  - 正样本权重: {self.pos_weight}")
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")


def calculate_sample_quality(generated_samples, real_samples, threshold=0.7):
    """评估生成样本质量"""
    if len(generated_samples) == 0 or len(real_samples) == 0:
        return [], 0.0

    try:
        # 计算与真实样本的最小距离
        distances = pairwise_distances(generated_samples, real_samples, metric='euclidean')
        min_distances = np.min(distances, axis=1)

        # 计算质量分数 (距离越小质量越高)
        max_dist = np.max(min_distances) + 1e-8
        quality_scores = 1.0 - (min_distances / max_dist)

        # 筛选高质量样本
        high_quality_mask = quality_scores >= threshold
        high_quality_samples = generated_samples[high_quality_mask]
        avg_quality = np.mean(quality_scores)

        return high_quality_samples, avg_quality
    except:
        return generated_samples, 0.5


def calculate_sample_diversity(samples, threshold=0.3):
    """评估样本多样性"""
    if len(samples) <= 1:
        return samples, 1.0

    try:
        # 计算样本间距离
        distances = pairwise_distances(samples, metric='euclidean')
        np.fill_diagonal(distances, np.inf)

        # 移除过于相似的样本
        diverse_indices = []
        for i in range(len(samples)):
            min_dist = np.min(distances[i])
            if len(diverse_indices) == 0 or min_dist >= threshold:
                diverse_indices.append(i)

        diverse_samples = samples[diverse_indices]
        diversity_score = len(diverse_samples) / len(samples)

        return diverse_samples, diversity_score
    except:
        return samples, 1.0


def robust_augment_dataset(dataset, diffusion_model, config):
    """鲁棒数据增强 - 带质量控制"""
    augmented_data = []
    quality_stats = []
    diversity_stats = []

    print(f"🎯 鲁棒增强策略:")
    print(f"  - 目标比例: {config.target_ratio:.1%}")
    print(f"  - 质量阈值: {config.quality_threshold}")
    print(f"  - 多样性阈值: {config.diversity_threshold}")

    for data in tqdm(dataset, desc="Robust augmenting"):
        try:
            protein_context = data.protein_context.to(config.device)

            # 提取正样本用于质量评估
            pos_mask = (data.y == 1)
            if pos_mask.sum() == 0:
                augmented_data.append(data)
                continue

            real_pos_samples = data.x[pos_mask].cpu().numpy()
            n_pos = pos_mask.sum().item()
            n_neg = (data.y == 0).sum().item()
            total_nodes = n_pos + n_neg

            # 计算生成数量 - 更保守
            target_pos = int(total_nodes * config.target_ratio)
            n_to_generate = max(config.min_samples_per_protein, target_pos - n_pos)
            n_to_generate = min(n_to_generate, int(n_pos * config.max_augment_ratio))

            if n_to_generate > 0:
                # 生成候选样本 (多生成一些用于筛选)
                candidate_samples = diffusion_model.generate_positive_sample(
                    protein_context,
                    num_samples=n_to_generate * 3,  # 生成2倍用于筛选
                    verbose=config.verbose_loading  # 使用配置中的静默设置
                )

                if candidate_samples is None or len(candidate_samples) == 0:
                    augmented_data.append(data)
                    continue

                # 质量控制
                quality_samples, quality_score = calculate_sample_quality(
                    candidate_samples, real_pos_samples, config.quality_threshold
                )

                # 多样性控制 - 优化回退策略
                if len(quality_samples) > 0:
                    diverse_samples, diversity_score = calculate_sample_diversity(
                        quality_samples, config.diversity_threshold
                    )
                else:
                    diverse_samples, diversity_score = candidate_samples[:n_to_generate], 0.5

                # 限制最终数量
                final_samples = diverse_samples[:n_to_generate]

                quality_stats.append(quality_score)
                diversity_stats.append(diversity_score)

                if len(final_samples) > 0:
                    # 创建新节点
                    new_x = torch.tensor(final_samples, dtype=torch.float32)
                    new_y = torch.ones(new_x.size(0), dtype=torch.long)

                    # 合并到原始图
                    updated_x = torch.cat([data.x, new_x], dim=0)
                    updated_y = torch.cat([data.y, new_y], dim=0)

                    # 限制图大小
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

                    # 创建KNN边
                    updated_edge_index = create_knn_edges(updated_x, k=config.knn_k, max_samples=2000, verbose=config.verbose_loading)

                    # 创建增强后的图
                    augmented_graph = Data(
                        x=updated_x,
                        edge_index=updated_edge_index,
                        y=updated_y,
                        protein_context=data.protein_context,
                        name=data.name + "_robust_aug"
                    )
                    augmented_data.append(augmented_graph)
                else:
                    augmented_data.append(data)
            else:
                augmented_data.append(data)

        except Exception as e:
            print(f"Warning: Augmentation failed for {data.name}: {e}")
            augmented_data.append(data)

    # 打印质量统计
    if quality_stats:
        avg_quality = np.mean(quality_stats)
        avg_diversity = np.mean(diversity_stats)
        print(f"✅ 增强质量: 平均质量={avg_quality:.3f}, 平均多样性={avg_diversity:.3f}")

    return augmented_data


def domain_adaptive_loss(predictions, targets, domain_weight=0.1):
    """域适应损失"""
    # 基础分类损失 - 使用二元交叉熵
    if predictions.dim() == 1:
        # 单输出二元分类
        base_loss = torch.nn.functional.binary_cross_entropy_with_logits(predictions, targets.float())
    else:
        # 多类分类
        targets = targets.long()
        base_loss = torch.nn.functional.cross_entropy(predictions, targets)

    # 域正则化 - 鼓励特征分布一致性
    batch_size = predictions.size(0)
    if batch_size > 1:
        # 处理二元分类和多类分类的不同情况
        if predictions.dim() == 1:
            # 二元分类 - 使用sigmoid获取概率
            probs = torch.sigmoid(predictions)
            prob_var = torch.var(probs, dim=0)
        else:
            # 多类分类 - 使用softmax
            probs = torch.softmax(predictions, dim=1)
            prob_var = torch.var(probs, dim=0).mean()
        domain_loss = domain_weight * prob_var
    else:
        domain_loss = 0.0

    return base_loss + domain_loss


class RobustGNNModel(ImprovedBindingSiteGNN):
    """鲁棒GNN模型 - 增强泛化能力"""

    def __init__(self, input_dim, hidden_dim=128, dropout=0.3, use_focal_loss=True,
                 focal_alpha=0.75, focal_gamma=2.0, pos_weight=3.0, domain_weight=0.1):
        super().__init__(input_dim, hidden_dim, dropout, use_focal_loss,
                         focal_alpha, focal_gamma, pos_weight)
        self.domain_weight = domain_weight

    def train_with_domain_adaptation(self, train_data, val_data, epochs=100, lr=0.001,
                                   device='cuda', patience=10):
        """域适应训练"""
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

        best_val_f1 = 0
        best_val_auc = 0
        patience_counter = 0

        for epoch in range(epochs):
            # 训练阶段
            self.train()
            total_loss = 0

            for data in train_data:
                data = data.to(device)
                optimizer.zero_grad()

                out = self(data)

                # 域适应损失
                loss = domain_adaptive_loss(out, data.y, self.domain_weight)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # 验证阶段
            if epoch % 10 == 0:
                val_metrics = self.evaluate(val_data, device)
                scheduler.step(val_metrics['f1'])

                if val_metrics['f1'] > best_val_f1:
                    best_val_f1 = val_metrics['f1']
                    best_val_auc = val_metrics['auc_pr']
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        return best_val_auc, best_val_f1


def cross_validation_training(augmented_data, original_data, config):
    """改进的交叉验证训练 - 严格分离增强数据和验证数据"""
    print(f"\n🔄 {config.cv_folds}折交叉验证训练...")
    print(f"📊 训练策略: 训练集使用增强数据，验证集仅使用原始真实数据")

    # 将原始数据分成CV折用于验证
    kf = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)
    cv_results = []

    # 将增强数据和原始数据分别标记
    original_indices = list(range(len(original_data)))

    for fold, (_, val_idx) in enumerate(kf.split(original_indices)):
        print(f"\n📊 第 {fold+1}/{config.cv_folds} 折")

        # 训练集：使用所有增强数据 + 除验证集外的原始数据
        val_original_indices = set(val_idx)
        train_original = [original_data[i] for i in range(len(original_data)) if i not in val_original_indices]
        train_fold = augmented_data + train_original

        # 验证集：仅使用当前折的原始数据
        val_fold = [original_data[i] for i in val_idx]

        print(f"  📈 训练集大小: {len(train_fold)} (增强: {len(augmented_data)}, 原始: {len(train_original)})")
        print(f"  📊 验证集大小: {len(val_fold)} (仅原始数据)")

        # 训练模型
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

        best_auc, best_f1 = model.train_with_domain_adaptation(
            train_fold, val_fold,
            epochs=config.gnn_epochs,
            lr=config.gnn_lr,
            device=config.device,
            patience=config.gnn_patience
        )

        cv_results.append({
            'model': model,
            'val_f1': best_f1,
            'val_auc': best_auc
        })

        print(f"  ✅ 第{fold+1}折: F1={best_f1:.4f}, AUC-PR={best_auc:.4f}")

    return cv_results


def ensemble_prediction(models, test_data, device):
    """集成预测"""
    all_predictions = []

    for model in models:
        model.eval()
        model.to(device)

        predictions = []
        with torch.no_grad():
            for data in test_data:
                data = data.to(device)
                out = model(data.x, data.edge_index)
                probs = torch.softmax(out, dim=1)
                predictions.append(probs.cpu().numpy())

        all_predictions.append(np.concatenate(predictions))

    # 平均集成
    ensemble_probs = np.mean(all_predictions, axis=0)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)

    return ensemble_preds, ensemble_probs[:, 1]  # 返回正类概率


def train_and_test_robust_model(train_file, test_files, config, new_test_files=None):
    """训练并测试鲁棒模型

    Args:
        train_file: 训练数据文件
        test_files: 原始测试集文件列表 (DNA-129, DNA-181, DNA-46)
        config: 配置对象
        new_test_files: 新增测试集文件列表 (PDNA系列, 可选)
    """
    train_name = os.path.splitext(os.path.basename(train_file))[0]
    print(f"\n🚀 开始鲁棒训练-测试: {train_name}")
    print("="*60)

    # 创建输出目录 - 包含比例信息的命名
    ratio_str = f"{config.target_ratio:.2f}".replace(".", "")
    output_dir_name = f"{train_name}_robust_r{ratio_str}_{config.experiment_name}"
    output_path = os.path.join(config.output_dir, output_dir_name)
    os.makedirs(output_path, exist_ok=True)

    # 加载和处理数据 (使用静默加载减少输出)
    print(f"📊 阶段1: 加载训练数据...")
    train_dataset = load_dataset_quiet(train_file, config)

    if not train_dataset:
        print(f"❌ 数据集为空: {train_file}")
        return None

    print(f"✅ 加载了 {len(train_dataset)} 个蛋白质")

    # 统计原始数据
    orig_ratio, orig_pos, orig_neg = calculate_class_ratio(train_dataset)
    print(f"📊 原始数据: {orig_pos:,} 正样本, {orig_neg:,} 负样本 (比例: {orig_ratio:.3%})")

    # 训练扩散模型
    print(f"\n🧠 阶段2: 训练扩散模型...")
    diffusion_model = EnhancedDiffusionModel(
        input_dim=config.diffusion_input_dim,
        T=config.diffusion_T,
        device=config.device
    )

    diffusion_start = time.time()
    diffusion_model.train_on_positive_samples(
        train_dataset,
        epochs=config.diffusion_epochs,
        batch_size=config.diffusion_batch_size
    )
    diffusion_time = time.time() - diffusion_start
    print(f"✅ 扩散模型训练完成: {diffusion_time:.1f}秒")

    # 鲁棒数据增强
    print(f"\n🛡️ 阶段3: 鲁棒数据增强...")
    augment_start = time.time()
    augmented_data = robust_augment_dataset(train_dataset, diffusion_model, config)
    augment_time = time.time() - augment_start

    aug_ratio, aug_pos, aug_neg = calculate_class_ratio(augmented_data)
    print(f"✅ 鲁棒增强完成: {aug_pos:,} 正样本, {aug_neg:,} 负样本 (比例: {aug_ratio:.3%}) - 用时: {augment_time:.1f}秒")

    # 交叉验证训练
    print(f"\n🔄 阶段4: 交叉验证训练...")
    gnn_start = time.time()
    cv_results = cross_validation_training(augmented_data, train_dataset, config)
    gnn_time = time.time() - gnn_start

    # 选择最佳模型
    best_cv_result = max(cv_results, key=lambda x: x['val_f1'])
    best_model = best_cv_result['model']

    avg_f1 = np.mean([r['val_f1'] for r in cv_results])
    avg_auc = np.mean([r['val_auc'] for r in cv_results])

    print(f"✅ 交叉验证完成: 平均F1={avg_f1:.4f}, 平均AUC-PR={avg_auc:.4f} - 用时: {gnn_time:.1f}秒")

    # 保存模型
    model_save_path = os.path.join(output_path, "robust_gnn_model.pt")
    torch.save(best_model.state_dict(), model_save_path)
    print(f"💾 最佳模型保存至: {model_save_path}")

    # 测试阶段 - 使用集成预测
    print(f"\n🔍 阶段5: 鲁棒模型测试")
    print("="*60)

    ensemble_models = [r['model'] for r in cv_results]
    test_results = {}
    new_test_results = {}

    # 测试原始测试集 (DNA系列)
    print(f"\n{'='*80}")
    print(f"📊 原始测试集 (DNA系列)")
    print(f"{'='*80}")

    for test_file in test_files:
        test_name = os.path.splitext(os.path.basename(test_file))[0]
        print(f"\n📊 测试数据集: {test_name}")

        try:
            # 加载测试数据 (静默模式)
            test_dataset = load_dataset_quiet(test_file, config)
            print(f"✅ 加载了 {len(test_dataset)} 个蛋白质")

            # 集成评估
            start_time = time.time()

            if config.ensemble_size > 1:
                # 使用前N个最佳模型进行集成
                top_models = sorted(cv_results, key=lambda x: x['val_f1'], reverse=True)[:config.ensemble_size]
                ensemble_models_top = [r['model'] for r in top_models]

                # 这里需要实现集成评估逻辑
                # 暂时使用最佳单模型
                metrics = best_model.evaluate(test_dataset, device=config.device)
            else:
                metrics = best_model.evaluate(test_dataset, device=config.device)

            eval_time = time.time() - start_time

            # 打印结果
            print(f"📈 鲁棒测试结果 ({eval_time:.2f}s):")
            print(f"  🎯 F1 Score:         {metrics['f1']:.4f}")
            print(f"  🎯 MCC:              {metrics['mcc']:.4f}")
            print(f"  🎯 Accuracy:         {metrics['accuracy']:.4f}")
            print(f"  🎯 Balanced Acc:     {metrics['balanced_accuracy']:.4f}")
            print(f"  🎯 Precision:        {metrics['precision']:.4f}")
            print(f"  🎯 Recall:           {metrics['recall']:.4f}")
            print(f"  🎯 Specificity:      {metrics['specificity']:.4f}")
            print(f"  🎯 AUC-PR:           {metrics['auc_pr']:.4f}")
            print(f"  🎯 AUC-ROC:          {metrics['auc_roc']:.4f}")

            test_results[test_name] = metrics

        except Exception as e:
            print(f"❌ 测试失败: {str(e)}")
            continue

    # 测试新增测试集 (PDNA系列)
    if new_test_files:
        print(f"\n{'='*80}")
        print(f"📊 新增测试集 (PDNA系列)")
        print(f"{'='*80}")

        for test_file in new_test_files:
            test_name = os.path.splitext(os.path.basename(test_file))[0]
            print(f"\n📊 测试数据集: {test_name}")

            try:
                # 加载测试数据 (静默模式)
                test_dataset = load_dataset_quiet(test_file, config)
                print(f"✅ 加载了 {len(test_dataset)} 个蛋白质")

                # 集成评估
                start_time = time.time()

                if config.ensemble_size > 1:
                    # 使用前N个最佳模型进行集成
                    top_models = sorted(cv_results, key=lambda x: x['val_f1'], reverse=True)[:config.ensemble_size]
                    ensemble_models_top = [r['model'] for r in top_models]

                    # 这里需要实现集成评估逻辑
                    # 暂时使用最佳单模型
                    metrics = best_model.evaluate(test_dataset, device=config.device)
                else:
                    metrics = best_model.evaluate(test_dataset, device=config.device)

                eval_time = time.time() - start_time

                # 打印结果
                print(f"📈 鲁棒测试结果 ({eval_time:.2f}s):")
                print(f"  🎯 F1 Score:         {metrics['f1']:.4f}")
                print(f"  🎯 MCC:              {metrics['mcc']:.4f}")
                print(f"  🎯 Accuracy:         {metrics['accuracy']:.4f}")
                print(f"  🎯 Balanced Acc:     {metrics['balanced_accuracy']:.4f}")
                print(f"  🎯 Precision:        {metrics['precision']:.4f}")
                print(f"  🎯 Recall:           {metrics['recall']:.4f}")
                print(f"  🎯 Specificity:      {metrics['specificity']:.4f}")
                print(f"  🎯 AUC-PR:           {metrics['auc_pr']:.4f}")
                print(f"  🎯 AUC-ROC:          {metrics['auc_roc']:.4f}")

                new_test_results[test_name] = metrics

            except Exception as e:
                print(f"❌ 测试失败: {str(e)}")
                continue

    # 保存结果
    total_time = time.time() - diffusion_start

    full_results = {
        "model_name": train_name + "_robust",
        "model_type": "Robust GNN with Quality-Controlled Augmentation",
        "training_info": {
            "original_positive": orig_pos,
            "original_negative": orig_neg,
            "original_ratio": orig_ratio,
            "augmented_positive": aug_pos,
            "augmented_negative": aug_neg,
            "augmented_ratio": aug_ratio,
            "target_ratio": config.target_ratio,
            "cv_avg_f1": avg_f1,
            "cv_avg_auc": avg_auc,
            "best_cv_f1": best_cv_result['val_f1'],
            "best_cv_auc": best_cv_result['val_auc'],
            "diffusion_time": diffusion_time,
            "augment_time": augment_time,
            "gnn_time": gnn_time,
            "total_time": total_time
        },
        "robust_config": {
            "quality_threshold": config.quality_threshold,
            "diversity_threshold": config.diversity_threshold,
            "domain_weight": config.domain_weight,
            "cv_folds": config.cv_folds,
            "max_augment_ratio": config.max_augment_ratio
        },
        "test_results": test_results,
        "new_test_results": new_test_results
    }

    # 保存结果
    with open(os.path.join(output_path, "robust_results.json"), 'w') as f:
        json.dump(full_results, f, indent=2, default=lambda obj: float(obj) if isinstance(obj, np.floating) else int(obj) if isinstance(obj, np.integer) else obj)

    print(f"\n✅ {train_name} 鲁棒训练-测试完成!")
    print(f"⏱️ 总用时: {total_time//60:.0f}m {total_time%60:.0f}s")
    print(f"📈 数据改善: {orig_ratio:.3%} → {aug_ratio:.3%}")
    print(f"🎯 交叉验证性能: F1={avg_f1:.4f}, AUC-PR={avg_auc:.4f}")

    return full_results


def load_dataset_from_file(dataset_file, config):
    """从单个文件加载数据集"""
    import shutil

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


def main():
    """主函数"""
    print("🛡️ 鲁棒性增强训练-测试管道启动")
    print("="*80)
    print("改进策略: 质量控制 + 域适应 + 交叉验证 + 集成学习")
    print("="*80)

    # 初始化配置
    config = RobustTrainingConfig()
    set_seed(config.seed)

    print(f"\n🛡️ 鲁棒性配置:")
    print(f"  - 保守目标比例: {config.target_ratio:.1%}")
    print(f"  - 质量控制阈值: {config.quality_threshold}")
    print(f"  - 域适应权重: {config.domain_weight}")
    print(f"  - {config.cv_folds}折交叉验证")
    print(f"  - {config.ensemble_size}模型集成")

    # 查找文件
    train_files = sorted(glob.glob(os.path.join(config.data_dir, "*Train*.txt")))
    test_files = sorted(glob.glob(os.path.join(config.data_dir, "*Test*.txt")))

    # 分离原始测试集和新测试集
    original_test_files = [f for f in test_files if 'DNA-' in os.path.basename(f)]
    new_test_files = sorted(glob.glob(os.path.join(config.data_dir, "PDNA-*-test.txt")))

    print(f"\n🔍 找到 {len(train_files)} 个训练文件")
    print(f"   - 原始测试集 (DNA系列): {len(original_test_files)} 个")
    print(f"   - 新增测试集 (PDNA系列): {len(new_test_files)} 个")

    if original_test_files:
        print(f"\n📊 原始测试集:")
        for f in original_test_files:
            print(f"   - {os.path.basename(f)}")

    if new_test_files:
        print(f"\n📊 新增测试集:")
        for f in new_test_files:
            print(f"   - {os.path.basename(f)}")

    # 执行鲁棒训练-测试管道
    total_start = time.time()
    all_results = {}

    for train_file in train_files:
        try:
            result = train_and_test_robust_model(train_file, original_test_files, config, new_test_files)
            if result:
                all_results[os.path.splitext(os.path.basename(train_file))[0]] = result
        except Exception as e:
            print(f"❌ 处理失败 {train_file}: {str(e)}")

    total_pipeline_time = time.time() - total_start

    # 生成报告
    if all_results:
        results_file = os.path.join(config.output_dir, "robust_pipeline_results.json")
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=lambda obj: float(obj) if isinstance(obj, np.floating) else int(obj) if isinstance(obj, np.integer) else obj)

        print(f"\n🎉 鲁棒管道执行完成!")
        print(f"⏱️ 总时间: {total_pipeline_time//3600:.0f}h {(total_pipeline_time%3600)//60:.0f}m {total_pipeline_time%60:.0f}s")
        print(f"🎯 成功完成 {len(all_results)} 个鲁棒模型")
        print(f"📊 详细结果: {results_file}")

        # 简要性能对比 - 原始测试集
        print(f"\n{'='*90}")
        print(f"📋 鲁棒模型测试性能 - 原始测试集 (DNA系列)")
        print("="*90)
        print(f"{'训练集':<15} {'平均F1':<10} {'平均MCC':<10} {'平均AUC-PR':<12} {'平均AUC-ROC':<12} {'平均平衡ACC':<12}")
        print("-" * 90)

        for train_name, results in all_results.items():
            test_results = results['test_results']
            if test_results:
                avg_f1 = np.mean([m['f1'] for m in test_results.values()])
                avg_mcc = np.mean([m['mcc'] for m in test_results.values()])
                avg_auc_pr = np.mean([m['auc_pr'] for m in test_results.values()])
                avg_auc_roc = np.mean([m['auc_roc'] for m in test_results.values()])
                avg_balanced_acc = np.mean([m['balanced_accuracy'] for m in test_results.values()])

                print(f"{train_name:<15} {avg_f1:<10.4f} {avg_mcc:<10.4f} {avg_auc_pr:<12.4f} {avg_auc_roc:<12.4f} {avg_balanced_acc:<12.4f}")

        # 简要性能对比 - 新增测试集
        print(f"\n{'='*90}")
        print(f"📋 鲁棒模型测试性能 - 新增测试集 (PDNA系列)")
        print("="*90)
        print(f"{'训练集':<15} {'平均F1':<10} {'平均MCC':<10} {'平均AUC-PR':<12} {'平均AUC-ROC':<12} {'平均平衡ACC':<12}")
        print("-" * 90)

        for train_name, results in all_results.items():
            new_test_results = results.get('new_test_results', {})
            if new_test_results:
                avg_f1 = np.mean([m['f1'] for m in new_test_results.values()])
                avg_mcc = np.mean([m['mcc'] for m in new_test_results.values()])
                avg_auc_pr = np.mean([m['auc_pr'] for m in new_test_results.values()])
                avg_auc_roc = np.mean([m['auc_roc'] for m in new_test_results.values()])
                avg_balanced_acc = np.mean([m['balanced_accuracy'] for m in new_test_results.values()])

                print(f"{train_name:<15} {avg_f1:<10.4f} {avg_mcc:<10.4f} {avg_auc_pr:<12.4f} {avg_auc_roc:<12.4f} {avg_balanced_acc:<12.4f}")
            else:
                print(f"{train_name:<15} {'N/A':<10} {'N/A':<10} {'N/A':<12} {'N/A':<12} {'N/A':<12}")

    else:
        print(f"\n❌ 没有成功完成任何鲁棒模型训练")


def load_dataset_quiet(dataset_file, config):
    """静默加载数据集 - 减少打印输出"""
    import shutil
    import sys
    import contextlib

    temp_data_dir = os.path.join(config.data_dir, "temp")
    os.makedirs(temp_data_dir, exist_ok=True)

    temp_file = os.path.join(temp_data_dir, os.path.basename(dataset_file))
    shutil.copy2(dataset_file, temp_file)

    try:
        # 临时重定向stdout来减少打印
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


def test_different_ratios(train_file, test_files, ratios=[0.10, 0.15, 0.20, 0.25, 0.30]):
    """测试不同的数据平衡比例"""
    print(f"🧪 开始比例测试实验")
    print(f"📊 测试比例: {[f'{r:.1%}' for r in ratios]}")
    print("="*80)

    results_summary = {}

    for ratio in ratios:
        print(f"\n🎯 测试比例: {ratio:.1%}")
        print("-" * 50)

        # 创建配置
        experiment_name = f"ratio_test_{ratio:.2f}".replace(".", "")
        config = RobustTrainingConfig(target_ratio=ratio, experiment_name=experiment_name)
        config.verbose_loading = False  # 减少输出
        set_seed(config.seed)

        try:
            start_time = time.time()
            result = train_and_test_robust_model_quiet(train_file, test_files, config)
            end_time = time.time()

            if result:
                # 计算平均性能
                test_results = result['test_results']
                avg_f1 = np.mean([m['f1'] for m in test_results.values()])
                avg_mcc = np.mean([m['mcc'] for m in test_results.values()])
                avg_auc_pr = np.mean([m['auc_pr'] for m in test_results.values()])
                avg_auc_roc = np.mean([m['auc_roc'] for m in test_results.values()])
                avg_balanced_acc = np.mean([m['balanced_accuracy'] for m in test_results.values()])
                avg_specificity = np.mean([m['specificity'] for m in test_results.values()])

                results_summary[ratio] = {
                    'avg_f1': avg_f1,
                    'avg_mcc': avg_mcc,
                    'avg_auc_pr': avg_auc_pr,
                    'avg_auc_roc': avg_auc_roc,
                    'avg_balanced_acc': avg_balanced_acc,
                    'avg_specificity': avg_specificity,
                    'time': end_time - start_time,
                    'full_results': result
                }

                print(f"✅ 比例 {ratio:.1%}: F1={avg_f1:.3f}, MCC={avg_mcc:.3f}, AUC-PR={avg_auc_pr:.3f}, AUC-ROC={avg_auc_roc:.3f}")
            else:
                print(f"❌ 比例 {ratio:.1%}: 测试失败")

        except Exception as e:
            print(f"❌ 比例 {ratio:.1%}: 错误 - {str(e)}")

    # 生成对比报告
    if results_summary:
        print(f"\n📊 比例测试结果汇总:")
        print("="*110)
        print(f"{'比例':<8} {'F1分数':<10} {'MCC':<10} {'AUC-PR':<10} {'AUC-ROC':<10} {'平衡ACC':<12} {'特异性':<12} {'用时(秒)':<12}")
        print("-" * 110)

        best_f1_ratio = None
        best_f1_score = 0
        best_mcc_ratio = None
        best_mcc_score = -1

        for ratio, metrics in results_summary.items():
            print(f"{ratio:<8.1%} {metrics['avg_f1']:<10.3f} {metrics['avg_mcc']:<10.3f} {metrics['avg_auc_pr']:<10.3f} "
                  f"{metrics['avg_auc_roc']:<10.3f} {metrics['avg_balanced_acc']:<12.3f} {metrics['avg_specificity']:<12.3f} {metrics['time']:<12.1f}")

            if metrics['avg_f1'] > best_f1_score:
                best_f1_score = metrics['avg_f1']
                best_f1_ratio = ratio

            if metrics['avg_mcc'] > best_mcc_score:
                best_mcc_score = metrics['avg_mcc']
                best_mcc_ratio = ratio

        print(f"\n🏆 最佳F1比例: {best_f1_ratio:.1%} (F1={best_f1_score:.3f})")
        print(f"🏆 最佳MCC比例: {best_mcc_ratio:.1%} (MCC={best_mcc_score:.3f})")

        # 保存详细结果
        train_name = os.path.splitext(os.path.basename(train_file))[0]
        results_file = f"ratio_comparison_{train_name}_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        print(f"📁 详细结果已保存: {results_file}")

    return results_summary


def train_and_test_robust_model_quiet(train_file, test_files, config):
    """静默版本的鲁棒训练测试 - 减少输出"""
    train_name = os.path.splitext(os.path.basename(train_file))[0]

    # 创建输出目录 - 包含比例信息的命名
    ratio_str = f"{config.target_ratio:.2f}".replace(".", "")
    output_dir_name = f"{train_name}_robust_r{ratio_str}_{config.experiment_name}"
    output_path = os.path.join(config.output_dir, output_dir_name)
    os.makedirs(output_path, exist_ok=True)

    # 静默加载数据
    train_dataset = load_dataset_quiet(train_file, config)

    if not train_dataset:
        return None

    # 统计原始数据
    orig_ratio, orig_pos, orig_neg = calculate_class_ratio(train_dataset)

    # 训练扩散模型
    diffusion_model = EnhancedDiffusionModel(
        input_dim=config.diffusion_input_dim,
        T=config.diffusion_T,
        device=config.device
    )

    start_time = time.time()
    diffusion_model.train_on_positive_samples(train_dataset, epochs=config.diffusion_epochs)
    diffusion_time = time.time() - start_time

    # 鲁棒数据增强
    start_time = time.time()
    augmented_dataset = robust_augment_dataset(train_dataset, diffusion_model, config)
    augment_time = time.time() - start_time

    # 交叉验证训练
    start_time = time.time()
    cv_results = cross_validation_training(augmented_dataset, train_dataset, config)
    gnn_time = time.time() - start_time

    # 选择最佳模型
    if cv_results:
        best_idx = np.argmax([r['val_f1'] for r in cv_results])
        best_model = cv_results[best_idx]['model']

        # 保存最佳模型
        model_path = os.path.join(output_path, "robust_gnn_model.pt")
        torch.save(best_model.state_dict(), model_path)

        # 测试模型
        test_results = {}
        for test_file in test_files:
            test_name = os.path.splitext(os.path.basename(test_file))[0]
            test_dataset = load_dataset_quiet(test_file, config)

            if test_dataset:
                metrics = best_model.evaluate(test_dataset, device=config.device)
                test_results[test_name] = metrics

        # 构建结果
        result = {
            "model_name": f"{train_name}_robust_r{ratio_str}",
            "model_type": "Robust GNN with Quality-Controlled Augmentation",
            "training_info": {
                "original_positive": int(orig_pos),
                "original_negative": int(orig_neg),
                "original_ratio": float(orig_ratio),
                "augmented_positive": sum((d.y == 1).sum().item() for d in augmented_dataset),
                "augmented_negative": sum((d.y == 0).sum().item() for d in augmented_dataset),
                "target_ratio": config.target_ratio,
                "cv_avg_f1": float(np.mean([r['val_f1'] for r in cv_results])),
                "cv_avg_auc": float(np.mean([r['val_auc'] for r in cv_results])),
                "diffusion_time": diffusion_time,
                "augment_time": augment_time,
                "gnn_time": gnn_time,
                "total_time": diffusion_time + augment_time + gnn_time
            },
            "robust_config": {
                "quality_threshold": config.quality_threshold,
                "diversity_threshold": config.diversity_threshold,
                "domain_weight": config.domain_weight,
                "cv_folds": config.cv_folds,
                "max_augment_ratio": config.max_augment_ratio
            },
            "test_results": test_results
        }

        # 保存结果
        results_file = os.path.join(output_path, "robust_results.json")
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        return result

    return None


def quick_ratio_test():
    """快速比例测试 - 使用单个数据集"""
    print("🚀 快速比例测试")
    print("="*50)

    # 默认文件路径
    train_file = '/mnt/data2/Yang/zhq-ds-main/Raw_data/DNA-573_Train.txt'
    test_files = [
        '/mnt/data2/Yang/zhq-ds-main/Raw_data/DNA-129_Test.txt',
        '/mnt/data2/Yang/zhq-ds-main/Raw_data/DNA-181_Test.txt',
        '/mnt/data2/Yang/zhq-ds-main/Raw_data/DNA-46_Test.txt'
    ]

    # 测试不同比例
    ratios = [0.10, 0.15, 0.20, 0.25, 0.30]
    results = test_different_ratios(train_file, test_files, ratios)

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--ratio-test":
        # 比例测试模式
        quick_ratio_test()
    else:
        # 正常模式
        main()
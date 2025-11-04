import torch
import numpy as np
import os
import random
from data_loader import ProteinDataset, create_knn_edges
from ddpm_diffusion_model import EnhancedDiffusionModel
from gnn_model import BindingSiteGNN, set_seed
from config import Config
from tqdm import tqdm
import torch_geometric.data as tg_data
import traceback
import time

# 确保Data类可用
Data = tg_data.Data


def calculate_class_ratio(dataset):
    """计算数据集的类别比例"""
    total_pos = 0
    total_neg = 0

    for data in dataset:
        total_pos += (data.y == 1).sum().item()
        total_neg += (data.y == 0).sum().item()

    total_samples = total_pos + total_neg
    ratio = total_pos / total_samples if total_samples > 0 else 0
    return ratio, total_pos, total_neg


def split_dataset(dataset, test_ratio=0.2):
    """分割数据集为训练集和测试集"""
    if not dataset:
        return [], []

    random.shuffle(dataset)
    split_idx = int(len(dataset) * (1 - test_ratio))
    return dataset[:split_idx], dataset[split_idx:]


def augment_dataset(dataset, diffusion_model, config):
    """增强数据集以达到目标比例"""
    augmented_data = []

    # 为每个蛋白质图生成新样本
    for data in tqdm(dataset, desc="Augmenting proteins"):
        try:
            protein_context = data.protein_context.to(config.device)

            # 计算当前比例
            n_pos = (data.y == 1).sum().item()
            n_neg = (data.y == 0).sum().item()
            total_nodes = n_pos + n_neg

            # 避免除以零
            current_ratio = n_pos / total_nodes if total_nodes > 0 else 0

            # 计算目标正样本数量
            target_pos = int(total_nodes * config.target_ratio)

            # 计算需要生成的数量
            n_to_generate = max(config.min_samples_per_protein, target_pos - n_pos)

            print(f"Generating {n_to_generate} positive samples for {data.name} (current: {n_pos} pos, {n_neg} neg)")

            # 生成新样本
            new_samples = diffusion_model.generate_positive_sample(
                protein_context,
                num_samples=n_to_generate
            )

            if new_samples is None or len(new_samples) == 0:
                print(f"Warning: Failed to generate samples for {data.name}")
                augmented_data.append(data)
                continue

            # 创建新节点
            new_x = torch.tensor(new_samples, dtype=torch.float32)
            new_y = torch.ones(new_x.size(0), dtype=torch.long)

            # 合并到原始图
            updated_x = torch.cat([data.x, new_x], dim=0)
            updated_y = torch.cat([data.y, new_y], dim=0)

            # 创建新边 (改进的KNN)
            updated_edge_index = create_knn_edges(updated_x, k=config.knn_k, max_samples=2000)

            # 创建增强后的图
            augmented_graph = Data(
                x=updated_x,
                edge_index=updated_edge_index,
                y=updated_y,
                protein_context=data.protein_context,
                name=data.name + "_aug"
            )
            augmented_data.append(augmented_graph)
        except Exception as e:
            print(f"Error augmenting {data.name}: {traceback.format_exc()}")
            augmented_data.append(data)

    return augmented_data


def oversample_positive_nodes(dataset, config):
    """对正样本节点进行过采样，限制最大节点数"""
    oversampled_data = []

    for data in tqdm(dataset, desc="Oversampling"):
        # 跳过空图
        if data.x.size(0) == 0:
            oversampled_data.append(data)
            continue

        # 分离正负样本
        pos_mask = (data.y == 1)
        neg_mask = (data.y == 0)

        pos_x = data.x[pos_mask]
        neg_x = data.x[neg_mask]

        n_pos = pos_x.size(0)
        n_neg = neg_x.size(0)

        # 计算目标正样本数量 (使用配置文件中的目标比例)
        target_pos = min(int((n_pos + n_neg) * config.target_ratio), int(config.max_nodes_per_graph * config.target_ratio))
        n_to_generate = max(0, target_pos - n_pos)

        if n_to_generate > 0:
            # 过采样正样本
            if n_pos > 0:
                indices = np.random.choice(n_pos, n_to_generate, replace=True)
                oversampled_x = pos_x[indices]
            else:
                # 如果没有正样本，创建随机样本
                oversampled_x = torch.randn(n_to_generate, data.x.size(1))

            # 限制负样本数量
            max_neg = min(n_neg, config.max_nodes_per_graph - n_pos - n_to_generate)
            if max_neg < n_neg:
                neg_indices = np.random.choice(n_neg, max_neg, replace=False)
                neg_x = neg_x[neg_indices]
                n_neg = max_neg

            # 合并样本
            all_x = torch.cat([pos_x, oversampled_x, neg_x], dim=0)
            all_y = torch.cat([
                torch.ones(n_pos + n_to_generate, dtype=torch.long),
                torch.zeros(n_neg, dtype=torch.long)
            ])
        else:
            # 不需要过采样
            all_x = torch.cat([pos_x, neg_x], dim=0)
            all_y = torch.cat([
                torch.ones(n_pos, dtype=torch.long),
                torch.zeros(n_neg, dtype=torch.long)
            ])

        # 如果节点数超过限制，随机采样
        if len(all_x) > config.max_nodes_per_graph:
            indices = np.random.choice(len(all_x), config.max_nodes_per_graph, replace=False)
            all_x = all_x[indices]
            all_y = all_y[indices]

        # 创建新边
        edge_index = create_knn_edges(all_x, k=5, max_samples=500)

        # 创建新图
        oversampled_graph = Data(
            x=all_x,
            edge_index=edge_index,
            y=all_y,
            protein_context=data.protein_context,
            name=data.name + "_oversampled"
        )
        oversampled_data.append(oversampled_graph)

    return oversampled_data


def main():
    try:
        start_time = time.time()

        # 加载配置
        config = Config()

        # 设置随机种子
        set_seed(config.seed)

        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        print(f"Created output directory: {config.output_dir}")

        # 加载数据集
        print("Loading dataset...")
        dataset_loader = ProteinDataset(config.data_dir, device=config.device)
        dataset = dataset_loader.proteins

        if len(dataset) == 0:
            print(f"Error: No proteins found in {config.data_dir}")
            return

        print(f"Successfully loaded {len(dataset)} proteins")
        print(f"Data loading time: {time.time() - start_time:.2f} seconds")

        # 检查整个数据集中是否有正样本
        total_pos = sum((data.y == 1).sum().item() for data in dataset)
        if total_pos == 0:
            print("Warning: No positive samples found in the entire dataset")
            print("Diffusion model will use random generation strategy")

        # 分割数据集
        train_data, test_data = split_dataset(dataset, config.test_ratio)
        print(f"Split dataset: {len(train_data)} training, {len(test_data)} testing")

        if not train_data:
            print("Error: Training set is empty")
            return

        # 计算原始类别比例
        orig_ratio, orig_pos, orig_neg = calculate_class_ratio(train_data)
        print(f"Original data - Positive ratio: {orig_ratio:.4f} ({orig_pos} pos, {orig_neg} neg)")

        # 初始化扩散模型
        print("\nInitializing diffusion model...")
        diffusion_model = EnhancedDiffusionModel(
            input_dim=config.diffusion_input_dim,
            T=config.diffusion_T,
            device=config.device
        )

        # 训练扩散模型
        print("\nTraining diffusion model...")
        diffusion_start = time.time()
        diffusion_model.train_on_positive_samples(
            train_data,
            epochs=config.diffusion_epochs,
            batch_size=config.diffusion_batch_size
        )
        print(f"Diffusion model training time: {time.time() - diffusion_start:.2f} seconds")

        # 保存模型
        if config.save_diffusion_model:
            model_path = os.path.join(config.output_dir, "diffusion_model.pt")
            torch.save(diffusion_model.state_dict(), model_path)
            print(f"Diffusion model saved to {model_path}")

        # 增强训练数据集
        print("\nAugmenting training dataset...")
        augment_start = time.time()
        augmented_train_data = augment_dataset(train_data, diffusion_model, config)
        print(f"Augmentation time: {time.time() - augment_start:.2f} seconds")

        # 计算增强后的类别比例
        if augmented_train_data:
            aug_ratio, aug_pos, aug_neg = calculate_class_ratio(augmented_train_data)
            print(f"Augmented data - Positive ratio: {aug_ratio:.4f} ({aug_pos} pos, {aug_neg} neg)")
        else:
            print("Warning: Augmented dataset is empty")
            augmented_train_data = train_data

        # 对正样本节点进行过采样
        print("\nOversampling positive nodes...")
        oversample_start = time.time()
        oversampled_train_data = oversample_positive_nodes(
            augmented_train_data,
            config
        )
        print(f"Oversampling time: {time.time() - oversample_start:.2f} seconds")

        # 计算过采样后的比例
        if oversampled_train_data:
            oversampled_ratio, oversampled_pos, oversampled_neg = calculate_class_ratio(oversampled_train_data)
            print(
                f"Oversampled data - Positive ratio: {oversampled_ratio:.4f} ({oversampled_pos} pos, {oversampled_neg} neg)")

        # 保存增强后的数据集
        if config.save_augmented_data and oversampled_train_data:
            print("\nSaving augmented dataset...")
            save_start = time.time()
            for i, data in enumerate(oversampled_train_data):
                save_path = os.path.join(config.output_dir, f'aug_data_{i}.pt')
                torch.save(data, save_path)
            print(f"Dataset saving time: {time.time() - save_start:.2f} seconds")

        # 训练下游GNN模型
        print("\nTraining downstream GNN model...")
        gnn_model = BindingSiteGNN(
            input_dim=config.diffusion_input_dim,
            hidden_dim=config.gnn_hidden_dim,
            dropout=config.gnn_dropout
        )

        # 检查验证集是否为空
        if not test_data:
            print("Warning: Validation set is empty, training only")
            val_data_for_train = []
        else:
            val_data_for_train = test_data

        # 训练模型
        gnn_start = time.time()
        best_auc, best_f1 = gnn_model.train_model(
            oversampled_train_data,
            val_data_for_train,
            epochs=config.gnn_epochs,
            lr=config.gnn_lr,
            device=config.device,
            patience=config.gnn_patience
        )
        print(f"GNN training time: {time.time() - gnn_start:.2f} seconds")

        # 最终评估
        if test_data:
            print("\nFinal evaluation on test set:")
            test_metrics = gnn_model.evaluate(test_data, device=config.device)
            print(f"F1 Score: {test_metrics['f1']:.4f}")
            print(f"Matthews Correlation Coefficient (MCC): {test_metrics['mcc']:.4f}")
            print(f"Area Under Precision-Recall Curve (AUC-PR): {test_metrics['auc_pr']:.4f}")
        else:
            print("\nNo test set available for final evaluation")

        total_time = time.time() - start_time
        print(
            f"\nTotal process time: {total_time // 3600:.0f}h {(total_time % 3600) // 60:.0f}m {total_time % 60:.2f}s")
        print("Process completed!")

    except Exception as e:
        print(f"Unhandled exception in main: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
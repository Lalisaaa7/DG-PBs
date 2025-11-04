import torch
import os


class Config:
    def __init__(self):
        # 随机种子
        self.seed = 42

        # 设备配置 - 使用GPU 6 (空闲)
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '6'  # 使用GPU 6
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device} (Physical GPU 6)")

        # 数据配置
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(current_dir, "Raw_data")
        self.output_dir = os.path.join(current_dir, "Augmented_data")
        self.test_ratio = 0.2

        # 扩散模型配置 - 针对A100优化
        self.diffusion_input_dim = 1280  # ESM嵌入维度
        self.diffusion_T = 200  # 减少时间步数 (500->200) 
        self.diffusion_epochs = 50  # 减少训练轮数 (150->50)
        self.diffusion_batch_size = 128  # 增大批大小利用A100内存 (32->128)
        self.diffusion_lr = 2e-4  # 提高学习率 (1e-4->2e-4)

        # 增强配置 - 1:1完全平衡
        self.target_ratio = 0.5  # 目标正样本比例 (50%, 1:1 完全平衡)
        self.min_samples_per_protein = 30  # 增加最少生成样本数支持1:1平衡 (20->30)
        self.knn_k = 3  # KNN邻居数 (保持优化值)
        self.oversample_ratio = 2.0  # 增加过采样支持1:1平衡 (1.5->2.0)
        self.max_nodes_per_graph = 2000  # 增加最大节点数利用A100内存 (1000->2000)

        # GNN模型配置 - A100优化
        self.gnn_hidden_dim = 512  # 增大隐藏层维度利用A100算力 (256->512)
        self.gnn_epochs = 50  # 减少训练轮数 (100->50)
        self.gnn_lr = 1e-3  # 提高学习率 (5e-4->1e-3)
        self.gnn_dropout = 0.2  # 减少dropout (0.3->0.2)
        self.gnn_patience = 8  # 减少早停耐心值 (15->8)

        # 保存选项
        self.save_diffusion_model = True
        self.save_augmented_data = True

        # 打印路径信息
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from scipy import stats
import warnings


class TimeEmbedding(nn.Module):
    def __init__(self, T, dim):
        super().__init__()
        self.embed = nn.Embedding(T, dim)
        self.linear = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t):
        return self.linear(self.embed(t))


class DiffusionPredictor(nn.Module):
    def __init__(self, input_dim, T, protein_dim=1280):
        super().__init__()
        # 时间嵌入
        self.time_embed = TimeEmbedding(T, 256)

        # 蛋白质结构条件编码器
        self.protein_cond = nn.Sequential(
            nn.Linear(protein_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )

        # 增强核心网络
        self.net = nn.Sequential(
            nn.Linear(input_dim + 256 + 256, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, input_dim)
        )

    def forward(self, x_t, t, protein_context):
        # 时间嵌入
        te = self.time_embed(t)

        # 蛋白质条件
        pc = self.protein_cond(protein_context)

        # 融合输入
        combined = torch.cat([x_t, te, pc], dim=-1)
        return self.net(combined)


class DiffusionProcess:
    def __init__(self, beta_start=1e-4, beta_end=0.02, num_timesteps=1000, device='cpu', schedule='cosine'):
        self.num_timesteps = num_timesteps
        self.device = device
        self.schedule = schedule

        # 使用余弦调度或线性调度
        if schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(num_timesteps, s=0.008).to(device)
        else:
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
            
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """余弦调度，通常比线性调度效果更好"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999).float()

    def diffuse(self, x_0, t):
        """前向扩散过程"""
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t])
        sqrt_one_minus_alpha_bar = torch.sqrt(1. - self.alpha_bars[t])
        noise = torch.randn_like(x_0)
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise

    def sample_timesteps(self, n):
        """随机采样时间步"""
        return torch.randint(0, self.num_timesteps, (n,), device=self.device)


class EnhancedDiffusionModel(nn.Module):
    def __init__(self, input_dim, T=500, protein_dim=1280, device='cpu'):
        super().__init__()
        self.T = T
        self.device = device
        self.input_dim = input_dim
        self.protein_dim = protein_dim
        self.has_positive_samples = False
        self.mean = None
        self.std = None
        self.training_losses = []

        # 扩散模型组件 - 使用余弦调度
        self.predictor = DiffusionPredictor(input_dim, T, protein_dim).to(device)
        self.diffusion = DiffusionProcess(num_timesteps=T, device=device, schedule='cosine')

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.predictor.parameters(),
            lr=1e-4,
            weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

    def train_on_positive_samples(self, all_data, epochs=100, batch_size=64):
        """训练扩散模型"""
        positive_vectors = []
        protein_contexts = []

        # 收集正样本及其蛋白质上下文
        for data in all_data:
            x = data.x.to(self.device)
            y = data.y.to(self.device)
            pos_mask = (y == 1)

            if pos_mask.sum() > 0:
                self.has_positive_samples = True
                pos_feats = x[pos_mask]
                protein_ctx = data.protein_context.to(self.device)

                positive_vectors.append(pos_feats)
                # 为每个正样本添加相同的蛋白质上下文
                protein_contexts.extend([protein_ctx] * len(pos_feats))

        if not positive_vectors:
            print("No positive samples available for training - skipping diffusion model training")
            return

        full_pos_data = torch.cat(positive_vectors, dim=0).to(self.device)
        protein_contexts = torch.stack(protein_contexts).to(self.device)

        data_size = full_pos_data.size(0)
        print(f"Total positive samples: {data_size}, starting diffusion model training")

        # 特征归一化 - 存储归一化参数
        self.mean = full_pos_data.mean(dim=0)
        self.std = full_pos_data.std(dim=0) + 1e-8
        full_pos_data = (full_pos_data - self.mean) / self.std
        
        print(f"Data normalization: mean range [{self.mean.min():.4f}, {self.mean.max():.4f}], "
              f"std range [{self.std.min():.4f}, {self.std.max():.4f}]")

        # 训练循环
        for epoch in range(epochs):
            perm = torch.randperm(data_size, device=self.device)
            losses = []
            for i in range(0, data_size, batch_size):
                idx = perm[i:i + batch_size]
                batch = full_pos_data[idx]
                ctx_batch = protein_contexts[idx]

                t = self.diffusion.sample_timesteps(batch.size(0))
                eps = torch.randn_like(batch)

                # 计算alpha_bar
                sqrt_ab = torch.sqrt(self.diffusion.alpha_bars[t])[:, None]
                sqrt_1m_ab = torch.sqrt(1 - self.diffusion.alpha_bars[t])[:, None]

                # 添加噪声
                x_t = sqrt_ab * batch + sqrt_1m_ab * eps

                # 预测噪声
                eps_pred = self.predictor(x_t, t, ctx_batch)

                # 计算损失
                loss = F.mse_loss(eps_pred, eps)

                self.optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)

                self.optimizer.step()
                losses.append(loss.item())

            avg_loss = np.mean(losses)
            self.training_losses.append(avg_loss)
            self.scheduler.step(avg_loss)
            
            # 监控训练质量
            if epoch % 20 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
                
                # 生成少量样本验证质量
                if epoch > 10:
                    self._validate_generation_quality(protein_contexts[:1], full_pos_data[:100])
            
            torch.cuda.empty_cache()

    def generate_positive_sample(self, protein_context, num_samples=100, add_diversity=True, verbose=True, max_batch_size=32):
        """
        生成正样本。改进点：
        1. 分块生成，避免 CPU/Windows 下单次反采样过重；
        2. 每一步清理 NaN/Inf，并限制归一化空间极端值；
        3. 反归一化后限制到训练正样本均值±5倍标准差，减少离群样本。
        """
        if num_samples <= 0:
            return np.empty((0, self.input_dim), dtype=np.float32)

        if not self.has_positive_samples:
            print("No positive samples for training - generation disabled")
            return None

        all_samples = []
        remaining = int(num_samples)

        while remaining > 0:
            cur_n = min(remaining, max_batch_size)
            remaining -= cur_n

            with torch.no_grad():
                x_t = torch.randn(cur_n, self.input_dim, device=self.device)

                if protein_context is not None:
                    pc = protein_context.to(self.device)
                    if pc.dim() == 1:
                        pc = pc.unsqueeze(0)
                    pc = pc.repeat(cur_n, 1)
                    if add_diversity:
                        pc = pc + torch.randn_like(pc) * 0.005
                    diverse_context = pc
                else:
                    diverse_context = None

                for t in reversed(range(self.T)):
                    t_tensor = torch.full((cur_n,), t, device=self.device, dtype=torch.long)
                    beta = self.diffusion.betas[t]
                    alpha = 1.0 - beta
                    ab = self.diffusion.alpha_bars[t]

                    eps_pred = self.predictor(x_t, t_tensor, diverse_context)
                    eps_pred = torch.nan_to_num(eps_pred, nan=0.0, posinf=0.0, neginf=0.0)

                    coeff1 = 1.0 / torch.sqrt(alpha)
                    coeff2 = (1.0 - alpha) / torch.sqrt(torch.clamp(1.0 - ab, min=1e-8))
                    mean = coeff1 * (x_t - coeff2 * eps_pred)
                    mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)

                    if t > 0:
                        noise = torch.randn_like(x_t)
                        x_t = mean + torch.sqrt(beta) * noise
                    else:
                        x_t = mean

                    # 避免反采样跑飞。这里是在归一化空间裁剪，不是复制真实样本。
                    x_t = torch.clamp(torch.nan_to_num(x_t, nan=0.0, posinf=0.0, neginf=0.0), -6.0, 6.0)

                if self.mean is not None and self.std is not None:
                    mean = self.mean.to(self.device)
                    std = self.std.to(self.device)
                    x_t = x_t * std + mean
                    low = mean - 5.0 * std
                    high = mean + 5.0 * std
                    x_t = torch.max(torch.min(x_t, high), low)

                x_t = torch.nan_to_num(x_t, nan=0.0, posinf=0.0, neginf=0.0)
                all_samples.append(x_t.cpu().float())

        generated_samples = torch.cat(all_samples, dim=0).numpy()

        if verbose:
            quality_score = self._evaluate_sample_quality(generated_samples)
            print(f"Generated {num_samples} diffusion samples with quality score: {quality_score:.4f}")

        return generated_samples

    def _validate_generation_quality(self, protein_contexts, real_samples):
        """训练过程中验证生成质量"""
        if protein_contexts is None or len(protein_contexts) == 0:
            return
            
        try:
            # 生成少量样本进行验证
            test_samples = self.generate_positive_sample(
                protein_contexts[0], num_samples=min(50, len(real_samples)), add_diversity=False
            )
            
            if test_samples is not None and len(test_samples) > 0:
                quality_score = self._evaluate_sample_quality(test_samples, real_samples)
                print(f"  -> Validation quality score: {quality_score:.4f}")
        except Exception as e:
            print(f"  -> Validation failed: {str(e)}")
    
    def _evaluate_sample_quality(self, generated_samples, real_samples=None):
        """评估生成样本的质量"""
        try:
            if generated_samples is None or len(generated_samples) == 0:
                return 0.0
            
            # 基本统计检查
            if isinstance(generated_samples, torch.Tensor):
                gen_samples = generated_samples.detach().cpu().numpy()
            else:
                gen_samples = np.array(generated_samples)
            
            # 检查是否包含异常值
            has_nan = np.isnan(gen_samples).any()
            has_inf = np.isinf(gen_samples).any()
            
            if has_nan or has_inf:
                print("Warning: Generated samples contain NaN or Inf values")
                return 0.0
            
            # 特征范围合理性
            feature_std = np.std(gen_samples, axis=0)
            reasonable_variance = np.mean(feature_std > 1e-6)  # 避免特征塌陷
            
            # 如果有真实样本，计算分布相似性
            if real_samples is not None and len(real_samples) > 10:
                if isinstance(real_samples, torch.Tensor):
                    real_samples = real_samples.detach().cpu().numpy()
                else:
                    real_samples = np.array(real_samples)
                try:
                    # KL散度近似（使用直方图）
                    kl_divs = []
                    for i in range(min(10, gen_samples.shape[1])):  # 只检查前10个特征
                        gen_hist, bins = np.histogram(gen_samples[:, i], bins=10, density=True)
                        real_hist, _ = np.histogram(real_samples[:, i], bins=bins, density=True)
                        
                        # 避免零值
                        gen_hist = gen_hist + 1e-8
                        real_hist = real_hist + 1e-8
                        
                        kl_div = stats.entropy(gen_hist, real_hist)
                        if not np.isnan(kl_div) and not np.isinf(kl_div):
                            kl_divs.append(kl_div)
                    
                    if kl_divs:
                        avg_kl = np.mean(kl_divs)
                        distribution_similarity = max(0, 1 - avg_kl / 5)  # 归一化到[0,1]
                    else:
                        distribution_similarity = 0.5
                except:
                    distribution_similarity = 0.5
                
                return 0.5 * reasonable_variance + 0.5 * distribution_similarity
            else:
                return reasonable_variance
                
        except Exception as e:
            print(f"Quality evaluation error: {str(e)}")
            return 0.0
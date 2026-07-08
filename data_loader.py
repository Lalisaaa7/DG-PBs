#!/usr/bin/env python3
"""
data_loader.py

完整可替换版本：
1. 使用 ESM-2 提取 1280 维残基级嵌入；
2. 使用 PyTorch 手动计算余弦相似度并构建 KNN 图；
3. 不再调用 sklearn.neighbors.kneighbors_graph(metric='cosine')，避免 Windows 下真实 ESM embedding 触发底层崩溃；
4. 输出无权 edge_index，适配 PyTorch Geometric；
5. 对超长序列做保护性跳过，避免 ESM-2 位置编码长度问题。
"""

import os
import glob
import re
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import esm


class ProteinDataset:
    def __init__(self, path, device='cpu', max_sequence_length=1022, verbose=True):
        self.path = path
        self.device = device
        self.max_sequence_length = max_sequence_length
        self.verbose = verbose

        # 加载 ESM-2 650M 模型
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.model = self.model.to(device)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()

        self.proteins = self.load_all()

    def _parse_protein_record(self, lines, file_name, record_index):
        """解析一个蛋白质记录：标识行、序列行、标签行。"""
        try:
            name_line = lines[0].strip()
            seq_line = lines[1].strip()
            label_line = lines[2].strip()

            name_match = re.match(r'>(\S+)', name_line)
            name = name_match.group(1) if name_match else f"{os.path.basename(file_name)}_record{record_index}"

            sequence = seq_line.strip().upper()

            # ESM-2 常用最大长度约为 1022；超长序列容易触发位置编码或内存问题。
            if self.max_sequence_length is not None and len(sequence) > self.max_sequence_length:
                print(f"Skip {name}: sequence too long ({len(sequence)} > {self.max_sequence_length})", flush=True)
                return None

            labels = [int(char) if char in '01' else 0 for char in label_line.strip()]

            if len(labels) != len(sequence):
                print(
                    f"Warning: Label length mismatch for {name}: "
                    f"seq_len={len(sequence)}, label_len={len(labels)}. Skip this protein.",
                    flush=True
                )
                return None

            if len(sequence) == 0:
                print(f"Warning: Empty sequence for {name}. Skip this protein.", flush=True)
                return None

            # 使用 ESM-2 提取残基级 1280 维嵌入
            data = [(name, sequence)]
            _, _, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)

            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[33])

            token_representations = results["representations"][33]

            # 去掉 CLS/EOS 标记，只保留残基表示
            embeddings = token_representations[0, 1:len(sequence) + 1, :]
            protein_context = embeddings.mean(dim=0)

            # 原始残基图：基于 ESM-2 嵌入空间的余弦相似度 KNN
            edge_index = self._create_knn_graph(embeddings.detach().cpu(), k=7)

            return Data(
                x=embeddings.detach().cpu(),
                edge_index=edge_index,
                y=torch.tensor(labels, dtype=torch.long),
                protein_context=protein_context.detach().cpu(),
                name=name
            )
        except Exception:
            print(f"Error parsing record: {traceback.format_exc()}", flush=True)
            return None

    def _create_knn_graph(self, features, k=7):
        """
        使用 PyTorch 手动计算余弦相似度构建 KNN 图。

        逻辑：
        1. 对每个残基嵌入做 L2 归一化；
        2. 计算 sim = x_norm @ x_norm.T，即余弦相似度矩阵；
        3. 排除自身；
        4. 对每个节点选相似度最高的 k 个邻居；
        5. 返回 PyG 所需的 edge_index，形状为 [2, num_edges]。
        """
        try:
            return build_torch_cosine_knn_edges(features, k=k, max_samples=None, verbose=False)
        except Exception as e:
            print(f"Error creating torch cosine KNN graph: {str(e)}", flush=True)
            return torch.empty((2, 0), dtype=torch.long)

    def load_all(self):
        proteins = []
        if not os.path.exists(self.path):
            print(f"Error: Directory {self.path} does not exist", flush=True)
            return proteins

        files = sorted(glob.glob(os.path.join(self.path, '*.txt')))
        if not files:
            print(f"Warning: No .txt files found in {self.path}", flush=True)
            return proteins

        print(f"Found {len(files)} files in data directory", flush=True)

        for file in files:
            try:
                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().splitlines()

                record_starts = [i for i, line in enumerate(content) if line.startswith('>')]
                if not record_starts:
                    continue

                record_starts.append(len(content))

                for i in range(len(record_starts) - 1):
                    start_idx = record_starts[i]
                    end_idx = record_starts[i + 1]
                    record_lines = content[start_idx:end_idx]

                    if not record_lines or len(record_lines) < 3:
                        continue

                    protein_data = self._parse_protein_record(record_lines[:3], file, i + 1)
                    if protein_data is not None:
                        proteins.append(protein_data)
                        if self.verbose:
                            pos_count = int((protein_data.y == 1).sum().item())
                            print(
                                f"Loaded {protein_data.name}: "
                                f"{len(protein_data.y)} residues, {pos_count} positive, "
                                f"edges={protein_data.edge_index.size(1)}",
                                flush=True
                            )
            except Exception:
                print(f"Error processing {file}: {traceback.format_exc()}", flush=True)
                continue

        return proteins

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, idx):
        return self.proteins[idx]


def _to_cpu_float_tensor(features):
    """把 numpy / torch / list 特征安全转成 CPU float32 Tensor。"""
    if isinstance(features, torch.Tensor):
        x = features.detach().cpu().float()
    elif isinstance(features, np.ndarray):
        x = torch.tensor(features, dtype=torch.float32)
    else:
        x = torch.tensor(np.asarray(features), dtype=torch.float32)

    # 清理 nan / inf，避免归一化或 topk 出错
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def build_torch_cosine_knn_edges(features, k=9, max_samples=None, verbose=False):
    """
    纯 PyTorch 余弦相似度 KNN 构图。

    参数：
    - features: [N, D] 节点特征；
    - k: 每个节点连接的近邻数；
    - max_samples: 如果图太大，只对采样节点构图，防止 N*N 相似度矩阵过大；
    - verbose: 是否打印构图信息。
    """
    x = _to_cpu_float_tensor(features)
    n_nodes = x.size(0)

    if n_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long)

    if k <= 0:
        return torch.empty((2, 0), dtype=torch.long)

    # 大图保护：对采样节点构图并映射回原始编号
    if max_samples is not None and n_nodes > max_samples:
        sample_size = int(max_samples)
        if sample_size <= 1:
            return torch.empty((2, 0), dtype=torch.long)

        perm = torch.randperm(n_nodes)[:sample_size]
        x_sub = x[perm]
        n_sub = x_sub.size(0)
        k_eff = min(k, n_sub - 1)

        if k_eff <= 0:
            return torch.empty((2, 0), dtype=torch.long)

        x_norm = F.normalize(x_sub, p=2, dim=1, eps=1e-12)
        sim = torch.matmul(x_norm, x_norm.t())
        sim.fill_diagonal_(-float('inf'))

        _, knn_idx = torch.topk(sim, k=k_eff, dim=1, largest=True, sorted=False)

        row_sub = torch.arange(n_sub).unsqueeze(1).expand(-1, k_eff).reshape(-1)
        col_sub = knn_idx.reshape(-1)

        row = perm[row_sub]
        col = perm[col_sub]
        edge_index = torch.stack([row, col], dim=0).long()

        if verbose:
            print(
                f"Created torch cosine KNN graph with subsampling: "
                f"{n_nodes} nodes, sampled={n_sub}, edges={edge_index.size(1)}, k={k_eff}",
                flush=True
            )
        return edge_index

    k_eff = min(k, n_nodes - 1)
    if k_eff <= 0:
        return torch.empty((2, 0), dtype=torch.long)

    x_norm = F.normalize(x, p=2, dim=1, eps=1e-12)
    sim = torch.matmul(x_norm, x_norm.t())
    sim.fill_diagonal_(-float('inf'))

    _, knn_idx = torch.topk(sim, k=k_eff, dim=1, largest=True, sorted=False)

    row = torch.arange(n_nodes).unsqueeze(1).expand(-1, k_eff).reshape(-1)
    col = knn_idx.reshape(-1)
    edge_index = torch.stack([row, col], dim=0).long()

    if verbose:
        print(f"Created torch cosine KNN graph: {n_nodes} nodes, {edge_index.size(1)} edges, k={k_eff}", flush=True)

    return edge_index


def create_knn_edges(features, k=9, max_samples=10000, verbose=True):
    """
    对外接口：构建余弦相似度 KNN 边。

    robust_pipeline.py 和 main.py 会调用这个函数重构增强图。
    这里不再依赖 sklearn.neighbors.kneighbors_graph，避免真实 ESM embedding 下底层崩溃。
    """
    try:
        return build_torch_cosine_knn_edges(
            features,
            k=k,
            max_samples=max_samples,
            verbose=verbose
        )
    except Exception as e:
        print(f"Error creating torch cosine KNN edges: {str(e)}", flush=True)

        # 后备：线性链图，保证程序不中断
        if isinstance(features, torch.Tensor):
            n_nodes = features.size(0)
        else:
            n_nodes = len(features)

        if n_nodes > 1:
            row = torch.arange(n_nodes - 1)
            col = torch.arange(1, n_nodes)
            return torch.stack([row, col], dim=0).long()

        return torch.empty((2, 0), dtype=torch.long)

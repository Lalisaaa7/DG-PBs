import torch
import os
import numpy as np
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph
import glob
import re
import traceback
import esm


class ProteinDataset:
    def __init__(self, path, device='cpu'):
        self.path = path
        self.device = device

        # 加载ESM模型
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.model = self.model.to(device)
        self.model.eval()  # 设置为评估模式
        self.batch_converter = self.alphabet.get_batch_converter()

        self.proteins = self.load_all()

    def _parse_protein_record(self, lines, file_name, record_index):
        """解析一个蛋白质记录（三行：标识行、序列行、标签行）"""
        try:
            # 获取三行记录
            name_line = lines[0].strip()
            seq_line = lines[1].strip()
            label_line = lines[2].strip()

            # 从标识行提取名称
            name_match = re.match(r'>(\S+)', name_line)
            if name_match:
                name = name_match.group(1)
            else:
                name = f"{os.path.basename(file_name)}_record{record_index}"

            sequence = seq_line

            # 标签处理
            labels = [int(char) if char in '01' else 0 for char in label_line]

            # 确保标签和序列长度一致
            if len(labels) != len(sequence):
                labels = [0] * len(sequence)
                print(f"Warning: Label length mismatch for {name}, using all zeros")

            # 使用ESM获取嵌入
            data = [(name, sequence)]
            batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)

            # 提取嵌入
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[33])
            token_representations = results["representations"][33]

            # 移除CLS和SEP标记
            residue_representations = token_representations[0, 1:len(sequence) + 1, :]
            embeddings = residue_representations

            # 计算蛋白质上下文
            protein_context = embeddings.mean(dim=0)

            # 创建边索引
            edge_index = self._create_knn_graph(embeddings.cpu().numpy(), k=5)

            return Data(
                x=embeddings.cpu(),
                edge_index=edge_index,
                y=torch.tensor(labels, dtype=torch.long),
                protein_context=protein_context.cpu(),
                name=name
            )
        except Exception as e:
            print(f"Error parsing record: {traceback.format_exc()}")
            return None

    def _create_knn_graph(self, features, k=5):
        """创建K最近邻图"""
        try:
            if len(features) < k:
                # 创建自环
                edge_index = torch.zeros((2, len(features)), dtype=torch.long)
                for i in range(len(features)):
                    edge_index[0, i] = i
                    edge_index[1, i] = i
                return edge_index

            adj = kneighbors_graph(features, k, mode='connectivity', include_self=False)
            edge_index = torch.tensor(np.array(adj.nonzero()), dtype=torch.long)
            return edge_index
        except Exception as e:
            print(f"Error creating KNN graph: {str(e)}")
            return torch.empty((2, 0), dtype=torch.long)

    def load_all(self):
        proteins = []
        if not os.path.exists(self.path):
            print(f"Error: Directory {self.path} does not exist")
            return proteins

        # 获取所有txt文件
        files = glob.glob(os.path.join(self.path, '*.txt'))
        if not files:
            print(f"Warning: No .txt files found in {self.path}")
            return proteins

        print(f"Found {len(files)} files in data directory")

        for file in files:
            try:
                with open(file, 'r') as f:
                    content = f.read().splitlines()

                # 查找所有以 '>' 开头的行
                record_starts = [i for i, line in enumerate(content) if line.startswith('>')]

                if not record_starts:
                    continue

                # 添加文件结束位置
                record_starts.append(len(content))

                # 处理每个记录
                for i in range(len(record_starts) - 1):
                    start_idx = record_starts[i]
                    end_idx = record_starts[i + 1]
                    record_lines = content[start_idx:end_idx]

                    if not record_lines or len(record_lines) < 3:
                        continue

                    protein_data = self._parse_protein_record(record_lines[:3], file, i + 1)
                    if protein_data is not None:
                        proteins.append(protein_data)
                        pos_count = (protein_data.y == 1).sum().item()
                        print(f"Loaded {protein_data.name}: {len(protein_data.y)} residues, {pos_count} positive")
            except Exception as e:
                print(f"Error processing {file}: {traceback.format_exc()}")
                continue

        return proteins

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, idx):
        return self.proteins[idx]


def create_knn_edges(features, k=5):
    """创建K最近邻边索引"""
    from sklearn.neighbors import kneighbors_graph
    try:
        if len(features) < k:
            edge_index = torch.zeros((2, len(features)), dtype=torch.long)
            for i in range(len(features)):
                edge_index[0, i] = i
                edge_index[1, i] = i
            return edge_index

        adj = kneighbors_graph(features.numpy(), k, mode='connectivity', include_self=False)
        edge_index = torch.tensor(np.array(adj.nonzero()), dtype=torch.long)
        return edge_index
    except Exception as e:
        print(f"Error creating KNN edges: {str(e)}")
        return torch.empty((2, 0), dtype=torch.long)
import torch
import torch.nn as nn
from torch_geometric.data import Data

class EdgePredictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc_transform = nn.Linear(dim, 358)
        self.model = nn.Sequential(
            nn.Linear(2 * 358, 358),
            nn.ReLU(),
            nn.Linear(358, 1),
            nn.Sigmoid()
        )

    def forward(self, xi, xj):
        # 将 xi 和 xj 映射到 358 维
        xi = self.fc_transform(xi)
        xj = self.fc_transform(xj)

        # 如果批量大小不一致，则扩展为所有组合
        if xi.size(0) != xj.size(0):
            # 扩展 xi：重复 xi 每一行，与 xj 的每一行组合
            xi_ext = xi.unsqueeze(1).repeat(1, xj.size(0), 1)   # 形状: (xi_batch, xj_batch, 358)
            xi_ext = xi_ext.view(-1, xi_ext.size(-1))           # 展平成二维张量: (xi_batch * xj_batch, 358)
            xi_ext = xi_ext.view(-1, xi_ext.size(-1))           # 展平成二维张量: (xi_batch * xj_batch, 358)
            # 扩展 xj：重复 xj，每一行与 xi 的每一行组合
            xj_ext = xj.unsqueeze(0).repeat(xi.size(0), 1, 1)   # 形状: (xi_batch, xj_batch, 358)
            xj_ext = xj_ext.view(-1, xj_ext.size(-1))           # 展平成二维张量: (xi_batch * xj_batch, 358)
            # 拼接扩展后的特征
            x_pair = torch.cat([xi_ext, xj_ext], dim=-1)        # 形状: (xi_batch * xj_batch, 716)
        else:
            # 如果批量大小相同，则逐行拼接（假设一一对应）
            x_pair = torch.cat([xi, xj], dim=-1)

        # 输入拼接特征对，输出边存在的概率
        return self.model(x_pair)



def connect_generated_nodes(original_data, generated_x, edge_predictor, device, threshold=0.5):
    original_x = original_data.x.to(device)
    generated_x = generated_x.to(device)
    edge_index = original_data.edge_index.t().tolist()  # ✅ 正确格式为 [[src, dst], [src, dst], ...]


    new_node_start_idx = original_x.size(0)
    num_new_nodes = generated_x.size(0)

    edge_predictor.eval()
    new_edges = []

    # 生成新节点之间的边
    with torch.no_grad():
        for i_new in range(num_new_nodes):
            # 扩展 xi，使其与 xj 的维度匹配
            xi = generated_x[i_new].unsqueeze(0).repeat(num_new_nodes, 1)  # (num_new_nodes, 358)
            xj = generated_x  # (num_new_nodes, 358)

            # 打印 xi 和 xj 的形状，调试用
            print(f"xi shape (new node): {xi.shape}")
            print(f"xj shape (new nodes): {xj.shape}")

            # 计算新节点之间的边的概率
            prob = edge_predictor(xi, xj).squeeze()  # 输出概率
            selected = (prob > threshold).nonzero(as_tuple=False).view(-1)

            for idx in selected:
                new_idx = new_node_start_idx + i_new
                new_edges.append([new_idx, idx.item() + new_node_start_idx])  # 新节点间连接
                new_edges.append([idx.item() + new_node_start_idx, new_idx])

    # 现有节点和新节点之间的边
    with torch.no_grad():
        for i_new in range(num_new_nodes):
            xi = generated_x[i_new].unsqueeze(0).repeat(original_x.size(0), 1)  # 每个新节点和所有原节点进行对比
            xj = original_x  # 新节点和原节点的连接

            # 打印 xi 和 xj 的形状，调试用
            print(f"xi shape (new node to original): {xi.shape}")
            print(f"xj shape (original nodes): {xj.shape}")

            prob = edge_predictor(xi, xj).squeeze()
            selected = (prob > threshold).nonzero(as_tuple=False).view(-1)

            for idx in selected:
                i_old = idx.item()
                new_idx = new_node_start_idx + i_new
                new_edges.append([i_old, new_idx])
                new_edges.append([new_idx, i_old])

    all_x = torch.cat([original_x.cpu(), generated_x.cpu()], dim=0)
    all_y = torch.cat([original_data.y.cpu(), torch.ones(num_new_nodes, dtype=torch.long)], dim=0)

    edge_index.extend(new_edges)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=all_x, edge_index=edge_index, y=all_y)


import torch
from torch.nn.functional import cosine_similarity
from torch_geometric.data import Data


def enhanced_connect_generated_nodes_with_topk(
        original_data, generated_x, edge_predictor, device,
        sim_threshold=0.6,
        dist_threshold=1.5,
        top_k=5
):
    # Move data to device
    original_x = original_data.x.to(device)
    generated_x = generated_x.to(device)

    # Ensure edge_index is a list of [src, dst] pairs
    edge_index = original_data.edge_index.t().tolist()  # Transpose to shape (E, 2), then convert to list

    new_node_start_idx = original_x.size(0)
    num_new_nodes = generated_x.size(0)
    edge_predictor.eval()

    # Prepare combined feature and label tensors
    all_x = torch.cat([original_x.cpu(), generated_x.cpu()], dim=0)
    all_y = torch.cat([original_data.y.cpu(), torch.ones(num_new_nodes, dtype=torch.long)], dim=0)

    # Connect each new node to original nodes based on multiple criteria
    with torch.no_grad():
        for i_new in range(num_new_nodes):
            # Feature of the current new node
            x_new = generated_x[i_new].unsqueeze(0).repeat(original_x.size(0), 1)  # shape: (N_orig, D)
            X_orig = original_x  # shape: (N_orig, D)

            # 1. Edge predictor scores for new node vs all original nodes
            pred_scores = edge_predictor(x_new, X_orig).squeeze()  # shape: (N_orig,)
            # 2. Cosine similarity for new node vs all original nodes
            cos_sim = cosine_similarity(x_new, X_orig)  # shape: (N_orig,)
            # 3. L2 distance for new node vs all original nodes
            dist = torch.norm(x_new - X_orig, dim=1)  # shape: (N_orig,)

            # 4. Apply threshold conditions
            condition_mask = (pred_scores > sim_threshold) & (cos_sim > 0.6) & (dist < dist_threshold)
            selected_idx = torch.nonzero(condition_mask).view(
                -1).tolist()  # indices of original nodes meeting all conditions

            # 5. Guarantee at least top_k connections by predictor score
            topk_idx = torch.topk(pred_scores, k=min(top_k, pred_scores.size(0))).indices.tolist()
            final_idx = set(selected_idx) | set(topk_idx)  # union of condition-met and top-K indices

            # 6. Add edges for this new node
            new_idx = new_node_start_idx + i_new  # absolute index of the new node in the combined graph
            for idx in final_idx:
                if 0 <= idx < original_x.size(0):  # valid original node index
                    edge_index.append([idx, new_idx])
                    edge_index.append([new_idx, idx])

    # Connect new nodes among themselves based on distance threshold
    with torch.no_grad():
        for i in range(num_new_nodes):
            for j in range(i + 1, num_new_nodes):
                # If the feature distance between new node i and j is less than threshold, connect them
                if torch.norm(generated_x[i] - generated_x[j]) < dist_threshold:
                    i_idx = new_node_start_idx + i
                    j_idx = new_node_start_idx + j
                    edge_index.append([i_idx, j_idx])
                    edge_index.append([j_idx, i_idx])

    # (Optional) Safety check to ensure all edges are in pair format
    for edge in edge_index:
        assert isinstance(edge, (list, tuple)) and len(edge) == 2, f"Invalid edge: {edge}"

    # Convert edge_index back to tensor [2, E] format for PyG Data
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Return the new graph data with combined nodes and edges
    return Data(x=all_x, edge_index=edge_index, y=all_y)







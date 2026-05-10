# 文件名: dual_attention_fusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class SemanticGATEncoder(nn.Module):
    """
    语义流图注意力编码器 (Semantic Stream Encoder)

    作用：
    接收 ESM-2 提取的高维进化语义特征 (1280维)，并在 3D 物理空间构建的
    K-NN 拓扑图上进行消息传递。
    这相当于强制模型去关注“在三维物理空间中相邻的氨基酸，它们在进化语义上有什么关联”。
    """

    def __init__(self, in_dim=1280, hidden_dim=128, heads=4, dropout=0.1):
        super().__init__()
        # 第一层 GAT: 多头注意力，输出维度自动 concat
        # hidden_dim // heads 确保 concat 后的总维度恰好是 hidden_dim
        assert hidden_dim % heads == 0, "hidden_dim 必须能被 heads 整除"
        self.gat1 = GATConv(
            in_channels=in_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            dropout=dropout,
            concat=True
        )

        # 第二层 GAT: 聚合特征，输出最终的 hidden_dim
        self.gat2 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=1,
            dropout=dropout,
            concat=False
        )

    def forward(self, x, edge_index):
        """
        Args:
            x: [Total_Nodes, 1280] ESM 语义特征
            edge_index: [2, Total_Edges] 物理空间 K-NN 图的边
        Returns:
            out: [Total_Nodes, 128] 聚合后的语义特征
        """
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        out = self.gat2(x, edge_index)
        # 不在这里加非线性激活，保持和 GVP 输出的线性空间一致，方便后续 Attention 融合
        return out


class Attention(nn.Module):

    def __init__(self, hidden_dim=128):
        super().__init__()

        # 共享打分网络 (Shared Scoring Network)
        # q^T * tanh(W * h + b)
        # 因为我们把两个模态 stack 在了一起，所以只需定义一个网络
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # W 和 b
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)  # q (查询向量，没有 bias)
        )

    def forward(self, h_spatial, h_semantic):
        """
        Args:
            h_spatial: [Total_Nodes, hidden_dim] (来自 GVP 的空间特征)
            h_semantic: [Total_Nodes, hidden_dim] (来自 GAT 的语义特征)
        Returns:
            z_fused: [Total_Nodes, hidden_dim] 融合后的特征
            alphas: [Total_Nodes, 2] 动态注意力权重 (用于分析和可视化)
        """
        # 1. 模态维度堆叠 (The Stacking Trick)
        # 将两个特征在维度 1 上拼接
        # h_stacked shape: [Total_Nodes, 2, hidden_dim]
        h_stacked = torch.stack([h_spatial, h_semantic], dim=1)

        # 2. 计算注意力原始得分 (Logits)
        # 打分网络一视同仁地对所有模态进行打分
        # w shape: [Total_Nodes, 2, 1]
        w = self.project(h_stacked)

        # 3. 归一化注意力权重
        # 在模态维度 (dim=1) 上做 Softmax，确保对于每个节点，alpha_spatial + alpha_semantic = 1
        # alphas shape: [Total_Nodes, 2, 1]
        alphas = torch.softmax(w, dim=1)

        # 4. 加权求和融合
        # 广播机制会将 alphas [N, 2, 1] 乘到 h_stacked [N, 2, D] 上
        # 然后在模态维度求和，得到最终特征
        # z_fused shape: [Total_Nodes, hidden_dim]
        z_fused = (alphas * h_stacked).sum(dim=1)

        # squeeze 掉最后维的 1，方便外部调用和分析
        # alphas_out shape: [Total_Nodes, 2]
        # alphas_out[:, 0] 是空间权重，alphas_out[:, 1] 是语义权重
        alphas_out = alphas.squeeze(-1)

        return z_fused, alphas_out
import torch
import torch.nn as nn
from torch_geometric.data import Batch

# 导入你原有的子模块
from .esm_wrapper import ESMWrapper
from .gvp_encoder import GVPGraphEncoder


from .dual_attention_fusion import SemanticGATEncoder, Attention

from torch_geometric.nn import knn_graph


class SpaSem(nn.Module):
    """
    Struct-MIF: Structure-aware Masked Inverse Folding (Dual-Stream Version)
    """

    def __init__(
            self,
            esm_model_name: str = "facebook/esm2_t33_650M_UR50D",
            gvp_node_out_dim: int = 128,  # GVP 和 GAT 的统一输出隐藏层维度
            gvp_layers: int = 6,  # GVP 层数
            dropout: float = 0.1,
            use_dual_stream: bool = True
    ):
        super().__init__()
        self.use_dual_stream = use_dual_stream

        # ============================================================
        # 1. 语义塔 (Semantic Tower) - Frozen
        # ============================================================
        self.esm_encoder = ESMWrapper(esm_model_name)
        self.esm_hidden_dim = self.esm_encoder.hidden_size  # 通常为 1280

        # ============================================================
        # 2. 空间流 (Spatial Stream) - GVP 几何引擎
        # ============================================================
        # 因为移除了 3Di 特征，我们需要将 1280 维的 ESM 降维，
        # 作为 GVP 的初始标量输入 (GVP 必须要有标量特征才能正常进行消息传递)
        self.spatial_scalar_proj = nn.Sequential(
            nn.Linear(self.esm_hidden_dim, gvp_node_out_dim),
            nn.LayerNorm(gvp_node_out_dim),
            nn.GELU()
        )

        self.gvp_encoder = GVPGraphEncoder(
            node_in_dim=gvp_node_out_dim,  # 降维后的输入维度: 128
            node_out_dim=gvp_node_out_dim,  # 输出维度: 128
            n_layers=gvp_layers,
            dropout=dropout
        )

        # ============================================================
        # 3. 语义流与融合层 (Semantic Stream & Fusion)
        # ============================================================
        if self.use_dual_stream:
            # GAT 通道：直接处理 1280 维特征，输出 128 维
            self.gat_encoder = SemanticGATEncoder(
                in_dim=self.esm_hidden_dim,
                hidden_dim=gvp_node_out_dim,
                dropout=dropout
            )

            # 注意力融合：接收两股 128 维特征，共享权重打分
            self.fusion_attention = Attention(
                hidden_dim=gvp_node_out_dim
            )

        # 用于存储最近一次 forward 的注意力权重，方便推理时画图分析
        self.current_alphas = None

        # ============================================================
        # 4. 预测头 (Prediction Head)
        # ============================================================
        # 将融合后的 128 维特征投影回 ESM 词表空间 (33 tokens)
        self.head = nn.Sequential(
            nn.LayerNorm(gvp_node_out_dim),
            nn.Linear(gvp_node_out_dim, self.esm_hidden_dim),
            nn.GELU(),
            nn.Linear(self.esm_hidden_dim, 33)
        )

    def forward(self, batch: Batch):
        """
        Args:
            batch: PyG Batch 对象，包含:
                - input_ids: [Batch, Max_Seq_Len] (ESM 输入)
                - esm_attention_mask: [Batch, Max_Seq_Len]
                - attention_mask: [Batch, Max_Seq_Len] (对齐 Mask: 1 for AA, 0 for CLS/EOS/Pad)
                - node_vectors: [Total_Nodes, 3, 3]
                - edge_index: [2, Total_Edges]
                - edge_vectors: [Total_Edges, 1, 3]
        """

        # ------------------------------------------------------------
        # Step 1: 获取并对齐 ESM 语义特征
        # ------------------------------------------------------------
        esm_out = self.esm_encoder(batch.input_ids, batch.esm_attention_mask)
        valid_mask = batch.attention_mask.bool()

        # flat_esm_feats: [Total_Nodes, 1280]
        flat_esm_feats = esm_out[valid_mask]

        # [安全检查] 确保提取出的 ESM 节点数与 GVP 图节点数一致
        # (因为移除了 x_3di，这里改为检查 node_vectors 的第一维)
        if flat_esm_feats.shape[0] != batch.node_vectors.shape[0]:
            raise ValueError(
                f"Dimension Mismatch! ESM features ({flat_esm_feats.shape[0]}) "
                f"!= Graph nodes ({batch.node_vectors.shape[0]}).\n"
                "Check 'collator.py' logic regarding graph_align_mask."
            )

        # ------------------------------------------------------------
        # Step 2: 空间流计算 (GVP)
        # ------------------------------------------------------------
        # 准备 GVP 的基础标量特征 [Total_Nodes, 128]
        gvp_scalar_input = self.spatial_scalar_proj(flat_esm_feats)

        # 在真实的 3D 物理空间上进行几何消息传递
        # h_spatial: [Total_Nodes, 128]
        h_spatial = self.gvp_encoder(
            h_scalar=gvp_scalar_input,
            h_vector=batch.node_vectors,
            edge_index=batch.edge_index,
            edge_vector=batch.edge_vectors
        )

        # ------------------------------------------------------------
        # Step 3: 分流与跨模态注意力融合
        # ------------------------------------------------------------
        if self.use_dual_stream:
            # 将 1280 维的 ESM 特征看作高维空间中的坐标
            # 动态寻找每个残基在“语义/进化特征空间”中最相似的 15 个邻居
            # batch.batch 是 PyG 自动生成的索引，用来防止它把不同蛋白质的残基连在一起
            semantic_edge_index = knn_graph(
                x=flat_esm_feats,
                k=15,
                batch=batch.batch,
                loop=True,
                cosine=True  # 1280 维高维特征强烈建议使用余弦相似度而不是欧氏距离
            )

            # 激活语义流 (GAT)：在实时构建的特征图 (semantic_edge_index) 上处理高维语义
            # 注意这里传入的是刚刚算出来的 semantic_edge_index，而不是物理的 edge_index
            h_semantic = self.gat_encoder(flat_esm_feats, semantic_edge_index)

            # 动态注意力融合
            z_fused, alphas = self.fusion_attention(h_spatial, h_semantic)

            # 保存权重
            self.current_alphas = alphas
        else:
            # 退化为单流 Baseline
            z_fused = h_spatial
            self.current_alphas = None

        # ------------------------------------------------------------
        # Step 4: 预测 Logits
        # ------------------------------------------------------------
        # logits: [Total_Nodes, 33]
        logits = self.head(z_fused)

        return logits
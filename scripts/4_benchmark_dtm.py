import os
import sys
import glob
import argparse
import torch
import pandas as pd
import logging
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoTokenizer

# 添加项目根目录
sys.path.append(os.getcwd())

# 尝试导入
try:
    from src.modeling.SpaSem import SpaSem
    from src.data.graph_builder import GraphBuilder
    from src.scoring import DMSBatchScorer
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    try:
        from src.modeling.SpaSem import SpaSem
        from src.data.graph_builder import GraphBuilder
        from src.scoring import DMSBatchScorer
    except ImportError:
        # 手动兜底 Scorer
        try:
            from src.modeling.SpaSem import SpaSem
            from src.data.graph_builder import GraphBuilder
        except ImportError as e3:
            print(f"❌ 严重错误: {e3}")
            sys.exit(1)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ==========================================
# 简单的 Scorer (兜底用)
# ==========================================
class SimpleScorer:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def score_mutations(self, data, mutations):
        data = data.to(self.device)
        with torch.no_grad():
            logits = self.model(data)  # [1, L, V] 或者 [Total_Nodes, V]
            # 兼容模型输出
            if logits.dim() == 2:
                logits = logits.unsqueeze(0)

            logits = torch.log_softmax(logits, dim=-1)

            preds = []
            seq = data.seq_str

            offset = 0
            if logits.shape[1] == len(seq) + 2:
                offset = 1

            for mut in mutations:
                try:
                    wt = mut[0]
                    pos = int(mut[1:-1]) - 1
                    mt = mut[-1]

                    if pos < 0 or pos >= len(seq):
                        preds.append(None)
                        continue
                    if seq[pos] != wt:
                        preds.append(None)
                        continue

                    wt_idx = self.tokenizer.convert_tokens_to_ids(wt)
                    mt_idx = self.tokenizer.convert_tokens_to_ids(mt)

                    idx = pos + offset
                    if idx >= logits.shape[1]:
                        preds.append(None)
                        continue

                    score = (logits[0, idx, mt_idx] - logits[0, idx, wt_idx]).item()
                    preds.append(score)
                except:
                    preds.append(None)
            return preds


# ==========================================
# 🔥 核心预处理 (Dual-Stream 版)
# ==========================================
def preprocess_graph(data, tokenizer):
    """
    更新说明：移除了 3Di (x_3di) 的逻辑，对齐了 dataset.py 的处理方式
    """
    if hasattr(data, 'seq_str') and data.seq_str:
        seq = data.seq_str
    elif hasattr(data, 'seq') and data.seq:
        seq = data.seq
    else:
        return None

    # 1. Tokenize
    token_out = tokenizer(seq, return_tensors="pt", padding=False, truncation=False, add_special_tokens=True)
    data.input_ids = token_out["input_ids"].squeeze(0)

    # 2. Masks
    data.esm_attention_mask = torch.ones_like(data.input_ids)
    align_mask = torch.zeros_like(data.input_ids)
    if len(align_mask) > 2:
        align_mask[1:-1] = 1
    data.attention_mask = align_mask

    # 兼容旧逻辑
    data.graph_align_mask = align_mask.bool()

    # 3. Edge Vectors (GVP 必需)
    if hasattr(data, 'edge_index') and hasattr(data, 'pos'):
        row, col = data.edge_index
        vec = data.pos[col] - data.pos[row]
        data.edge_vectors = vec.unsqueeze(1)
    else:
        num_edges = data.edge_index.shape[1] if hasattr(data, 'edge_index') else 0
        data.edge_vectors = torch.zeros(num_edges, 1, 3)

    # 4. Node Vectors (GVP 必需)
    if hasattr(data, 'node_vectors'):
        # 补全叉乘方向向量 (如果只有 N->CA 和 CA->C 两个向量的话)
        if data.node_vectors.shape[1] == 2:
            n_vec = data.node_vectors[:, 0, :]
            c_vec = data.node_vectors[:, 1, :]
            cross_vec = torch.cross(n_vec, c_vec, dim=1)
            data.node_vectors = torch.cat([data.node_vectors, cross_vec.unsqueeze(1)], dim=1)

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtm_root", type=str, default="data/DTM/DATASET")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--esm_model", type=str, default="./pretrained_models/esm2_t33_650M_UR50D")
    parser.add_argument("--output_csv", type=str, default="dtm_benchmark_results.csv")
    parser.add_argument("--mut_col", type=str, default="mutant")
    parser.add_argument("--score_col", type=str, default="score")

    # [新增] 模型架构超参数
    parser.add_argument("--gvp_layers", type=int, default=6)
    parser.add_argument("--gvp_node_out_dim", type=int, default=128)
    parser.add_argument("--disable_dual_stream", action="store_true", help="如果是测试消融实验，请加上此参数")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型 (匹配新版架构)
    logger.info("📦 Loading Model (SpatialGlue Architecture)...")
    use_dual = not args.disable_dual_stream
    model = SpaSem(
        esm_model_name=args.esm_model,
        gvp_node_out_dim=args.gvp_node_out_dim,
        gvp_layers=args.gvp_layers,
        use_dual_stream=use_dual
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

    # 清理 DataParallel 的 'module.' 前缀
    new_sd = {k.replace('module.', ''): v for k, v in state_dict.items()}

    try:
        model.load_state_dict(new_sd, strict=True)
        logger.info("✅ 权重加载成功！")
    except RuntimeError as e:
        logger.warning(f"⚠️ Strict Load 失败，尝试 loose load: {e}")
        model.load_state_dict(new_sd, strict=False)

    model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.esm_model)

    # [更新] GraphBuilder 初始化不再需要 foldseek_bin
    graph_builder = GraphBuilder(spatial_k=30, semantic_k=15)

    try:
        scorer = DMSBatchScorer(model, tokenizer, batch_size=32, device=device)
        logger.info("✅ 使用原生 DMSBatchScorer")
    except NameError:
        scorer = SimpleScorer(model, tokenizer, device)
        logger.info("⚠️ 使用简化版 Scorer")

    # 2. 遍历数据集
    subdirs = sorted(glob.glob(os.path.join(args.dtm_root, "*")))
    logger.info(f"🔍 发现 {len(subdirs)} 个数据集")

    results = []

    for subdir in tqdm(subdirs):
        name = os.path.basename(subdir)
        pdb = os.path.join(subdir, f"{name}.ef.pdb")
        tsv = os.path.join(subdir, f"{name}.tsv")
        fasta = os.path.join(subdir, f"{name}.fasta")

        if not os.path.exists(pdb) or not os.path.exists(tsv):
            continue

        try:
            wt_seq = ""
            if os.path.exists(fasta):
                with open(fasta) as f:
                    wt_seq = "".join([l.strip() for l in f if not l.startswith(">")])

            try:
                data = graph_builder.process(pdb)
            except Exception as e:
                print(f"❌ [{name}] GraphBuilder 崩溃: {e}")
                continue

            if data is None:
                print(f"❌ [{name}] GraphBuilder 返回 None")
                continue

            if wt_seq:
                data.seq = wt_seq
                data.seq_str = wt_seq
            elif hasattr(data, 'seq'):
                wt_seq = data.seq

            # 🔥 调用修复后的预处理
            data = preprocess_graph(data, tokenizer)

            if data is None:
                print(f"❌ [{name}] 预处理失败 (缺序列)")
                continue

            df = pd.read_csv(tsv, sep='\t')
            m_col, s_col = args.mut_col, args.score_col
            if m_col not in df.columns:
                cand = [c for c in df.columns if 'mut' in c.lower()]
                if cand: m_col = cand[0]
            if s_col not in df.columns:
                cand = [c for c in df.columns if 'score' in c.lower()]
                if cand: s_col = cand[0]

            if m_col not in df.columns or s_col not in df.columns:
                print(f"❌ [{name}] 列名不对: 找 {m_col}/{s_col}, 实际 {list(df.columns)}")
                continue

            muts = df[m_col].tolist()
            labels = df[s_col].tolist()

            preds = scorer.score_mutations(data, muts)

            clean_preds = []
            clean_labels = []
            for p, l in zip(preds, labels):
                if p is not None and not pd.isna(p):
                    clean_preds.append(p)
                    clean_labels.append(l)

            if len(clean_preds) > 5:
                rho, _ = spearmanr(clean_preds, clean_labels)
                results.append({"Dataset": name, "Spearman": rho, "N": len(clean_preds)})
            else:
                print(f"⚠️ [{name}] 有效预测太少 ({len(clean_preds)})")

        except Exception as e:
            print(f"❌ [{name}] 未知错误: {e}")

    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(args.output_csv, index=False)
        print(f"\n✅ 平均 Spearman: {res_df['Spearman'].mean():.4f}")
    else:
        print("\n💀 没有有效结果。")


if __name__ == "__main__":
    main()
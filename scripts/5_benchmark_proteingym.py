import os
import sys
import argparse
import json
import torch
import pandas as pd
import logging
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoTokenizer
from torch_geometric.data import Batch

# 添加项目根目录
sys.path.append(os.getcwd())

# 尝试导入核心模块
try:
    from src.modeling.SpaSem import SpaSem
    from src.data.graph_builder import GraphBuilder
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    try:
        from src.modeling.SpaSem import SpaSem
        from src.data.graph_builder import GraphBuilder
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        sys.exit(1)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


# ==========================================
# 🔥 核心打分器 (极速版 + 多点突变支持)
# ==========================================
class DMSBatchScorer:
    def __init__(self, model, tokenizer, batch_size, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size

    def score_mutations(self, data, mutations, dataset_id="Unknown"):
        # 1. 获取序列
        if hasattr(data, 'seq_str') and data.seq_str:
            seq_str = data.seq_str
        elif hasattr(data, 'seq') and data.seq:
            seq_str = data.seq
        else:
            return [None] * len(mutations)

        seq_len = len(seq_str)
        preds = []

        # 2. 创建单样本 Batch (为了复用模型 forward 逻辑)
        batch = Batch.from_data_list([data]).to(self.device)

        # 维度自动修复 (确保是 [1, L] 而不是 [L])
        if batch.input_ids.ndim == 1:
            batch.input_ids = batch.input_ids.reshape(1, -1)
        if hasattr(batch, 'esm_attention_mask') and batch.esm_attention_mask is not None:
            if batch.esm_attention_mask.ndim == 1:
                batch.esm_attention_mask = batch.esm_attention_mask.reshape(1, -1)
        if hasattr(batch, 'attention_mask') and batch.attention_mask is not None:
            if batch.attention_mask.ndim == 1:
                batch.attention_mask = batch.attention_mask.reshape(1, -1)
        if hasattr(batch, 'graph_align_mask') and batch.graph_align_mask is not None:
            if batch.graph_align_mask.ndim == 1:
                batch.graph_align_mask = batch.graph_align_mask.reshape(1, -1)

        # 3. 预计算野生型 Logits (极速版核心：只跑一次模型！)
        with torch.no_grad():
            wt_logits = self.model(batch)
            wt_logits = torch.log_softmax(wt_logits, dim=-1)

        # 智能判断输出维度类型 (图模式 vs 序列模式)
        is_graph_output = (wt_logits.ndim == 2)

        # 计算 offset (处理 CLS token)
        if is_graph_output:
            offset = 0
            if wt_logits.shape[0] == seq_len + 2: offset = 1
        else:
            offset = 1 if wt_logits.shape[1] == seq_len + 2 else 0

        # 4. 极速查表推理
        for mut_str in tqdm(mutations, desc=f"Scoring {dataset_id}", leave=False):
            try:
                # 解析突变 (支持 'A123T:G456C')
                if ":" in mut_str:
                    sub_parts = mut_str.split(":")
                elif "," in mut_str:
                    sub_parts = mut_str.split(",")
                else:
                    sub_parts = [mut_str]

                score = 0.0
                valid_mut = True

                for p in sub_parts:
                    p = p.strip()
                    if not p: continue
                    try:
                        wt, pos_str, mt = p[0], p[1:-1], p[-1]
                        pos = int(pos_str) - 1
                    except:
                        valid_mut = False
                        break

                    # 边界检查
                    if pos < 0 or pos >= seq_len:
                        valid_mut = False
                        break

                    wt_id = self.tokenizer.convert_tokens_to_ids(wt)
                    mt_id = self.tokenizer.convert_tokens_to_ids(mt)

                    # 查表计算 LLR (Log-Likelihood Ratio)
                    idx = pos + offset
                    if is_graph_output:
                        val = (wt_logits[idx, mt_id] - wt_logits[idx, wt_id]).item()
                    else:
                        val = (wt_logits[0, idx, mt_id] - wt_logits[0, idx, wt_id]).item()

                    score += val

                if valid_mut:
                    preds.append(score)
                else:
                    preds.append(None)

            except Exception:
                preds.append(None)

        return preds


# ==========================================
# 🛠️ 辅助函数
# ==========================================
def get_seq_from_pdb(pdb_path):
    """从 PDB 读取序列 (兜底方案)"""
    try:
        three_to_one = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
                        'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
                        'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
        seq = []
        with open(pdb_path, 'r') as f:
            last_res_id = None
            for line in f:
                if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                    res_name = line[17:20].strip()
                    res_id = line[22:27].strip()
                    if res_id != last_res_id:
                        seq.append(three_to_one.get(res_name, 'X'))
                        last_res_id = res_id
        return "".join(seq)
    except Exception:
        return None


def preprocess_graph(data, tokenizer, pdb_path=None):
    """标准预处理流程 (移除 3Di，适配 SpatialGlue)"""
    seq = None
    if hasattr(data, 'seq_str') and data.seq_str:
        seq = data.seq_str
    elif hasattr(data, 'seq') and data.seq:
        seq = data.seq

    # 兜底：从 PDB 读序列
    if not seq and pdb_path and os.path.exists(pdb_path):
        seq = get_seq_from_pdb(pdb_path)
        data.seq_str = seq
        data.seq = seq

    if not seq: return None

    token_out = tokenizer(seq, return_tensors="pt", padding=False, truncation=False, add_special_tokens=True)
    data.input_ids = token_out["input_ids"].squeeze(0)
    mask = token_out["attention_mask"].squeeze(0)
    data.esm_attention_mask = mask

    # Graph Align Mask (去掉 CLS/EOS)
    align_mask = mask.clone()
    align_mask[0] = 0
    align_mask[-1] = 0
    data.attention_mask = align_mask
    data.graph_align_mask = align_mask.clone()

    # 向量特征补全
    if hasattr(data, 'node_vectors'):
        if data.node_vectors.shape[1] == 2:
            n_vec = data.node_vectors[:, 0, :]
            c_vec = data.node_vectors[:, 1, :]
            cross_vec = torch.cross(n_vec, c_vec, dim=1)
            data.node_vectors = torch.cat([data.node_vectors, cross_vec.unsqueeze(1)], dim=1)

    # 边向量补全
    if hasattr(data, 'edge_index') and hasattr(data, 'pos'):
        try:
            row, col = data.edge_index
            vec = data.pos[col] - data.pos[row]
            data.edge_vectors = vec.unsqueeze(1)
        except:
            num = data.edge_index.shape[1] if len(data.edge_index.shape) > 1 else 0
            data.edge_vectors = torch.zeros(num, 1, 3)
    else:
        num = data.edge_index.shape[1] if hasattr(data, 'edge_index') else 0
        data.edge_vectors = torch.zeros(num, 1, 3)

    return data


def main():
    parser = argparse.ArgumentParser(description="Run ProteinGym Benchmark")
    parser.add_argument("--json_file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="proteingym_results.csv")
    parser.add_argument("--esm_model", type=str, default="./pretrained_models/esm2_t33_650M_UR50D")

    # 兼容旧参数防止报错
    parser.add_argument("--foldseek_bin", type=str, default="", help="[Deprecated]")

    # 模型架构参数
    parser.add_argument("--gvp_layers", type=int, default=6)
    parser.add_argument("--gvp_node_out_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--category", type=str, default="all")
    parser.add_argument("--disable_dual_stream", action="store_true", help="禁用双流(用于消融实验)")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_dual = not args.disable_dual_stream
    logger.info(f"📦 Loading SpaSem Model (SpatialGlue Dual-Stream: {use_dual})...")

    # 初始化最新的模型
    try:
        model = SpaSem(
            esm_model_name=args.esm_model,
            gvp_node_out_dim=args.gvp_node_out_dim,
            gvp_layers=args.gvp_layers,
            use_dual_stream=use_dual
        )
    except NameError:
        logger.error("❌ Failed to import SpaSem.")
        return

    # 加载权重并清理前缀
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

    new_sd = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        # 清理旧模型的遗留配置，防止干扰
        if name.startswith("fusion_norm") or name.startswith("fusion_proj"):
            continue
        new_sd[name] = v

    try:
        model.load_state_dict(new_sd, strict=True)
    except:
        logger.warning("⚠️ Strict load failed, trying non-strict...")
        model.load_state_dict(new_sd, strict=False)

    model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.esm_model)
    # 使用最新的建图逻辑
    graph_builder = GraphBuilder(spatial_k=30, semantic_k=15)

    scorer = DMSBatchScorer(model, tokenizer, args.batch_size, device)

    # ===============================================
    # 🚀 完美复刻你的 JSON 解析逻辑
    # ===============================================
    if not os.path.exists(args.json_file):
        logger.error(f"JSON file not found: {args.json_file}")
        return

    with open(args.json_file, 'r') as f:
        json_data = json.load(f)

    all_tasks = []
    if isinstance(json_data, list):
        all_tasks = json_data
    elif isinstance(json_data, dict):
        for cat, items in json_data.items():
            for item in items:
                if 'category' not in item: item['category'] = cat
                all_tasks.append(item)

    # 筛选任务
    tasks = []
    target_category = args.category.lower().strip()
    if target_category == "all":
        tasks = all_tasks
    else:
        for task in all_tasks:
            task_cat = task.get('category', '').lower()
            if target_category in task_cat:
                tasks.append(task)

    logger.info(f"📋 Loaded {len(tasks)} tasks (Category filter: {args.category}).")
    results = []

    # ===============================================
    # 循环执行打分
    # ===============================================
    for task in tqdm(tasks, desc=f"Benchmarking ({args.category})", leave=True):
        dataset_id = task.get('id', 'unknown')
        pdb_path = task.get('pdb_path')
        dms_path = task.get('dms_path')
        category = task.get('category', 'Uncategorized')

        if not os.path.exists(pdb_path) or not os.path.exists(dms_path): continue

        try:
            if device.type == 'cuda': torch.cuda.empty_cache()

            # 构建图
            try:
                data = graph_builder.process(pdb_path)
            except:
                data = None
            if not data: continue

            # 预处理
            data = preprocess_graph(data, tokenizer, pdb_path)
            if not data: continue

            # 读取 DMS 数据
            try:
                df = pd.read_csv(dms_path)
            except:
                continue

            mut_col = next((c for c in df.columns if 'mut' in c.lower()), None)
            score_col = next((c for c in df.columns if 'score' in c.lower()), None)

            if not mut_col or not score_col: continue

            muts = df[mut_col].tolist()
            labels = df[score_col].tolist()

            # 🔥 推理
            preds = scorer.score_mutations(data, muts, dataset_id)

            # 结果清洗
            clean_preds, clean_labels = [], []
            for p, l in zip(preds, labels):
                if p is not None and not pd.isna(p) and not pd.isna(l):
                    clean_preds.append(p)
                    clean_labels.append(l)

            if len(clean_preds) > 5:
                sp_res = spearmanr(clean_preds, clean_labels)
                try:
                    rho = sp_res[0]
                except:
                    rho = sp_res.statistic if hasattr(sp_res, 'statistic') else sp_res

                results.append({
                    "id": dataset_id,
                    "category": category,
                    "spearman": rho,
                    "N": len(clean_preds)
                })

        except Exception as e:
            logger.error(f"Error on {dataset_id}: {e}")

    # 汇总
    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(args.output_csv, index=False)

        print("\n" + "=" * 40)
        print(f"🏆 ProteinGym Results ({args.category})")
        print("=" * 40)
        summary = res_df.groupby("category")["spearman"].agg(['mean', 'count', 'std'])
        summary.columns = ['Avg Spearman', 'Count', 'Std']
        print(summary)
        print("-" * 40)
        print(f"Overall Average: {res_df['spearman'].mean():.4f}")
        print("=" * 40)
    else:
        logger.error("No results generated!")


if __name__ == "__main__":
    main()
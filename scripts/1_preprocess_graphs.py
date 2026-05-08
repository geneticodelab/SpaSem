import os
import glob
import multiprocessing
import logging
import torch
from tqdm import tqdm
from src.data.graph_builder import GraphBuilder

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置路径
PDB_DIR = "./data/raw_pdb/dompdb"  # 替换为你的 PDB 文件夹路径
OUTPUT_DIR = "./data/processed_graphs/train"  # 替换为你的输出文件夹路径

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 实例化建图器 (无需 foldseek 路径)
# 默认设置: 空间图 30 个邻居，语义序列图 15 个邻居
builder = GraphBuilder(spatial_k=30, semantic_k=15)


def process_single_file(pdb_path):
    """处理单个 PDB 文件并保存为 .pt"""
    try:
        # 获取文件名 (不含扩展名)
        basename = os.path.basename(pdb_path)
        name = basename.split('.')[0]
        output_path = os.path.join(OUTPUT_DIR, f"{name}.pt")

        # 如果已经处理过，直接跳过
        if os.path.exists(output_path):
            return True

        # 构建双图 Data 对象
        data = builder.process(pdb_path)

        if data is not None:
            # 保存到硬盘
            torch.save(data, output_path)
            return True
        else:
            return False

    except Exception as e:
        logger.error(f"Error processing {pdb_path}: {e}")
        return False


def main():
    # 获取所有 pdb/pdb.gz 文件
    pdb_files = glob.glob(os.path.join(PDB_DIR, "*.pdb")) + \
                glob.glob(os.path.join(PDB_DIR, "*.pdb.gz")) + \
                glob.glob(os.path.join(PDB_DIR, "*.ent"))

    if not pdb_files:
        logger.warning(f"No PDB files found in {PDB_DIR}")
        return

    logger.info(f"Found {len(pdb_files)} PDB files. Starting processing...")

    # 使用多进程加速
    num_cores = max(1, multiprocessing.cpu_count() - 2)
    success_count = 0

    with multiprocessing.Pool(processes=num_cores) as pool:
        # 使用 list(tqdm(...)) 消耗迭代器以显示进度条
        results = list(tqdm(pool.imap(process_single_file, pdb_files), total=len(pdb_files), desc="Processing PDBs"))

        success_count = sum(1 for r in results if r)

    logger.info(f"Processing complete! Successfully built dual-graphs for {success_count}/{len(pdb_files)} structures.")


if __name__ == "__main__":
    main()
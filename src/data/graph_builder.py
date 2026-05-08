import torch
import numpy as np
import os
import tempfile
import shutil
import gzip
import warnings
from Bio.PDB import PDBParser
from Bio import BiopythonWarning
from scipy.spatial import cKDTree
from torch_geometric.data import Data

warnings.simplefilter('ignore', BiopythonWarning)


class InternalPDBFeatureExtractor:
    def __init__(self):
        self.parser = PDBParser(QUIET=True)
        self.aa_map = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
            'MSE': 'M', 'SEC': 'C'
        }

    def parse(self, pdb_path):
        temp_pdb = None
        parse_path = pdb_path

        try:
            if pdb_path.endswith('.gz'):
                fd, temp_pdb = tempfile.mkstemp(suffix=".pdb")
                os.close(fd)
                with gzip.open(pdb_path, 'rb') as f_in:
                    with open(temp_pdb, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                parse_path = temp_pdb

            structure = self.parser.get_structure('structure', parse_path)

            coords = []
            seq = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.id[0] != ' ': continue
                        resname = residue.get_resname().strip()
                        if resname not in self.aa_map: continue
                        if not all(atom in residue for atom in ['N', 'CA', 'C']): continue
                        try:
                            n = residue['N'].get_coord()
                            ca = residue['CA'].get_coord()
                            c = residue['C'].get_coord()
                            res_coords = np.stack([n, ca, c])
                            coords.append(res_coords)
                            seq.append(self.aa_map[resname])
                        except Exception:
                            continue
                break

            if len(coords) == 0: return None
            return {"coords": np.array(coords), "seq": "".join(seq)}

        except Exception:
            return None
        finally:
            if temp_pdb and os.path.exists(temp_pdb):
                os.remove(temp_pdb)


class GraphBuilder:
    def __init__(self, spatial_k=30, semantic_k=15):
        """
        Args:
            spatial_k: 空间图的 K-NN 邻居数 (基于 3D 坐标)
            semantic_k: 语义图的 K-NN 邻居数 (基于 1D 序列距离)
        """
        self.spatial_k = spatial_k
        self.semantic_k = semantic_k
        self.extractor = InternalPDBFeatureExtractor()

    def process(self, pdb_path):
        # 1. 解析 PDB
        pdb_data = self.extractor.parse(pdb_path)
        if pdb_data is None: return None

        coords = pdb_data['coords']  # [L, 3, 3]
        seq = pdb_data['seq']
        num_nodes = len(seq)

        # 序列太短没有建图意义
        if num_nodes < 5: return None

        try:
            ca_coords = coords[:, 1, :]

            # ==========================================
            # 图 A: 构建 3D 空间物理图 (Spatial Graph)
            # ==========================================
            tree_3d = cKDTree(ca_coords)
            # k 需要 +1 因为 query 会包含节点自身
            dists_3d, idxs_3d = tree_3d.query(ca_coords, k=min(self.spatial_k + 1, num_nodes))

            src_spatial, dst_spatial = [], []
            for i, neighbors in enumerate(idxs_3d):
                for n_idx in neighbors:
                    if i == n_idx: continue
                    src_spatial.append(i)
                    dst_spatial.append(n_idx)

            edge_index = torch.tensor([src_spatial, dst_spatial], dtype=torch.long)

            # ==========================================
            # 图 B: 构建 1D 序列语义图 (Semantic Graph)
            # ==========================================
            # 将序列索引 [0, 1, ..., N-1] 转换为 1D 坐标
            seq_indices = np.arange(num_nodes).reshape(-1, 1)
            tree_1d = cKDTree(seq_indices)
            dists_1d, idxs_1d = tree_1d.query(seq_indices, k=min(self.semantic_k + 1, num_nodes))

            src_semantic, dst_semantic = [], []
            for i, neighbors in enumerate(idxs_1d):
                for n_idx in neighbors:
                    if i == n_idx: continue
                    src_semantic.append(i)
                    dst_semantic.append(n_idx)

            semantic_edge_index = torch.tensor([src_semantic, dst_semantic], dtype=torch.long)

            # ==========================================
            # 提取节点向量和位置特征
            # ==========================================
            pos = torch.tensor(ca_coords, dtype=torch.float)

            node_coords_th = torch.tensor(coords, dtype=torch.float)
            n_vec = node_coords_th[:, 1] - node_coords_th[:, 0]
            c_vec = node_coords_th[:, 2] - node_coords_th[:, 1]
            node_vectors = torch.stack([n_vec, c_vec], dim=1)

            # 封装 Data 对象
            data = Data(
                num_nodes=num_nodes,
                edge_index=edge_index,  # 空间图
                semantic_edge_index=semantic_edge_index,  # 语义图
                node_vectors=node_vectors,
                pos=pos,
                seq=seq
            )
            return data

        except Exception:
            return None
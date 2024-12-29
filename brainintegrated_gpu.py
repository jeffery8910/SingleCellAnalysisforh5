import h5py
import pandas as pd
import scipy.sparse as sp
import numpy as np
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import cupy as cp
from sklearn.decomposition import PCA
from umap import UMAP

def load_h5_data(file_path):
    """載入 H5 數據文件"""
    try:
        print(f"正在讀取文件：{file_path}")
        adata = sc.read_10x_h5(file_path)
        print(f"成功載入數據，形狀：{adata.shape}")
        return adata
    except Exception as e:
        print(f"載入數據時出錯：{str(e)}")
        raise

def load_marker_genes_from_csv(csv_path):
    """從 CSV 文件讀取 marker genes"""
    try:
        print(f"正在讀取標記基因文件：{csv_path}")
        df = pd.read_csv(csv_path)
        marker_genes_dict = {}
        for cell_type in df['List'].unique():
            genes = df[df['List'] == cell_type]['Name'].tolist()
            marker_genes_dict[cell_type] = genes
            print(f"- {cell_type}: {len(genes)} 個標記基因")
        return marker_genes_dict
    except Exception as e:
        print(f"讀取標記基因文件時出錯：{str(e)}")
        raise

def analyze_cell_types(adata, marker_genes_dict, method='umap'):
    """使用 CuPy 進行 GPU 加速的細胞類型分析"""
    print("\n=== 開始 GPU 加速數據分析 ===")
    
    # 1. 預處理
    print("\n正在進行數據標準化...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # 2. 細胞類型評分計算 - 使用更小的批次大小
    print("\n計算細胞類型評分...")
    batch_size = 1000  # 減小批次大小
    
    for cell_type, markers in tqdm(marker_genes_dict.items(), desc="處理細胞類型"):
        available_markers = [m for m in markers if m in adata.var_names]
        if available_markers:
            try:
                scores = np.zeros(adata.shape[0])
                for batch_idx in range(0, adata.shape[0], batch_size):
                    end_idx = min(batch_idx + batch_size, adata.shape[0])
                    batch = adata[batch_idx:end_idx, available_markers].X
                    
                    if sp.issparse(batch):
                        batch = batch.toarray()
                    batch_gpu = cp.array(batch, dtype=cp.float32)
                    
                    # GPU 計算評分
                    batch_scores = cp.mean(batch_gpu, axis=1).get()
                    scores[batch_idx:end_idx] = batch_scores
                    
                    # 清理 GPU 記憶體
                    del batch_gpu
                    cp.get_default_memory_pool().free_all_blocks()
                
                adata.obs[f'{cell_type}_score'] = scores
                
            except Exception as e:
                print(f"\n處理 {cell_type} 時出錯: {str(e)}")
                continue
    
    # 3. GPU 加速降維 - 使用批次處理
    print("\n執行 GPU 加速降維...")
    n_comps = min(50, adata.shape[1] - 1)
    
    # PCA - 使用批次處理
    print("\n執行 PCA...")
    pca = PCA(n_components=n_comps)
    
    # 分批處理 PCA
    pca_results = np.zeros((adata.shape[0], n_comps))
    for batch_idx in tqdm(range(0, adata.shape[0], batch_size), desc="PCA 處理"):
        end_idx = min(batch_idx + batch_size, adata.shape[0])
        batch = adata[batch_idx:end_idx].X
        
        if sp.issparse(batch):
            batch = batch.toarray()
        
        pca_results[batch_idx:end_idx] = pca.fit_transform(batch)
    
    adata.obsm['X_pca'] = pca_results
    
    # UMAP
    if method.lower() == 'umap':
        print("\n執行 UMAP...")
        umap = UMAP(n_neighbors=15, n_components=2)
        adata.obsm['X_umap'] = umap.fit_transform(adata.obsm['X_pca'])
    
    # 4. 確定細胞類型
    print("\n確定細胞類型...")
    score_columns = [f'{ct}_score' for ct in marker_genes_dict.keys() 
                    if f'{ct}_score' in adata.obs.columns]
    
    if score_columns:
        score_df = adata.obs[score_columns]
        adata.obs['predicted_cell_type'] = score_df.idxmax(axis=1).str.replace('_score', '')
    
    print("\n=== GPU 分析完成！===")
    return adata

# 主程序保持不變
if __name__ == "__main__":
    start_time = time.time()
    
    print("\n正在載入數據文件...")
    adata = load_h5_data('./neuron_10k_v3_raw_feature_bc_matrix.h5')
    print(f"數據載入完成，形狀：{adata.shape}")
    
    print("\n正在載入標記基因...")
    marker_genes = load_marker_genes_from_csv('./mm_brain_markers.csv')
    
    analysis_start = time.time()
    adata = analyze_cell_types(adata, marker_genes, method='umap')
    print(f"\n總分析耗時：{time.time() - analysis_start:.2f}秒")
    
    print("\n正在生成可視化結果...")
    plot_cell_types(adata, method='umap', save_path='cell_types_analysis.png') 
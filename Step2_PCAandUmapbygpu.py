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
    """載入 h5ad 數據文件"""
    try:
        print(f"正在讀取文件：{file_path}")
        
        # 直接使用 scanpy 讀取 h5ad 格式的數據
        adata = sc.read_h5ad(file_path)
        
        print(f"成功載入數據，形狀：{adata.shape}")
        print(f"基因名稱示例：{list(adata.var_names)[:5]}")
        
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
    
    # 2. 細胞類型評分計算
    print("\n計算細胞類型評分...")
    batch_size = 1000
    successful_scores = []
    
    # 檢查基因名稱匹配情況
    print("\n基因名稱匹配檢查：")
    for cell_type, markers in marker_genes_dict.items():
        available_markers = [m for m in markers if m in adata.var_names]
        print(f"\n{cell_type}:")
        print(f"- 標記基因總數: {len(markers)}")
        print(f"- 匹配到的基因數: {len(available_markers)}")
        if available_markers:
            print(f"- 匹配到的基因示例: {available_markers[:3]}")
            
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
                successful_scores.append(cell_type)
                print(f"完成 {cell_type} 的評分計算")
                
            except Exception as e:
                print(f"計算評分時出錯: {str(e)}")
                continue
    
    # 檢查評分計算結果
    if not successful_scores:
        print("\n詳細診斷信息:")
        print(f"1. 數據形狀: {adata.shape}")
        print(f"2. 標記基因字典大小: {len(marker_genes_dict)}")
        print(f"3. 數據類型: {type(adata.X)}")
        print("4. 基因名稱示例:")
        print(list(adata.var_names)[:5])
        raise ValueError("沒有成功計算任何細胞類型的評分")
    
    # 3. GPU 加速降維
    print("\n執行 GPU 加速降維...")
    n_comps = min(10, adata.shape[1] - 1)
    
    # PCA
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
    score_columns = [f'{ct}_score' for ct in successful_scores]  # 只使用成功計算的評分
    
    if score_columns:
        score_df = adata.obs[score_columns]
        adata.obs['predicted_cell_type'] = score_df.idxmax(axis=1).str.replace('_score', '')
        print(f"已確定 {len(np.unique(adata.obs['predicted_cell_type']))} 種細胞類型")
    else:
        raise ValueError("沒有可用的細胞類型評分")
    
    print("\n=== GPU 分析完成！===")
    return adata

def plot_cell_types(adata, method='umap', save_path=None):
    """繪製細胞類型分布圖"""
    print("\n正在生成可視化圖表...")
    
    plt.figure(figsize=(12, 8))
    
    if method.lower() == 'umap':
        # 使用 scanpy 的繪圖功能
        sc.pl.umap(adata, 
                  color='predicted_cell_type',
                  title='Cell Types Distribution (UMAP)',
                  frameon=False,
                  show=False)
    
    if save_path:
        plt.savefig(save_path, 
                   dpi=300, 
                   bbox_inches='tight')
        print(f"圖表已保存至：{save_path}")
    
    plt.close()

# 主程序修改
if __name__ == "__main__":
    start_time = time.time()
    
    print("\n正在載入數據文件...")
    try:
        # 改為直接讀取 filtered_data.h5ad
        adata = load_h5_data('./filtered_data.h5ad')
        print(f"數據載入完成，形狀：{adata.shape}")
        
        print("\n正在載入標記基因...")
        marker_genes = load_marker_genes_from_csv('./mm_brain_markers.csv')
        
        analysis_start = time.time()
        adata = analyze_cell_types(adata, marker_genes, method='umap')
        print(f"\n總分析耗時：{time.time() - analysis_start:.2f}秒")
        
        print("\n正在生成可視化結果...")
        plot_cell_types(adata, method='umap', save_path='cell_types_analysis.png')
    except Exception as e:
        print(f"\n程序執行出錯：{str(e)}")
        print("請檢查數據文件格式是否正確") 
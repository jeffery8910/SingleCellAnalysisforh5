import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

def perform_qc_analysis(adata, save_path='qc_plots.png'):
    """執行QC分析並繪製圖表"""
    print("執行QC分析...")
    
    # 計算基本QC指標
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    
    # 創建QC圖表
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # 檢查並處理數據
    print("\n檢查數據質量...")
    print(f"數據形狀: {adata.shape}")
    print(f"包含NaN的基因數: {adata.var.isna().sum().sum()}")
    print(f"包含NaN的細胞數: {adata.obs.isna().sum().sum()}")
    
    # nFeature_RNA (基因數量分布)
    data = adata.obs['n_genes_by_counts'].dropna()
    if len(data) > 0:
        sns.violinplot(data=data, ax=axs[0])
        axs[0].set_title('nFeature_RNA')
        axs[0].set_ylabel('Number of genes')
    else:
        print("警告：n_genes_by_counts 數據全為空")
    
    # nCount_RNA (UMI計數分布)
    data = adata.obs['total_counts'].dropna()
    if len(data) > 0:
        sns.violinplot(data=data, ax=axs[1])
        axs[1].set_title('nCount_RNA')
        axs[1].set_ylabel('Number of UMIs')
    else:
        print("警告：total_counts 數據全為空")
    
    # percent.mt (線粒體基因比例)
    try:
        adata.var['mt'] = adata.var_names.str.startswith('mt-')
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
        data = adata.obs['pct_counts_mt'].dropna()
        if len(data) > 0:
            sns.violinplot(data=data, ax=axs[2])
            axs[2].set_title('percent.mt')
            axs[2].set_ylabel('Percentage of mt-genes')
        else:
            print("警告：pct_counts_mt 數據全為空")
    except Exception as e:
        print(f"計算線粒體比例時出錯：{str(e)}")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 打印有效的QC統計信息
    print("\nQC統計信息：")
    try:
        print(f"中位數基因數：{np.median(adata.obs['n_genes_by_counts'].dropna()):.0f}")
        print(f"中位數UMI數：{np.median(adata.obs['total_counts'].dropna()):.0f}")
        print(f"中位數線粒體比例：{np.median(adata.obs['pct_counts_mt'].dropna()):.2f}%")
    except Exception as e:
        print(f"計算統計信息時出錯：{str(e)}")
    
    return adata

def filter_cells(adata, min_genes=200, max_genes=8000, max_mt_pct=20):
    """根據設定的標準過濾細胞"""
    print("\n執行細胞過濾...")
    print(f"過濾前細胞數：{adata.n_obs}")
    
    # 打印過濾前的統計信息
    print("\n過濾前數據分布：")
    print(f"基因數範圍：{adata.obs['n_genes_by_counts'].min():.0f} - {adata.obs['n_genes_by_counts'].max():.0f}")
    print(f"線粒體比例範圍：{adata.obs['pct_counts_mt'].min():.2f}% - {adata.obs['pct_counts_mt'].max():.2f}%")
    
    # 檢查每個過濾條件會過濾掉多少細胞
    n_genes_low = sum(adata.obs['n_genes_by_counts'] <= min_genes)
    n_genes_high = sum(adata.obs['n_genes_by_counts'] >= max_genes)
    n_mt_high = sum(adata.obs['pct_counts_mt'] >= max_mt_pct)
    
    print("\n各條件將過濾掉的細胞數：")
    print(f"基因數 < {min_genes}: {n_genes_low} 個細胞")
    print(f"基因數 > {max_genes}: {n_genes_high} 個細胞")
    print(f"線粒體比例 > {max_mt_pct}%: {n_mt_high} 個細胞")
    
    # 應用過濾條件
    adata = adata[
        (adata.obs['n_genes_by_counts'] > min_genes) &
        (adata.obs['n_genes_by_counts'] < max_genes) &
        (adata.obs['pct_counts_mt'] < max_mt_pct)
    ].copy()
    
    print(f"\n過濾後細胞數：{adata.n_obs}")
    
    if adata.n_obs == 0:
        raise ValueError("過濾條件太嚴格，導致所有細胞都被過濾掉了！")
    
    return adata

def plot_highly_variable_genes(adata, save_path='highly_variable_genes.png'):
    """繪製高變異基因圖"""
    print("\n分析高變異基因...")
    
    # 1. 數據預處理
    print("正在標準化數據...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    try:
        # 2. 計算高變異基因
        print("計算高變異基因...")
        sc.pp.highly_variable_genes(adata, 
                                  min_mean=0.0125, 
                                  max_mean=3, 
                                  min_disp=0.5,
                                  n_bins=20,  # 減少bin數量
                                  span=0.3)   # 調整平滑參數
        
        # 3. 繪圖
        print(f"選擇的高變異基因數：{sum(adata.var.highly_variable)}")
        plt.figure(figsize=(10, 5))
        sc.pl.highly_variable_genes(adata, show=False)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"警告：計算高變異基因時出錯 - {str(e)}")
        print("嘗試使用替代方法...")
        
        try:
            # 替代方法：使用更簡單的參數設置
            sc.pp.highly_variable_genes(adata, 
                                      flavor='seurat',  # 使用Seurat方法
                                      n_top_genes=2000) # 選擇前2000個高變異基因
            
            print(f"使用替代方法選擇的高變異基因數：{sum(adata.var.highly_variable)}")
            plt.figure(figsize=(10, 5))
            sc.pl.highly_variable_genes(adata, show=False)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e2:
            print(f"替代方法也失敗：{str(e2)}")
            print("跳過高變異基因分析...")
            return adata
    
    return adata

if __name__ == "__main__":
    # 載入數據
    print("載入數據...")
    try:
        # 首先載入原始數據
        adata = sc.read_10x_mtx(
            './filtered_gene_bc_matrices/',  # 替換為您的數據路徑
            var_names='gene_symbols',
            cache=True
        )
        
        # 執行QC分析
        adata = perform_qc_analysis(adata)
        
        # 過濾細胞（使用更寬鬆的標準）
        try:
            adata = filter_cells(adata, 
                               min_genes=200,    
                               max_genes=8000,   
                               max_mt_pct=20)    
        except ValueError as e:
            print(f"\n警告：{str(e)}")
            print("請根據上述統計信息調整過濾參數！")
            exit(1)
        
        # 繪製高變異基因圖
        adata = plot_highly_variable_genes(adata)
        
        # 保存處理後的數據
        print("\n保存處理後的數據...")
        adata.write('filtered_data.h5ad')
        print("數據已保存至 filtered_data.h5ad")
        
    except Exception as e:
        print(f"分析過程中出錯：{str(e)}") 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# ================= 配置区域 =================
FILE_DETAILED = 'ExperimentResult.xlsx'  # 含有 pass@1-64 的详细文件
FILE_SUMMARY = 'results_summary.csv'                   # 含有 GRPO 模型数据的汇总文件
OUTPUT_DIR = 'final_plots'
# ===========================================

def clean_detailed_data(file_path):
    """清洗详细数据文件 (ExperimentResult)"""
    if not os.path.exists(file_path):
        print(f"警告: 找不到文件 {file_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(file_path)
    except:
        df = pd.read_excel(file_path)
        
    # 统一列名
    df.columns = [c.strip() for c in df.columns]
    
    # 统一数据集名称 (全部小写)
    if 'Dataset' in df.columns:
        df['Dataset'] = df['Dataset'].astype(str).str.lower()
    
    # 统一模型名称 (全部小写)
    if 'Model' in df.columns:
        df['Model'] = df['Model'].astype(str).str.lower()

    # 统一温度格式 (转为 float)
    if 'Temperature' in df.columns:
        df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')

    # 数值转换：如果是 0-1 之间的小数，转为百分比
    metric_cols = [c for c in df.columns if c.startswith('pass@') or c.startswith('maj@')]
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].max() <= 1.0:
            df[col] = df[col] * 100.0
            
    return df

def clean_summary_data(file_path):
    """清洗汇总数据文件 (results_summary)"""
    if not os.path.exists(file_path):
        print(f"警告: 找不到文件 {file_path}")
        return pd.DataFrame()
        
    df = pd.read_csv(file_path)
    
    # 映射列名以匹配详细数据
    col_map = {
        'Pass@1 (%)': 'pass@1',
        'Maj@1 (%)': 'maj@1',
        'Temperature': 'Temperature_Raw' # 先暂存原始格式
    }
    df = df.rename(columns=col_map)
    
    # 清洗 Dataset 和 Model
    df['Dataset'] = df['Dataset'].astype(str).str.lower()
    df['Model'] = df['Model'].astype(str).str.lower()
    
    # 清洗温度 (t06 -> 0.6, t10 -> 1.0)
    def parse_temp(val):
        val = str(val).lower()
        if 't' in val:
            val = val.replace('t', '')
            # 假设 t06 = 0.6, t10 = 1.0
            if len(val) == 2:
                return float(val[0] + '.' + val[1])
            elif len(val) > 2: # t100? t12?
                return float(val) / 10.0 if float(val) > 10 else float(val)
        try:
            return float(val)
        except:
            return None
            
    df['Temperature'] = df['Temperature_Raw'].apply(parse_temp)
    
    return df

def plot_1_scaling_curves(df):
    """图1: Pass@k 扩展曲线 (利用详细数据文件)"""
    print("正在绘制 Scaling Curves...")
    pass_cols = [c for c in df.columns if c.startswith('pass@')]
    if not pass_cols: return

    # 融合数据用于绘图
    id_vars = ['Dataset', 'Temperature', 'Model']
    df_melt = df.melt(id_vars=[c for c in id_vars if c in df.columns], 
                      value_vars=pass_cols, var_name='k_str', value_name='Accuracy')
    
    df_melt['k'] = df_melt['k_str'].apply(lambda x: int(x.split('@')[1]))
    df_melt = df_melt.dropna(subset=['Accuracy'])

    # 只画 Base 模型 (因为通常只有 Base 跑了全量 scaling)，或者都画
    datasets = df_melt['Dataset'].unique()
    
    for ds in datasets:
        subset = df_melt[df_melt['Dataset'] == ds]
        if subset.empty: continue
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=subset, x='k', y='Accuracy', hue='Temperature', style='Model',
                     palette='viridis', markers=True, dashes=False, linewidth=2.5)
        
        plt.xscale('log', base=2)
        plt.title(f'Pass@k Scaling Law Analysis - {ds.upper()}', fontsize=14, fontweight='bold')
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.xlabel('Number of Samples (k) - Log Scale', fontsize=12)
        plt.grid(True, which="both", ls="-", alpha=0.2)
        
        # 设置 X 轴刻度
        ks = sorted(subset['k'].unique())
        plt.xticks(ks, ks)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'1_scaling_curve_{ds}.png'), dpi=300)
        plt.close()

def plot_2_model_comparison(df):
    """图2: Base vs Instruct 模型对比 (利用汇总数据文件)"""
    print("正在绘制模型对比图...")
    # 筛选出有多个模型的数据集
    counts = df.groupby('Dataset')['Model'].nunique()
    target_datasets = counts[counts > 1].index.tolist()
    
    if not target_datasets:
        # 如果没有数据集包含两个模型，尝试画所有
        target_datasets = df['Dataset'].unique()

    subset = df[df['Dataset'].isin(target_datasets)]
    
    # 甚至我们可以固定 Temperature = 1.0 (最公平的对比)
    subset_t1 = subset[subset['Temperature'] == 1.0]
    if subset_t1.empty: subset_t1 = subset # 如果没有 1.0，就用全部

    plt.figure(figsize=(10, 6))
    sns.barplot(data=subset_t1, x='Dataset', y='maj@1', hue='Model', 
                palette='Set2', edgecolor='black')
    
    plt.title('Base vs Instruct (GRPO) Model Performance (Maj@1)', fontsize=14, fontweight='bold')
    plt.ylabel('Majority Vote Accuracy (%)', fontsize=12)
    plt.xlabel('Dataset', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '2_model_comparison_maj1.png'), dpi=300)
    plt.close()

def plot_3_temp_impact(df):
    """图3: 温度对准确率的影响 (Base 模型)"""
    print("正在绘制温度影响图...")
    # 只看 Base 模型，因为 Base 对温度更敏感
    subset = df[df['Model'] == 'base']
    if subset.empty: subset = df
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=subset, x='Dataset', y='pass@1', hue='Temperature', 
                palette='coolwarm', edgecolor='black')
    
    plt.title('Impact of Temperature on Single-Shot Accuracy (Pass@1)', fontsize=14, fontweight='bold')
    plt.ylabel('Pass@1 Accuracy (%)', fontsize=12)
    plt.legend(title='Temperature')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '3_temp_impact_pass1.png'), dpi=300)
    plt.close()

def plot_4_voting_gain(df_summary, df_detailed):
    """图4: 投票收益分析 (Maj@1 vs Pass@1)"""
    print("正在绘制投票收益散点图...")
    # 合并两个数据源，优先使用 summary (包含更多模型)，补充 detailed 中的数据
    cols = ['Dataset', 'Model', 'Temperature', 'pass@1', 'maj@1']
    combined = pd.concat([
        df_summary[cols],
        df_detailed[cols]
    ], ignore_index=True).drop_duplicates()
    
    combined = combined.dropna(subset=['pass@1', 'maj@1'])

    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=combined, x='pass@1', y='maj@1', hue='Dataset', style='Model', 
                    s=150, alpha=0.8, palette='deep')
    
    # 绘制 y=x 参考线
    limit = max(combined['pass@1'].max(), combined['maj@1'].max()) + 5
    plt.plot([0, limit], [0, limit], 'r--', label='No Gain (y=x)')
    
    plt.title('Performance Gain: Majority Voting vs Single Pass', fontsize=14, fontweight='bold')
    plt.xlabel('Single-Shot Accuracy (Pass@1) %', fontsize=12)
    plt.ylabel('Majority Vote Accuracy (Maj@1) %', fontsize=12)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    plt.xlim(0, limit)
    plt.ylim(0, limit)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '4_voting_gain_scatter.png'), dpi=300)
    plt.close()

def plot_5_heatmap(df):
    """图5: 热力图展示 Base 模型在不同数据集和温度下的 Maj@1"""
    print("正在绘制热力图...")
    subset = df[df['Model'] == 'base']
    if subset.empty: return

    # Pivot Data: Index=Dataset, Columns=Temperature, Values=maj@1
    try:
        pivot = subset.pivot(index='Dataset', columns='Temperature', values='maj@1')
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Accuracy (%)'})
        plt.title('Heatmap: Base Model Accuracy (Maj@1)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '5_heatmap_base_maj1.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"无法绘制热力图: {e}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 1. 加载并清洗数据
    df_detailed = clean_detailed_data(FILE_DETAILED)
    df_summary = clean_summary_data(FILE_SUMMARY)
    
    print(f"详细数据加载: {len(df_detailed)} 行")
    print(f"汇总数据加载: {len(df_summary)} 行")
    
    # 2. 生成图表
    # 图1: 使用详细数据画 Scaling Curve
    if not df_detailed.empty:
        plot_1_scaling_curves(df_detailed)
    
    # 图2 & 图3: 使用汇总数据 (因为它包含 GRPO 和 Math 数据集)
    if not df_summary.empty:
        plot_2_model_comparison(df_summary)
        plot_3_temp_impact(df_summary)
        plot_5_heatmap(df_summary)
    
    # 图4: 结合两者画散点图
    if not df_summary.empty or not df_detailed.empty:
        plot_4_voting_gain(df_summary, df_detailed)

    print(f"\n✅ 所有绘图已完成！请查看文件夹: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
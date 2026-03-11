import pandas as pd
import json
import os
import numpy as np

def auto_refine_features(output_dir='outputs_final', threshold=0.1, target_col='SOH_True'):
    """
    自动筛选低贡献特征并更新训练配置
    """
    corr_path = os.path.join(output_dir, 'hi_corr_pearson.csv')
    config_path = 'feature_config.json'
    
    if not os.path.exists(corr_path):
        print(f"❌ 未找到相关性分析文件: {corr_path}")
        print("💡 请确保已运行 soh_final_pipeline.py 且输出了 hi_corr_pearson.csv")
        return

    # 1. 加载相关性矩阵
    # 假设 CSV 的索引行是特征名
    df_corr = pd.read_csv(corr_path, index_col=0)
    
    if target_col not in df_corr.columns:
        print(f"❌ 目标列 {target_col} 不在相关性矩阵中。")
        return

    # 2. 提取与目标 (SOH) 的相关性绝对值
    correlations = df_corr[target_col].abs().sort_values(ascending=False)
    
    # 定义必须保留的物理基准特征（如 days）
    essential_features = ['days']
    
    # 定义待筛选的候选标量特征 (Scalars)
    # 根据 soh_pi_uae.py 逻辑，通常包括 seg_ah, avg_curr, avg_temp 等
    candidate_features = [f for f in correlations.index if f not in [target_col] + essential_features]
    
    # 3. 自动筛选
    selected_scalars = [f for f in candidate_features if correlations[f] >= threshold]
    dropped_scalars = [f for f in candidate_features if correlations[f] < threshold]

    # 4. 生成新配置
    # 注意：input_size 需要根据 selected_scalars 长度调整
    new_config = {
        "feature_selection": {
            "target": target_col,
            "threshold": threshold,
            "selected_scalars": selected_scalars,
            "dropped_scalars": dropped_scalars,
            "input_scalar_dim": len(selected_scalars)
        },
        "model_params": {
            "regressor_input_dim": 64 + len(selected_scalars) # 64是电压指纹降维后的特征
        }
    }

    # 5. 保存配置
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(new_config, f, indent=4, ensure_ascii=False)

    print(f"✅ 特征筛选完成！")
    print(f"📊 保留特征: {selected_scalars}")
    print(f"🗑️ 剔除特征: {dropped_scalars}")
    print(f"📝 配置文件已更新: {config_path}")

if __name__ == "__main__":
    # 执行筛选逻辑
    auto_refine_features(threshold=0.15, target_col='soh_true') # 可根据热力图观察结果调整阈值
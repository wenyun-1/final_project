import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score

# ⚡️ 注意：这里导入的不再是 SOH_MIN, SOH_MAX，而是 AH_MAX_SCALE
from soh_train import PIDataset, MultiModal_PI_UAE, RAW_FILES, BATCH_SIZE, AH_MAX_SCALE

def evaluate_and_plot():
    if not os.path.exists('pi_uae_model_weights.pth') or not os.path.exists('scaler_stats.npz'):
        print("❌ 找不到模型或参数文件！请先运行 soh_train.py")
        return

    print("🚀 启动离线评估模式 (离散区间容量验证)...")
    
    test_ds = PIDataset(RAW_FILES, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModal_PI_UAE().to(device)
    model.load_state_dict(torch.load('pi_uae_model_weights.pth', map_location=device, weights_only=True))
    model.eval() 

    results_by_veh = {}
    with torch.no_grad():
        for curr_fp, curr_sc, _, _, y_curr, vids, days in test_loader:
            latent_curr, _ = model(curr_fp.to(device), curr_sc.to(device))
            
            # ⚡️ 反归一化：直接还原出真实的区间物理容量 (Ah)
            preds_ah = latent_curr.cpu().numpy().flatten() * AH_MAX_SCALE
            trues_ah = y_curr.numpy() * AH_MAX_SCALE
            vids, days = vids.numpy(), days.numpy()
            
            for i in range(len(vids)):
                vid = vids[i]
                if vid not in results_by_veh: results_by_veh[vid] = {'days':[], 'trues_ah':[], 'preds_ah':[]}
                results_by_veh[vid]['days'].append(days[i])
                results_by_veh[vid]['trues_ah'].append(trues_ah[i])
                results_by_veh[vid]['preds_ah'].append(preds_ah[i])

    num_vehicles = len(results_by_veh)
    if num_vehicles == 0: return
        
    fig, axes = plt.subplots(num_vehicles, 1, figsize=(10, 4 * num_vehicles))
    if num_vehicles == 1: axes = [axes]
    
    overall_rmse_discrete = []
    
    for i, (vid, res) in enumerate(results_by_veh.items()):
        sorted_indices = np.argsort(res['days'])
        days_sorted = np.array(res['days'])[sorted_indices]
        trues_ah_sorted = np.array(res['trues_ah'])[sorted_indices]
        preds_ah_sorted = np.array(res['preds_ah'])[sorted_indices]
        
        # ⚡️ 核心步骤：将离散物理 Ah 转换为 SOH 百分比
        # 为了稳健，取前 5 个离散真实点的平均值作为该车的出厂 100% 基准
        initial_ah_baseline = np.mean(trues_ah_sorted[:5])
        
        trues_soh_discrete = (trues_ah_sorted / initial_ah_baseline) * 100.0
        preds_soh_discrete = (preds_ah_sorted / initial_ah_baseline) * 100.0
        
        # 严格使用离散点计算误差！
        rmse_val = np.sqrt(mean_squared_error(trues_soh_discrete, preds_soh_discrete))
        r2_val = r2_score(trues_soh_discrete, preds_soh_discrete)
        overall_rmse_discrete.append(rmse_val)
        
        veh_name = test_ds.vehicle_names.get(vid, f"Vehicle_{vid}")
        print(f"🚗 {veh_name}: 基于离散物理容量估计 RMSE = {rmse_val:.3f}% (R2={r2_val:.3f})")
        
        # ----------------- 画图部分 -----------------
        ax = axes[i]
        
        # 1. 画真实的离散点 (完全抛弃死板的拟合线)
        ax.scatter(days_sorted, trues_soh_discrete, c='black', alpha=0.6, s=25, label='True SOH (Discrete Interval Capacity)')
        
        # 2. 画预测的离散点
        ax.scatter(days_sorted, preds_soh_discrete, c='royalblue', alpha=0.5, s=25, label='Predicted SOH (Discrete Model Output)')
        
        # 3. 添加一条由预测散点滑动平均而来的红线 (不参与误差计算，仅供肉眼观察老化的非线性趋势)
        if len(preds_soh_discrete) > 10:
            visual_trend = pd.Series(preds_soh_discrete).rolling(window=15, min_periods=1, center=True).mean()
            ax.plot(days_sorted, visual_trend, 'r-', alpha=0.9, linewidth=2.5, label='Visual Trend of Predictions')

        ax.set_title(f"{veh_name} | RMSE={rmse_val:.3f}%, R2={r2_val:.3f}")
        ax.set_xlabel("Days")
        ax.set_ylabel("SOH (%)")
        ax.grid(True, alpha=0.4)
        ax.legend()
    # === 在 soh_eval.py 循环结束后，出图之前，加入这段导出代码 ===
    print("\n💾 正在将第三章的 SOH 预测结果导出，准备为第四章 SOC 估计提供老化参数...")
    export_data = []
    for vid, res in results_by_veh.items():
        veh_name = test_ds.vehicle_names.get(vid, f"Vehicle_{vid}")
        # res['days'] 是原始天数，smoothed_preds 是你跑出的那条平滑红线
        # 我们用模型预测出的、去噪后的红线作为真实的后续输入！
        window_size = 15
        smoothed_preds = pd.Series(res['preds']).rolling(window=window_size, min_periods=1, center=True).mean().values
        
        for d, p_soh in zip(res['days'], smoothed_preds):
            export_data.append({
                'Vehicle': veh_name,
                'Days': d,
                'Pred_SOH': p_soh / 100.0  # 转换为 0~1 的小数供 GRU 使用
            })
    
    df_export = pd.DataFrame(export_data)
    df_export.to_csv("SOH_Predictions_For_SOC.csv", index=False)
    print("✅ SOH 老化特征已成功保存至: SOH_Predictions_For_SOC.csv")

    print("\n" + "="*50)
    print(f"🏆 全新范式：基于离散区间物理容量的最终平均 RMSE: {np.mean(overall_rmse_discrete):.3f}%")
    print("="*50)
    
    plt.tight_layout()
    plt.savefig('PI_UAE_Interval_Capacity_Paradigm.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    evaluate_and_plot()
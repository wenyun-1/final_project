import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score

# ⚡️ 重新导回大论文标准框架的参数
from exam import PIDataset, MultiModal_PI_UAE, RAW_FILES, BATCH_SIZE, SOH_MIN, SOH_MAX

def evaluate_and_plot():
    if not os.path.exists('pi_uae_model_weights.pth') or not os.path.exists('scaler_stats.npz'):
        print("❌ 找不到模型或参数文件！请先运行 soh_train.py")
        return

    print("🚀 启动离线评估模式 (保序回归物理验证)...")
    
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
            
            # 反归一化为百分比 SOH
            preds = latent_curr.cpu().numpy().flatten() * (SOH_MAX - SOH_MIN) + SOH_MIN
            trues = y_curr.numpy() * (SOH_MAX - SOH_MIN) + SOH_MIN
            vids, days = vids.numpy(), days.numpy()
            
            for i in range(len(vids)):
                vid = vids[i]
                if vid not in results_by_veh: results_by_veh[vid] = {'days':[], 'trues':[], 'preds':[]}
                results_by_veh[vid]['days'].append(days[i])
                results_by_veh[vid]['trues'].append(trues[i])
                results_by_veh[vid]['preds'].append(preds[i])

    num_vehicles = len(results_by_veh)
    if num_vehicles == 0: return
        
    fig, axes = plt.subplots(num_vehicles, 1, figsize=(10, 4 * num_vehicles))
    if num_vehicles == 1: axes = [axes]
    
    overall_rmse_raw, overall_rmse_smooth = [], []
    
    for i, (vid, res) in enumerate(results_by_veh.items()):
        sorted_indices = np.argsort(res['days'])
        days_sorted = np.array(res['days'])[sorted_indices]
        trues_sorted = np.array(res['trues'])[sorted_indices]
        preds_sorted = np.array(res['preds'])[sorted_indices]
        
        rmse_raw = np.sqrt(mean_squared_error(trues_sorted, preds_sorted))
        
        # 预测结果的时序平滑 (去除神经网路的少许局部预测噪声)
        window_size = 15 
        smoothed_preds = pd.Series(preds_sorted).rolling(window=window_size, min_periods=1, center=True).mean().values
        
        rmse_smooth = np.sqrt(mean_squared_error(trues_sorted, smoothed_preds))
        r2_smooth = r2_score(trues_sorted, smoothed_preds)
        
        overall_rmse_raw.append(rmse_raw)
        overall_rmse_smooth.append(rmse_smooth)
        
        veh_name = test_ds.vehicle_names.get(vid, f"Vehicle_{vid}")
        print(f"🚗 {veh_name}: 预测 SOH RMSE = {rmse_smooth:.3f}% (R2={r2_smooth:.3f})")
        
        # ----------------- 画图部分 -----------------
        ax = axes[i]
        
        # ⚡️ 黑线：由于是保序回归生成的，黑线自然呈现为“下楼梯”的阶梯状，完美契合物理不可逆老化
        ax.plot(days_sorted, trues_sorted, 'k-', linewidth=3, alpha=0.9, label='True SOH (Isotonic Physical Trend)')
        
        # 蓝点：模型的单次原始预测
        ax.scatter(days_sorted, preds_sorted, c='royalblue', alpha=0.4, s=20, label='Raw Network Output')
        
        # 红线：模型预测的平滑结果，用以对比真值的阶梯
        ax.plot(days_sorted, smoothed_preds, 'r-', alpha=0.9, linewidth=2.5, label='Filtered Final SOH')

        ax.set_title(f"{veh_name} | Evaluation RMSE={rmse_smooth:.3f}%, R2={r2_smooth:.3f}")
        ax.set_xlabel("Days")
        ax.set_ylabel("SOH (%)")
        ax.grid(True, alpha=0.4)
        ax.legend()

    print("\n" + "="*50)
    print(f"🏆 保序物理回归范式：最终平均 RMSE: {np.mean(overall_rmse_smooth):.3f}%")
    print("="*50)
    
    plt.tight_layout()
    plt.savefig('PI_UAE_Isotonic_Regression_Validation.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    evaluate_and_plot()
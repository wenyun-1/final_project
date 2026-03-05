import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader

from soh_train import AH_MAX_SCALE, BATCH_SIZE, PIDataset, RAW_FILES, MultiModal_PI_UAE


def evaluate_and_plot():
    if not os.path.exists("pi_uae_model_weights.pth") or not os.path.exists("scaler_stats.npz"):
        print("❌ 找不到模型或参数文件！请先运行 soh_train.py")
        return

    print("🚀 启动离线评估模式 (离散区间容量验证)...")
    test_ds = PIDataset(RAW_FILES, is_train=False)
    if len(test_ds) == 0:
        print("❌ 测试集为空，无法评估。")
        return

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModal_PI_UAE().to(device)
    model.load_state_dict(torch.load("pi_uae_model_weights.pth", map_location=device, weights_only=True))
    model.eval()

    results_by_veh = {}
    with torch.no_grad():
        for curr_fp, curr_sc, _, _, y_curr, vids, days in test_loader:
            pred_curr, _ = model(curr_fp.to(device), curr_sc.to(device))
            preds_ah = pred_curr.cpu().numpy().flatten() * AH_MAX_SCALE
            trues_ah = y_curr.numpy() * AH_MAX_SCALE
            vids, days = vids.numpy(), days.numpy()

            for i in range(len(vids)):
                vid = vids[i]
                if vid not in results_by_veh:
                    results_by_veh[vid] = {"days": [], "trues_ah": [], "preds_ah": []}
                results_by_veh[vid]["days"].append(days[i])
                results_by_veh[vid]["trues_ah"].append(trues_ah[i])
                results_by_veh[vid]["preds_ah"].append(preds_ah[i])

    if not results_by_veh:
        print("❌ 没有聚合到车辆级结果。")
        return

    num_vehicles = len(results_by_veh)
    fig, axes = plt.subplots(num_vehicles, 1, figsize=(10, 4 * num_vehicles))
    if num_vehicles == 1:
        axes = [axes]

    overall_rmse_discrete = []
    export_data = []

    for i, (vid, res) in enumerate(results_by_veh.items()):
        sorted_idx = np.argsort(res["days"])
        days_sorted = np.array(res["days"])[sorted_idx]
        trues_ah_sorted = np.array(res["trues_ah"])[sorted_idx]
        preds_ah_sorted = np.array(res["preds_ah"])[sorted_idx]

        base_count = min(5, len(trues_ah_sorted))
        initial_ah_baseline = np.mean(trues_ah_sorted[:base_count])
        if initial_ah_baseline <= 1e-6:
            print(f"⚠️ 车辆 {vid} 基准容量异常，跳过。")
            continue

        trues_soh_discrete = (trues_ah_sorted / initial_ah_baseline) * 100.0
        preds_soh_discrete = (preds_ah_sorted / initial_ah_baseline) * 100.0

        rmse_val = np.sqrt(mean_squared_error(trues_soh_discrete, preds_soh_discrete))
        r2_val = r2_score(trues_soh_discrete, preds_soh_discrete)
        overall_rmse_discrete.append(rmse_val)

        veh_name = test_ds.vehicle_names.get(vid, f"Vehicle_{vid}")
        print(f"🚗 {veh_name}: RMSE={rmse_val:.3f}% (R2={r2_val:.3f})")

        ax = axes[i]
        ax.scatter(days_sorted, trues_soh_discrete, c="black", alpha=0.6, s=25, label="True SOH")
        ax.scatter(days_sorted, preds_soh_discrete, c="royalblue", alpha=0.5, s=25, label="Predicted SOH")

        smoothed_preds = pd.Series(preds_soh_discrete).rolling(window=15, min_periods=1, center=True).mean().values
        if len(preds_soh_discrete) > 10:
            ax.plot(days_sorted, smoothed_preds, "r-", alpha=0.9, linewidth=2.5, label="Prediction Trend")

        ax.set_title(f"{veh_name} | RMSE={rmse_val:.3f}%, R2={r2_val:.3f}")
        ax.set_xlabel("Days")
        ax.set_ylabel("SOH (%)")
        ax.grid(True, alpha=0.4)
        ax.legend()

        for d, p_soh in zip(days_sorted, smoothed_preds):
            export_data.append({"Vehicle": veh_name, "Days": int(d), "Pred_SOH": float(p_soh / 100.0)})

    if export_data:
        df_export = pd.DataFrame(export_data)
        df_export.to_csv("SOH_Predictions_For_SOC.csv", index=False)
        print("✅ SOH 老化特征已成功保存至: SOH_Predictions_For_SOC.csv")

    if overall_rmse_discrete:
        print("\n" + "=" * 50)
        print(f"🏆 平均 RMSE: {np.mean(overall_rmse_discrete):.3f}%")
        print("=" * 50)

    plt.tight_layout()
    plt.savefig("PI_UAE_Interval_Capacity_Paradigm.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    evaluate_and_plot()

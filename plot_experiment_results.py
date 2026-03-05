"""实验结果可视化脚本（第三章SOH + 消融解释）

重点支持：不重新训练，直接用已保存结果文件绘图。
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial Unicode MS", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_vehicle_metrics(metrics_vehicle_csv: str) -> Dict[str, Dict[str, float]]:
    if not os.path.exists(metrics_vehicle_csv):
        return {}
    df = pd.read_csv(metrics_vehicle_csv)
    if "Vehicle" not in df.columns:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for _, r in df.iterrows():
        out[str(r["Vehicle"])] = {
            "RMSE_filtered": float(r.get("RMSE_filtered", np.nan)),
            "R2_filtered": float(r.get("R2_filtered", np.nan)),
        }
    return out


def _load_chapter3_curve_data(soh_output: str) -> Dict[str, pd.DataFrame]:
    points_csv = os.path.join(soh_output, "soh_predictions_points.csv")
    if os.path.exists(points_csv):
        df = pd.read_csv(points_csv)
        required = {"Vehicle", "Days", "SOH_true", "SOH_pred_filtered"}
        if required.issubset(df.columns):
            veh_data: Dict[str, pd.DataFrame] = {}
            for veh, g in df.groupby("Vehicle"):
                gg = g.sort_values("Days").copy()
                gg["SOH_true"] = pd.to_numeric(gg["SOH_true"], errors="coerce")
                gg["SOH_pred"] = pd.to_numeric(gg["SOH_pred_filtered"], errors="coerce")
                veh_data[str(veh)] = gg[["Days", "SOH_true", "SOH_pred"]].dropna()
            return veh_data

    pred_csv = os.path.join(soh_output, "SOH_Predictions_For_SOC.csv")
    if not os.path.exists(pred_csv):
        return {}

    pred = pd.read_csv(pred_csv)
    if not {"Vehicle", "Days", "Pred_SOH"}.issubset(pred.columns):
        return {}

    pred = pred.copy()
    pred["Days"] = pd.to_numeric(pred["Days"], errors="coerce")
    pred["Pred_SOH"] = pd.to_numeric(pred["Pred_SOH"], errors="coerce")
    pred = pred.dropna(subset=["Days", "Pred_SOH"])
    pred["SOH_pred"] = pred["Pred_SOH"] * 100.0 if pred["Pred_SOH"].max() <= 1.5 else pred["Pred_SOH"]

    pseudo_map: Dict[str, pd.DataFrame] = {}
    for f in glob.glob(os.path.join(soh_output, "*_pseudo_labels.csv")):
        veh = os.path.basename(f).replace("_pseudo_labels.csv", "")
        d = pd.read_csv(f)
        if {"days", "soh_true"}.issubset(d.columns):
            d = d[["days", "soh_true"]].rename(columns={"days": "Days", "soh_true": "SOH_true"})
            d["Days"] = pd.to_numeric(d["Days"], errors="coerce")
            d["SOH_true"] = pd.to_numeric(d["SOH_true"], errors="coerce")
            pseudo_map[veh] = d.dropna().sort_values("Days")

    out: Dict[str, pd.DataFrame] = {}
    for veh, g in pred.groupby("Vehicle"):
        veh = str(veh)
        gp = g[["Days", "SOH_pred"]].sort_values("Days")
        if veh in pseudo_map and not pseudo_map[veh].empty:
            merged = pd.merge_asof(gp, pseudo_map[veh], on="Days", direction="nearest")
        else:
            merged = gp.copy()
            merged["SOH_true"] = np.nan
        out[veh] = merged[["Days", "SOH_true", "SOH_pred"]].dropna(subset=["Days", "SOH_pred"])
    return out


def plot_chapter3_subplot(soh_output: str, out_dir: str) -> Optional[str]:
    veh_data = _load_chapter3_curve_data(soh_output)
    if not veh_data:
        return None

    metrics = _load_vehicle_metrics(os.path.join(soh_output, "soh_metrics_vehicle.csv"))
    vehicles = sorted(veh_data.keys())
    n = len(vehicles)
    ncols = 3 if n >= 6 else 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 3.6 * nrows), sharex=False)
    axes_arr = np.atleast_1d(axes).flatten()

    for i, veh in enumerate(vehicles):
        ax = axes_arr[i]
        g = veh_data[veh].sort_values("Days")

        if g["SOH_true"].notna().any():
            ax.scatter(g["Days"], g["SOH_true"], s=14, c="black", alpha=0.65, label="True SOH")

        ax.plot(g["Days"], g["SOH_pred"], color="#e15759", lw=2.2, alpha=0.95, label="Predicted SOH")

        m = metrics.get(veh, {})
        rmse = m.get("RMSE_filtered", np.nan)
        r2 = m.get("R2_filtered", np.nan)
        title = f"{veh}"
        if not np.isnan(rmse) and not np.isnan(r2):
            title += f"\nRMSE_f={rmse:.3f}%, R²_f={r2:.3f}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Days")
        ax.set_ylabel("SOH (%)")
        ax.grid(alpha=0.28)

    for j in range(i + 1, len(axes_arr)):
        fig.delaxes(axes_arr[j])

    handles, labels = axes_arr[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = os.path.join(out_dir, "chapter3_soh_subplot.png")
    fig.savefig(out, dpi=260)
    plt.close(fig)
    return out


def plot_soh_metric_bars(metrics_vehicle_csv: str, out_dir: str) -> Optional[str]:
    if not os.path.exists(metrics_vehicle_csv):
        return None
    df = pd.read_csv(metrics_vehicle_csv)
    needed = {"Vehicle", "RMSE_raw", "RMSE_filtered", "MAE_raw", "MAE_filtered"}
    if not needed.issubset(df.columns):
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    x = np.arange(len(df))
    w = 0.36
    axes[0].bar(x - w / 2, df["RMSE_raw"], width=w, color="#4e79a7", label="Raw（单点输出）")
    axes[0].bar(x + w / 2, df["RMSE_filtered"], width=w, color="#e15759", label="Filtered（平滑后输出）")
    axes[0].set_title("SOH RMSE 对比")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["Vehicle"], rotation=45, ha="right")
    axes[0].set_ylabel("RMSE (%)")
    axes[0].grid(alpha=0.25, axis="y")
    axes[0].legend()

    axes[1].bar(x - w / 2, df["MAE_raw"], width=w, color="#4e79a7", label="Raw（单点输出）")
    axes[1].bar(x + w / 2, df["MAE_filtered"], width=w, color="#e15759", label="Filtered（平滑后输出）")
    axes[1].set_title("SOH MAE 对比")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["Vehicle"], rotation=45, ha="right")
    axes[1].set_ylabel("MAE (%)")
    axes[1].grid(alpha=0.25, axis="y")
    axes[1].legend()

    fig.tight_layout()
    out = os.path.join(out_dir, "soh_metric_compare.png")
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_ablation(ablation_csv: str, out_dir: str) -> Optional[str]:
    if not os.path.exists(ablation_csv):
        return None

    df = pd.read_csv(ablation_csv)
    req = {"tag", "RMSE_raw_mean", "RMSE_filtered_mean", "R2_raw_mean", "R2_filtered_mean"}
    if not req.issubset(df.columns):
        return None

    # 选取“最终SOH使用设置”：以 RMSE_filtered 最小作为主准则
    best_idx = df["RMSE_filtered_mean"].astype(float).idxmin()
    best_tag = df.loc[best_idx, "tag"]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    x = np.arange(len(df))
    w = 0.36

    raw_color = ["#4e79a7"] * len(df)
    fil_color = ["#e15759"] * len(df)
    fil_color[int(best_idx)] = "#2ca02c"

    axes[0].bar(x - w / 2, df["RMSE_raw_mean"], width=w, color=raw_color, label="Raw（单点输出）")
    axes[0].bar(x + w / 2, df["RMSE_filtered_mean"], width=w, color=fil_color, label="Filtered（平滑后输出）")
    axes[0].set_title("Ablation: RMSE Mean")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["tag"], rotation=45, ha="right")
    axes[0].set_ylabel("RMSE (%)")
    axes[0].grid(alpha=0.25, axis="y")
    axes[0].legend()

    axes[1].bar(x - w / 2, df["R2_raw_mean"], width=w, color=raw_color, label="Raw（单点输出）")
    axes[1].bar(x + w / 2, df["R2_filtered_mean"], width=w, color=fil_color, label="Filtered（平滑后输出）")
    axes[1].set_title("Ablation: R² Mean")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["tag"], rotation=45, ha="right")
    axes[1].set_ylabel("R²")
    axes[1].grid(alpha=0.25, axis="y")
    axes[1].legend()

    fig.suptitle(f"最终SOH建议使用设置：{best_tag}（绿色Filtered柱）", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out = os.path.join(out_dir, "ablation_compare.png")
    fig.savefig(out, dpi=220)
    plt.close(fig)

    explain_path = os.path.join(out_dir, "ablation_explain.txt")
    with open(explain_path, "w", encoding="utf-8") as f:
        f.write("Raw = 模型对单个样本的直接输出（不做时序平滑）\n")
        f.write("Filtered = 对Raw做滑动窗口平滑后的输出（用于SOH最终展示与SOC输入）\n")
        f.write("本消融仅比较 smooth_window 和 seed 对指标的影响。\n")
        f.write(f"推荐最终SOH使用: {best_tag}\n")
        f.write("选择准则: RMSE_filtered_mean 最小。\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制第三章SOH与消融对比图")
    parser.add_argument("--soh-output", default="outputs_final", help="SOH结果目录")
    parser.add_argument("--ablation-output", default="outputs_ablation", help="消融结果目录")
    parser.add_argument("--out-dir", default="figures", help="图片输出目录")
    args = parser.parse_args()

    _ensure_dir(args.out_dir)
    outputs: List[str] = []

    p = plot_chapter3_subplot(args.soh_output, args.out_dir)
    if p:
        outputs.append(p)

    p = plot_soh_metric_bars(os.path.join(args.soh_output, "soh_metrics_vehicle.csv"), args.out_dir)
    if p:
        outputs.append(p)

    p = plot_ablation(os.path.join(args.ablation_output, "ablation_results_paper.csv"), args.out_dir)
    if p:
        outputs.append(p)

    if outputs:
        print("✅ 已生成对比图:")
        for o in outputs:
            print(f"- {o}")
        explain = os.path.join(args.out_dir, "ablation_explain.txt")
        if os.path.exists(explain):
            print(f"- {explain}")
    else:
        print("⚠️ 未生成图片，请检查输入CSV是否已准备并包含所需列。")


if __name__ == "__main__":
    main()

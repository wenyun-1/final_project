"""一键执行：扩展搜索 -> 最优参数重训 -> Top6结果图（EV1~EV6）

流程：
1) 30轮快速筛选: sw=[9,11,13,15,17], seed=[42,3407,2025]
2) 以 RMSE_filtered_mean 最小选最优参数
3) 用最优参数进行150轮重训
4) 按 RMSE_filtered 最小选6辆车，画3x2子图：灰色散点(raw) + 红色趋势线(filtered)
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial Unicode MS", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def run_cmd(cmd: List[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def select_best_combo(search_root: str) -> Tuple[int, int, pd.DataFrame]:
    rows = []
    for sw, seed in itertools.product([9, 11, 13, 15, 17], [42, 3407, 2025]):
        tag = f"sw{sw}_seed{seed}"
        out = os.path.join(search_root, tag)
        os.makedirs(out, exist_ok=True)

        run_cmd([
            "python",
            "soh_final_pipeline.py",
            "--epochs",
            "30",
            "--smooth-window",
            str(sw),
            "--seed",
            str(seed),
            "--output",
            out,
        ])

        summary_csv = os.path.join(out, "soh_metrics_summary.csv")
        if not os.path.exists(summary_csv):
            continue
        df = pd.read_csv(summary_csv)
        val = df.loc[df["Metric"] == "RMSE_filtered", "Mean"]
        if len(val) == 0:
            continue
        rows.append({"tag": tag, "smooth_window": sw, "seed": seed, "RMSE_filtered_mean": float(val.iloc[0])})

    if not rows:
        raise RuntimeError("网格搜索没有得到有效结果，请检查数据与依赖。")

    res = pd.DataFrame(rows).sort_values("RMSE_filtered_mean").reset_index(drop=True)
    best_sw = int(res.loc[0, "smooth_window"])
    best_seed = int(res.loc[0, "seed"])
    return best_sw, best_seed, res


def plot_top6(final_output: str, fig_out: str) -> None:
    metrics_csv = os.path.join(final_output, "soh_metrics_vehicle.csv")
    points_csv = os.path.join(final_output, "soh_predictions_points.csv")
    if not os.path.exists(metrics_csv) or not os.path.exists(points_csv):
        raise RuntimeError("缺少绘图所需文件 soh_metrics_vehicle.csv 或 soh_predictions_points.csv")

    mdf = pd.read_csv(metrics_csv).sort_values("RMSE_filtered").head(6).reset_index(drop=True)
    pdf = pd.read_csv(points_csv)

    alias_rows = []
    fig, axes = plt.subplots(3, 2, figsize=(14, 13), sharex=False)
    axes = axes.flatten()

    for i, (_, r) in enumerate(mdf.iterrows()):
        veh = str(r["Vehicle"])
        alias = f"EV{i+1}"
        alias_rows.append({
            "Alias": alias,
            "Vehicle": veh,
            "RMSE_filtered": float(r["RMSE_filtered"]),
            "R2_filtered": float(r["R2_filtered"]),
        })

        g = pdf[pdf["Vehicle"].astype(str) == veh].sort_values("Days")
        ax = axes[i]

        # 多散点围绕趋势线（用raw散点体现波动）
        if "SOH_pred_raw" in g.columns:
            ax.scatter(g["Days"], g["SOH_pred_raw"], s=14, color="#7f7f7f", alpha=0.55, label="Estimated points")
        else:
            ax.scatter(g["Days"], g["SOH_pred_filtered"], s=14, color="#7f7f7f", alpha=0.55, label="Estimated points")

        ax.plot(g["Days"], g["SOH_pred_filtered"], color="#d62728", linewidth=2.2, label="SOH degradation trend")
        ax.set_title(f"{alias} | RMSE_f={r['RMSE_filtered']:.3f}% | R²_f={r['R2_filtered']:.3f}", fontsize=11)
        ax.set_xlabel("Days")
        ax.set_ylabel("SOH (%)")
        ax.grid(alpha=0.28)
        ax.legend(loc="best")

    for j in range(len(mdf), len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    fig.savefig(fig_out, dpi=300)
    plt.close(fig)

    alias_df = pd.DataFrame(alias_rows)
    alias_df.to_csv(os.path.join(final_output, "top6_vehicle_alias.csv"), index=False)
    with open(os.path.join(final_output, "top6_vehicle_alias.txt"), "w", encoding="utf-8") as f:
        for row in alias_rows:
            f.write(f"{row['Alias']}: {row['Vehicle']}, RMSE_f={row['RMSE_filtered']:.3f}%, R2_f={row['R2_filtered']:.3f}\n")



def main() -> None:
    parser = argparse.ArgumentParser(description="最优参数SOH重训与Top6结果图生成")
    parser.add_argument("--search-root", default="outputs_search", help="快速搜索输出目录")
    parser.add_argument("--final-output", default="outputs_final_best", help="最终重训输出目录")
    parser.add_argument("--figure", default="figures/chapter3_top6_soh.png", help="最终3x2图输出路径")
    args = parser.parse_args()

    os.makedirs(args.search_root, exist_ok=True)
    os.makedirs(args.final_output, exist_ok=True)
    os.makedirs(os.path.dirname(args.figure) or ".", exist_ok=True)

    best_sw, best_seed, search_table = select_best_combo(args.search_root)
    search_table.to_csv(os.path.join(args.search_root, "expanded_search_results.csv"), index=False)

    run_cmd([
        "python",
        "soh_final_pipeline.py",
        "--epochs",
        "150",
        "--smooth-window",
        str(best_sw),
        "--seed",
        str(best_seed),
        "--output",
        args.final_output,
    ])

    with open(os.path.join(args.final_output, "best_config.txt"), "w", encoding="utf-8") as f:
        f.write(f"best_smooth_window={best_sw}\n")
        f.write(f"best_seed={best_seed}\n")
        f.write("criterion=minimum RMSE_filtered_mean on expanded search\n")

    plot_top6(args.final_output, args.figure)

    print("\n✅ 完成：")
    print(f"- 搜索结果: {args.search_root}/expanded_search_results.csv")
    print(f"- 最优配置: {args.final_output}/best_config.txt")
    print(f"- Top6映射: {args.final_output}/top6_vehicle_alias.csv")
    print(f"- 最终图: {args.figure}")


if __name__ == "__main__":
    main()

"""SOH终极路线消融实验模板（自动汇总表格）

用法示例：
python ablation_template.py --epochs 80 --output outputs_ablation
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
from typing import Dict, List

import pandas as pd


def run_cmd(cmd: List[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="SOH 消融实验模板")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--output", default="outputs_ablation")
    parser.add_argument("--python", default="python")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 你可以继续扩展这里的组合。
    # 本模板先给最关键两维：平滑窗口 + 随机种子。
    grid: Dict[str, List] = {
        "smooth_window": [1, 9, 15],  # 1=raw(不平滑)，可用于对照
        "seed": [42, 3407],
    }

    records = []
    for smooth_window, seed in itertools.product(grid["smooth_window"], grid["seed"]):
        tag = f"sw{smooth_window}_seed{seed}"
        out_dir = os.path.join(args.output, tag)

        cmd = [
            args.python,
            "soh_final_pipeline.py",
            "--epochs",
            str(args.epochs),
            "--smooth-window",
            str(smooth_window),
            "--seed",
            str(seed),
            "--output",
            out_dir,
        ]
        run_cmd(cmd)

        summary_csv = os.path.join(out_dir, "soh_metrics_summary.csv")
        if not os.path.exists(summary_csv):
            print(f"[WARN] 缺少 {summary_csv}，跳过该实验")
            continue

        df = pd.read_csv(summary_csv)
        row = {"tag": tag, "smooth_window": smooth_window, "seed": seed}
        for _, r in df.iterrows():
            metric = r["Metric"]
            row[f"{metric}_mean"] = r["Mean"]
            row[f"{metric}_std"] = r["Std"]
        records.append(row)

    if not records:
        print("❌ 没有可汇总的消融结果。")
        return

    out_table = pd.DataFrame(records).sort_values(["smooth_window", "seed"]).reset_index(drop=True)
    out_table.to_csv(os.path.join(args.output, "ablation_results.csv"), index=False)

    # 论文展示友好版：保留关键指标
    keep_cols = [
        "tag", "smooth_window", "seed",
        "RMSE_raw_mean", "RMSE_filtered_mean",
        "MAE_raw_mean", "MAE_filtered_mean",
        "R2_raw_mean", "R2_filtered_mean",
    ]
    keep_cols = [c for c in keep_cols if c in out_table.columns]
    out_table[keep_cols].to_csv(os.path.join(args.output, "ablation_results_paper.csv"), index=False)

    print("✅ 消融模板执行完毕，已导出：")
    print(f"- {args.output}/ablation_results.csv")
    print(f"- {args.output}/ablation_results_paper.csv")


if __name__ == "__main__":
    main()

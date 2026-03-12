"""按“电压窗口 × 学习率”批量实验并汇总 MAE/RMSE。

默认执行：
- 窗口A: V_START=538.0, V_END=558.0
- 窗口B: V_START=540.0, V_END=555.0
- 学习率: 1e-4, 3e-4, 5e-4

并固定测试车为两辆（默认 LFP604EV3 与 LFP604EV9，排除 EV10）。
"""

from __future__ import annotations

import argparse
import os
import subprocess
from typing import List, Tuple

import pandas as pd


WINDOWS: List[Tuple[float, float, str]] = [
    (538.0, 558.0, "A_538_558"),
    (540.0, 555.0, "B_540_555"),
]
LRS = [1e-4, 3e-4, 5e-4]


def run_cmd(cmd: List[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def read_summary(summary_csv: str) -> dict:
    df = pd.read_csv(summary_csv)
    out = {}
    for metric in ["RMSE_raw", "MAE_raw", "RMSE_filtered", "MAE_filtered"]:
        val = df.loc[df["Metric"] == metric, "Mean"]
        out[metric] = float(val.iloc[0]) if len(val) > 0 else float("nan")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="电压窗口与学习率组合实验")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smooth-window", type=int, default=15)
    p.add_argument("--base-output", default="outputs_window_lr_sweep")
    p.add_argument("--test-vehicles", nargs="*", default=["LFP604EV3", "LFP604EV9"], help="固定2辆测试车，默认排除EV10")
    p.add_argument("--train-vehicle-count", type=int, default=9)
    p.add_argument("--test-vehicle-count", type=int, default=2)
    args = p.parse_args()

    os.makedirs(args.base_output, exist_ok=True)

    rows = []
    for v_start, v_end, tag in WINDOWS:
        for lr in LRS:
            lr_tag = f"lr_{lr:.0e}".replace("-", "m")
            out_dir = os.path.join(args.base_output, f"{tag}_{lr_tag}")
            os.makedirs(out_dir, exist_ok=True)

            cmd = [
                "python",
                "soh_final_pipeline.py",
                "--output",
                out_dir,
                "--epochs",
                str(args.epochs),
                "--seed",
                str(args.seed),
                "--smooth-window",
                str(args.smooth_window),
                "--v-start",
                str(v_start),
                "--v-end",
                str(v_end),
                "--learning-rate",
                str(lr),
                "--train-vehicle-count",
                str(args.train_vehicle_count),
                "--test-vehicle-count",
                str(args.test_vehicle_count),
                "--test-vehicles",
                *args.test_vehicles,
            ]
            run_cmd(cmd)

            summary_csv = os.path.join(out_dir, "soh_metrics_summary.csv")
            if not os.path.exists(summary_csv):
                print(f"[WARN] 未找到 {summary_csv}")
                continue

            metrics = read_summary(summary_csv)
            rows.append(
                {
                    "window": f"{v_start:.1f}-{v_end:.1f}",
                    "learning_rate": lr,
                    **metrics,
                    "output_dir": out_dir,
                }
            )

    if not rows:
        raise RuntimeError("没有获得有效结果，请检查数据与运行日志。")

    res = pd.DataFrame(rows).sort_values(["RMSE_filtered", "MAE_filtered"]).reset_index(drop=True)
    out_csv = os.path.join(args.base_output, "window_lr_comparison.csv")
    res.to_csv(out_csv, index=False)

    best = res.iloc[0]
    print("\n✅ 对比完成")
    print(f"- 汇总结果: {out_csv}")
    print(
        f"- 最优组合: window={best['window']}, lr={best['learning_rate']}, "
        f"RMSE_f={best['RMSE_filtered']:.4f}, MAE_f={best['MAE_filtered']:.4f}"
    )


if __name__ == "__main__":
    main()

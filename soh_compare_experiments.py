"""对比实验脚本：
1) 保留当前伪标签策略，替换 train/test 车辆
2) 保持同一 split，替换伪标签策略
"""

from __future__ import annotations

import argparse
import copy
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from soh_final_pipeline import (
    Config,
    collect_files,
    export_feature_correlation,
    normalize_vehicle_name,
    set_seed,
    build_vehicle_frames,
    train_and_eval,
)


def _choose_alt_test_vehicles(available: List[str], baseline: List[str], n_test: int) -> List[str]:
    baseline_set = set(baseline)
    candidates = [v for v in available if v not in baseline_set]
    if len(candidates) >= n_test:
        return sorted(candidates[:n_test])

    rotated = [v for v in available if v in baseline_set] + candidates
    if len(rotated) < n_test:
        raise ValueError(f"可用车辆不足，无法构造替代测试集（需要{n_test}辆，当前{len(available)}辆）。")
    return sorted(rotated[:n_test])


def _run_one(name: str, cfg: Config, files: List[str], output_root: str) -> pd.DataFrame:
    out_dir = os.path.join(output_root, name)
    os.makedirs(out_dir, exist_ok=True)
    frames = build_vehicle_frames(files, cfg, out_dir)
    if not frames:
        raise RuntimeError(f"{name}: 无可用车辆样本")
    export_feature_correlation(frames, out_dir)
    metrics = train_and_eval(frames, cfg, out_dir)
    if metrics.empty:
        raise RuntimeError(f"{name}: 训练/评估无有效输出")
    return metrics


def _summarize(metrics: pd.DataFrame, name: str, split_desc: str, pseudo_method: str) -> Dict:
    return {
        "experiment": name,
        "split": split_desc,
        "pseudo_label_method": pseudo_method,
        "n_test_vehicle": int(metrics["Vehicle"].nunique()),
        "RMSE_raw_mean": float(metrics["RMSE_raw"].mean()),
        "MAE_raw_mean": float(metrics["MAE_raw"].mean()),
        "R2_raw_mean": float(metrics["R2_raw"].mean()),
        "RMSE_filtered_mean": float(metrics["RMSE_filtered"].mean()),
        "MAE_filtered_mean": float(metrics["MAE_filtered"].mean()),
        "R2_filtered_mean": float(metrics["R2_filtered"].mean()),
    }


def _plot_compare(df: pd.DataFrame, out_png: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    x = range(len(df))

    axes[0].bar(x, df["RMSE_filtered_mean"], color=["#4e79a7", "#f28e2b", "#59a14f"])
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(df["experiment"], rotation=15)
    axes[0].set_title("Filtered RMSE 对比")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(alpha=0.25, axis="y")

    axes[1].bar(x, df["R2_filtered_mean"], color=["#4e79a7", "#f28e2b", "#59a14f"])
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(df["experiment"], rotation=15)
    axes[1].set_title("Filtered R2 对比")
    axes[1].set_ylabel("R2")
    axes[1].grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="SOH对比实验：换split + 换伪标签")
    parser.add_argument("--output-root", default="outputs_compare")
    parser.add_argument("--data-dirs", nargs="+", default=["data"])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smooth-window", type=int, default=15)
    parser.add_argument("--read-chunk-size", type=int, default=200000)
    parser.add_argument("--baseline-test-vehicles", nargs="*", default=["LFP604EV3", "LFP604EV10", "LFP604EV9"])
    parser.add_argument("--alt-test-vehicles", nargs="*", default=None)
    parser.add_argument("--alt-pseudo-method", choices=["rolling_monotone", "isotonic_monotone"], default="isotonic_monotone")
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    files = collect_files(args.data_dirs)
    if not files:
        raise FileNotFoundError(f"未找到CSV数据，请检查目录: {args.data_dirs}")

    base_cfg = Config(
        epochs=args.epochs,
        seed=args.seed,
        smooth_window=args.smooth_window,
        split_mode="cross_vehicle",
        train_vehicle_count=9,
        test_vehicle_count=3,
        fixed_test_vehicles=[normalize_vehicle_name(v) for v in args.baseline_test_vehicles],
        read_chunk_size=args.read_chunk_size,
        pseudo_label_method="robust_linear",
    )
    set_seed(base_cfg.seed)

    baseline_metrics = _run_one("baseline", base_cfg, files, args.output_root)

    baseline_frames = build_vehicle_frames(files, base_cfg, os.path.join(args.output_root, "_probe"))
    all_vehicles = sorted(baseline_frames.keys())
    n_test = len(base_cfg.fixed_test_vehicles)

    if args.alt_test_vehicles:
        alt_test = [normalize_vehicle_name(v) for v in args.alt_test_vehicles]
    else:
        alt_test = _choose_alt_test_vehicles(all_vehicles, base_cfg.fixed_test_vehicles, n_test)

    split_cfg = copy.deepcopy(base_cfg)
    split_cfg.fixed_test_vehicles = alt_test
    split_cfg.reuse_if_same_trainset = False
    split_metrics = _run_one("alt_split", split_cfg, files, args.output_root)

    pseudo_cfg = copy.deepcopy(base_cfg)
    pseudo_cfg.pseudo_label_method = args.alt_pseudo_method
    pseudo_cfg.reuse_if_same_trainset = False
    pseudo_metrics = _run_one("alt_pseudo", pseudo_cfg, files, args.output_root)

    rows = [
        _summarize(baseline_metrics, "baseline", f"test={base_cfg.fixed_test_vehicles}", base_cfg.pseudo_label_method),
        _summarize(split_metrics, "alt_split", f"test={alt_test}", split_cfg.pseudo_label_method),
        _summarize(pseudo_metrics, "alt_pseudo", f"test={base_cfg.fixed_test_vehicles}", pseudo_cfg.pseudo_label_method),
    ]
    cmp_df = pd.DataFrame(rows)
    cmp_csv = os.path.join(args.output_root, "comparison_metrics.csv")
    cmp_df.to_csv(cmp_csv, index=False)
    _plot_compare(cmp_df, os.path.join(args.output_root, "comparison_metrics.png"))
    print("✅ 对比实验完成：")
    print(f"- {cmp_csv}")
    print(f"- {os.path.join(args.output_root, 'comparison_metrics.png')}")


if __name__ == "__main__":
    main()

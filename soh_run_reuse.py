"""快速运行入口：训练集不变时复用已有 SOH 模型并跳过训练。"""

from __future__ import annotations

import argparse
import subprocess


def main() -> None:
    p = argparse.ArgumentParser(description="SOH 快速运行（复用模型）")
    p.add_argument("--output", default="outputs_final")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--smooth-window", type=int, default=15)
    p.add_argument("--data-dirs", nargs="+", default=["data"])
    p.add_argument("--train-vehicle-count", type=int, default=10)
    p.add_argument("--test-vehicle-count", type=int, default=2)
    p.add_argument("--read-chunk-size", type=int, default=200000)
    args = p.parse_args()

    cmd = [
        "python", "soh_final_pipeline.py",
        "--output", args.output,
        "--epochs", str(args.epochs),
        "--smooth-window", str(args.smooth_window),
        "--train-vehicle-count", str(args.train_vehicle_count),
        "--test-vehicle-count", str(args.test_vehicle_count),
        "--read-chunk-size", str(args.read_chunk_size),
        "--reuse-if-same-trainset",
        "--data-dirs", *args.data_dirs,
    ]
    print("🚀 快速运行命令:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

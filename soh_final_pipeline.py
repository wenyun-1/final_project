"""简化 SOH 估计管线（12车 / SOC 70-95 / 10训2测 / PI-UAE）"""

from __future__ import annotations

import argparse
import glob
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import interp1d
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import ConcatDataset, DataLoader, Dataset

TIME_FORMAT = "mixed"
DATA_DIRS = ["data"]
DEFAULT_OUTPUT_DIR = "outputs_final"

# 仅保留任务要求的 SOC 筛选窗口
SOC_WINDOW_START = 70.0
SOC_WINDOW_END = 95.0

# 仅使用 12 辆车：10 训练 + 2 测试
TOTAL_VEHICLES = 12
TRAIN_VEHICLES = 10
TEST_VEHICLES = 2


@dataclass
class Config:
    batch_size: int = 32
    epochs: int = 80
    learning_rate: float = 1e-3
    lambda_recon: float = 0.5
    seed: int = 42
    min_seg_points: int = 30
    max_gap_seconds: int = 60
    min_soc_delta: float = 20.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_files(data_dirs: List[str]) -> List[str]:
    files: List[str] = []
    for d in data_dirs:
        files.extend(glob.glob(os.path.join(d, "*.csv")))
    return sorted(set(files))


def read_vehicle_csv(path: str) -> pd.DataFrame:
    sample = pd.read_csv(path, nrows=5)
    use_cols = ["DATA_TIME", "totalCurrent", "totalVoltage", "SOC"]
    if "maxTemperature" in sample.columns:
        use_cols.append("maxTemperature")

    df = pd.read_csv(path, usecols=use_cols, low_memory=False)
    df["DATA_TIME"] = pd.to_datetime(df["DATA_TIME"], format=TIME_FORMAT, errors="coerce")
    for c in ["totalCurrent", "totalVoltage", "SOC"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "maxTemperature" in df.columns:
        df["maxTemperature"] = pd.to_numeric(df["maxTemperature"], errors="coerce")
    else:
        df["maxTemperature"] = 25.0

    return df.dropna().sort_values("DATA_TIME").reset_index(drop=True)


def extract_charge_segments(df: pd.DataFrame, cfg: Config) -> List[Dict]:
    """提取充电片段，并在每个片段中截取 SOC 70->95 计算容量。"""
    segments: List[List[Dict]] = []
    curr: List[Dict] = []

    for row in df.to_dict("records"):
        # 与原脚本一致：totalCurrent < -1 视为充电
        if row["totalCurrent"] < -1.0:
            curr.append(row)
            continue

        if len(curr) > cfg.min_seg_points:
            segments.append(curr)
        curr = []

    if len(curr) > cfg.min_seg_points:
        segments.append(curr)

    out: List[Dict] = []
    for seg in segments:
        df_seg = pd.DataFrame(seg)
        soc_vals = df_seg["SOC"].to_numpy(dtype=float)

        idx_s = int(np.argmin(np.abs(soc_vals - SOC_WINDOW_START)))
        idx_e = int(np.argmin(np.abs(soc_vals - SOC_WINDOW_END)))
        if idx_e <= idx_s:
            continue

        df_sub = df_seg.iloc[idx_s: idx_e + 1].copy()
        if len(df_sub) <= 10:
            continue

        dt = df_sub["DATA_TIME"].diff().dt.total_seconds().fillna(10)
        if dt.max() > cfg.max_gap_seconds:
            continue

        soc_delta = float(df_sub["SOC"].iloc[-1] - df_sub["SOC"].iloc[0])
        if soc_delta < cfg.min_soc_delta:
            continue

        curr_abs = df_sub["totalCurrent"].abs()
        ah = (curr_abs * dt).sum() / 3600.0
        raw_cap = ah / (soc_delta / 100.0)

        v_seq = df_sub["totalVoltage"].to_numpy(dtype=float)
        f_interp = interp1d(np.linspace(0, 1, len(v_seq)), v_seq, kind="linear")
        fp_raw = f_interp(np.linspace(0, 1, 100))
        fp_min, fp_max = float(np.min(fp_raw)), float(np.max(fp_raw))
        fingerprint = (fp_raw - fp_min) / max(fp_max - fp_min, 1e-6)

        out.append(
            {
                "days": int((df_sub["DATA_TIME"].iloc[0] - pd.Timestamp("2020-01-01")).days),
                "raw_cap": float(raw_cap),
                "fingerprint": fingerprint,
                "avg_curr": float(curr_abs.mean()),
                "avg_temp": float(df_sub["maxTemperature"].mean()),
            }
        )

    return out


def build_soh_labels(segments: List[Dict]) -> pd.DataFrame:
    """用每个充电片段容量计算 SOH 标签。"""
    df = pd.DataFrame(segments)
    if df.empty:
        return df

    df = df.sort_values("days").reset_index(drop=True)

    # 以早期片段容量均值作为基准容量，得到 SOH 百分比
    n_ref = max(3, min(10, len(df) // 5 if len(df) >= 5 else len(df)))
    baseline_cap = float(df["raw_cap"].iloc[:n_ref].mean())
    baseline_cap = max(baseline_cap, 1e-6)

    df["baseline_cap"] = baseline_cap
    df["soh_true"] = (df["raw_cap"] / baseline_cap * 100.0).clip(lower=0.0, upper=120.0)
    return df


class SOHDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, mean: np.ndarray, std: np.ndarray):
        self.rows = frame.sort_values("days").reset_index(drop=True).to_dict("records")
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        sc = (np.array([r["avg_curr"], r["avg_temp"]]) - self.mean) / self.std
        y = r["soh_true"] / 100.0
        return (
            torch.tensor(r["fingerprint"], dtype=torch.float32).unsqueeze(0),
            torch.tensor(sc, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            r["days"],
        )


class PIUAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 3, 1, 1), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, 1, 1), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Flatten(), nn.Linear(32 * 25, 64), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 32 * 25), nn.ReLU(), nn.Unflatten(1, (32, 25)),
            nn.Upsample(scale_factor=2), nn.Conv1d(32, 16, 3, 1, 1), nn.ReLU(),
            nn.Upsample(scale_factor=2), nn.Conv1d(16, 1, 3, 1, 1), nn.Sigmoid(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(64 + 2, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, fp, sc):
        feat = self.encoder(fp)
        pred = self.regressor(torch.cat((feat, sc), dim=1))
        recon = self.decoder(feat)
        return pred, recon


def split_vehicles(files: List[str]) -> Tuple[List[str], List[str]]:
    if len(files) < TOTAL_VEHICLES:
        raise ValueError(f"车辆文件不足 {TOTAL_VEHICLES} 个，当前仅 {len(files)} 个")

    selected = files[:TOTAL_VEHICLES]
    train_files = selected[:TRAIN_VEHICLES]
    test_files = selected[TRAIN_VEHICLES:TRAIN_VEHICLES + TEST_VEHICLES]
    return train_files, test_files


def build_vehicle_frames(files: List[str], cfg: Config, output_dir: str) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}

    for path in files:
        veh = os.path.splitext(os.path.basename(path))[0]
        try:
            df = read_vehicle_csv(path)
            segments = extract_charge_segments(df, cfg)
            labeled = build_soh_labels(segments)
            if labeled.empty:
                print(f"[WARN] {veh} 无有效片段，已跳过")
                continue
            frames[veh] = labeled
            labeled[["days", "raw_cap", "baseline_cap", "soh_true"]].to_csv(
                os.path.join(output_dir, f"{veh}_segments_soh.csv"), index=False
            )
        except Exception as e:
            print(f"[WARN] {veh} 处理失败: {e}")

    return frames


def compute_train_scaler(train_frames: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
    rows = []
    for frame in train_frames.values():
        rows.extend(frame[["avg_curr", "avg_temp"]].to_dict("records"))

    if not rows:
        raise ValueError("训练数据为空，无法计算标准化参数")

    arr = np.array([[r["avg_curr"], r["avg_temp"]] for r in rows], dtype=float)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0) + 1e-6
    return mean, std


def train_and_eval(
    train_frames: Dict[str, pd.DataFrame],
    test_frames: Dict[str, pd.DataFrame],
    cfg: Config,
    output_dir: str,
) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean, std = compute_train_scaler(train_frames)
    np.savez(os.path.join(output_dir, "train_scaler.npz"), mean=mean, std=std)

    train_datasets = [SOHDataset(frame, mean, std) for frame in train_frames.values() if len(frame) > 0]
    if not train_datasets:
        raise ValueError("训练集无可用样本")

    train_loader = DataLoader(
        ConcatDataset(train_datasets),
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    model = PIUAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    mse = nn.MSELoss()

    for _ in range(cfg.epochs):
        model.train()
        for fp, sc, y, _ in train_loader:
            fp, sc, y = fp.to(device), sc.to(device), y.to(device)
            y_pred, recon = model(fp, sc)
            loss = mse(y_pred.squeeze(), y) + cfg.lambda_recon * mse(recon, fp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), os.path.join(output_dir, "pi_uae_soh_model.pth"))

    all_metrics = []
    all_points = []
    model.eval()
    with torch.no_grad():
        for veh, frame in test_frames.items():
            ds = SOHDataset(frame, mean, std)
            if len(ds) == 0:
                continue
            loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)

            days, y_true, y_pred = [], [], []
            for fp, sc, y, d in loader:
                p, _ = model(fp.to(device), sc.to(device))
                y_pred.extend((p.cpu().numpy().flatten() * 100.0).tolist())
                y_true.extend((y.numpy() * 100.0).tolist())
                days.extend(d.numpy().tolist())

            idx = np.argsort(days)
            days = np.array(days)[idx]
            y_true = np.array(y_true)[idx]
            y_pred = np.array(y_pred)[idx]

            metrics = {
                "Vehicle": veh,
                "N_test": int(len(y_true)),
                "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "MAE": float(mean_absolute_error(y_true, y_pred)),
                "R2": float(r2_score(y_true, y_pred)),
            }
            all_metrics.append(metrics)

            for d, yt, yp in zip(days, y_true, y_pred):
                all_points.append(
                    {
                        "Vehicle": veh,
                        "Days": int(d),
                        "SOH_true": float(yt),
                        "SOH_pred": float(yp),
                    }
                )

    metrics_df = pd.DataFrame(all_metrics)
    points_df = pd.DataFrame(all_points)

    if not metrics_df.empty:
        metrics_df.to_csv(os.path.join(output_dir, "soh_metrics_vehicle.csv"), index=False)
        summary = pd.DataFrame(
            {
                "Metric": ["RMSE", "MAE", "R2"],
                "Mean": [metrics_df["RMSE"].mean(), metrics_df["MAE"].mean(), metrics_df["R2"].mean()],
                "Std": [
                    metrics_df["RMSE"].std(ddof=1) if len(metrics_df) > 1 else 0.0,
                    metrics_df["MAE"].std(ddof=1) if len(metrics_df) > 1 else 0.0,
                    metrics_df["R2"].std(ddof=1) if len(metrics_df) > 1 else 0.0,
                ],
            }
        )
        summary.to_csv(os.path.join(output_dir, "soh_metrics_summary.csv"), index=False)

    if not points_df.empty:
        points_df.to_csv(os.path.join(output_dir, "soh_predictions_points.csv"), index=False)

    return metrics_df


def main() -> None:
    parser = argparse.ArgumentParser(description="简化 SOH 估计脚本（12车 / SOC70-95 / 10训2测）")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Config(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
    set_seed(cfg.seed)

    os.makedirs(args.output, exist_ok=True)

    files = collect_files(DATA_DIRS)
    if not files:
        print("❌ 未找到数据文件，请将 CSV 放入 data/")
        return

    try:
        train_files, test_files = split_vehicles(files)
    except ValueError as e:
        print(f"❌ {e}")
        return

    print("使用车辆：")
    print("- 训练集(10):", ", ".join(os.path.splitext(os.path.basename(f))[0] for f in train_files))
    print("- 测试集(2):", ", ".join(os.path.splitext(os.path.basename(f))[0] for f in test_files))

    train_frames = build_vehicle_frames(train_files, cfg, args.output)
    test_frames = build_vehicle_frames(test_files, cfg, args.output)

    if len(train_frames) < TRAIN_VEHICLES:
        print(f"❌ 训练车辆有效样本不足：期望 {TRAIN_VEHICLES}，实际 {len(train_frames)}")
        return
    if len(test_frames) < TEST_VEHICLES:
        print(f"❌ 测试车辆有效样本不足：期望 {TEST_VEHICLES}，实际 {len(test_frames)}")
        return

    metrics = train_and_eval(train_frames, test_frames, cfg, args.output)
    if metrics.empty:
        print("❌ 未生成有效评估结果")
        return

    print("✅ 训练与评估完成，已导出：")
    print(f"- {args.output}/soh_metrics_vehicle.csv")
    print(f"- {args.output}/soh_metrics_summary.csv")
    print(f"- {args.output}/soh_predictions_points.csv")


if __name__ == "__main__":
    main()

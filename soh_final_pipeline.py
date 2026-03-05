"""SOH 终极路线：单脚本可复现实验管线

核心思想：
1) 伪标签：前20%数据的85分位容量作为基准 + 一阶稳健拟合 + SOH<=100%物理上限
2) 模型：PI-UAE(电压指纹 + 温度/电流标量)
3) 评估：raw/filtered 双指标 + 自动导出论文表格 + SOC接口文件
"""

from __future__ import annotations

import argparse
import glob
import os
import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import interp1d
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset

TIME_FORMAT = "mixed"
DATA_DIRS = ["data", "data1", "samples"]
DEFAULT_OUTPUT_DIR = "outputs_final"

# 电压窗口建议选在相对稳定恒流区，可按车型调参
V_START = 538.0
V_END = 558.0


@dataclass
class Config:
    batch_size: int = 32
    epochs: int = 120
    learning_rate: float = 5e-4
    lambda_recon: float = 0.5
    alpha_physics: float = 0.02
    smooth_window: int = 15
    seed: int = 42
    min_seg_points: int = 30
    max_gap_seconds: int = 60
    min_soc_delta: float = 20.0
    train_split_mod: int = 5


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


def extract_segments(df: pd.DataFrame, cfg: Config) -> List[Dict]:
    segs: List[Dict] = []
    curr: List[Dict] = []
    for row in df.to_dict("records"):
        if row["totalCurrent"] < -1.0:
            curr.append(row)
            continue
        if len(curr) > cfg.min_seg_points:
            segs.append({"records": curr})
        curr = []
    if len(curr) > cfg.min_seg_points:
        segs.append({"records": curr})

    out = []
    for s in segs:
        df_seg = pd.DataFrame(s["records"])
        if df_seg["totalVoltage"].min() >= V_START or df_seg["totalVoltage"].max() <= V_END:
            continue

        idx_s = (df_seg["totalVoltage"] - V_START).abs().idxmin()
        idx_e = (df_seg["totalVoltage"] - V_END).abs().idxmin()
        if idx_e <= idx_s:
            continue

        df_sub = df_seg.loc[idx_s:idx_e].copy()
        if len(df_sub) <= 10:
            continue

        dt = df_sub["DATA_TIME"].diff().dt.total_seconds().fillna(10)
        if dt.max() > cfg.max_gap_seconds:
            continue

        soc_delta = df_seg["SOC"].iloc[-1] - df_seg["SOC"].iloc[0]
        if soc_delta <= cfg.min_soc_delta:
            continue

        curr_abs = df_seg["totalCurrent"].abs()
        ah = (curr_abs * df_seg["DATA_TIME"].diff().dt.total_seconds().fillna(10)).sum() / 3600
        raw_cap = ah / (soc_delta / 100.0)

        v_seq = df_sub["totalVoltage"].values
        f_interp = interp1d(np.linspace(0, 1, len(v_seq)), v_seq, kind="linear")
        fingerprint = (f_interp(np.linspace(0, 1, 100)) - V_START) / (V_END - V_START)

        out.append(
            {
                "days": int((df_seg["DATA_TIME"].iloc[0] - pd.Timestamp("2020-01-01")).days),
                "raw_cap": float(raw_cap),
                "fingerprint": fingerprint,
                "avg_curr": float(df_sub["totalCurrent"].abs().mean()),
                "avg_temp": float(df_sub["maxTemperature"].mean()),
            }
        )
    return out


def build_pseudo_labels(rows: List[Dict], robust_linear: bool = True) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("days").reset_index(drop=True)
    q1, q99 = df["raw_cap"].quantile([0.01, 0.99])
    df["raw_cap_clip"] = df["raw_cap"].clip(lower=q1, upper=q99)

    # 终极路线：前20%样本的85分位作基准
    n_head = max(5, int(len(df) * 0.2))
    baseline_cap = float(np.percentile(df["raw_cap_clip"].iloc[:n_head], 85))

    # 一阶拟合抽取“本征衰减主趋势”
    z = np.polyfit(df["days"], df["raw_cap_clip"], 1)
    trend_cap = np.poly1d(z)(df["days"]) if robust_linear else df["raw_cap_clip"].values

    soh_true = (trend_cap / baseline_cap) * 100.0
    soh_true = np.clip(soh_true, 0.0, 100.0)

    df["baseline_cap"] = baseline_cap
    df["soh_true"] = soh_true
    return df


class SOHDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, is_train: bool, split_mod: int, scaler_path: str):
        all_rows = frame.to_dict("records")
        if is_train:
            rows = [r for i, r in enumerate(all_rows) if i % split_mod != 0]
        else:
            rows = [r for i, r in enumerate(all_rows) if i % split_mod == 0]

        if is_train:
            scalars = np.array([[r["avg_curr"], r["avg_temp"]] for r in rows])
            mean, std = scalars.mean(0), scalars.std(0) + 1e-6
            np.savez(scaler_path, mean=mean, std=std)
        else:
            stats = np.load(scaler_path)
            mean, std = stats["mean"], stats["std"]

        self.rows = rows
        self.mean, self.std = mean, std

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        prev = self.rows[max(0, idx - 1)]
        curr_sc = (np.array([r["avg_curr"], r["avg_temp"]]) - self.mean) / self.std
        prev_sc = (np.array([prev["avg_curr"], prev["avg_temp"]]) - self.mean) / self.std
        y = r["soh_true"] / 100.0
        return (
            torch.tensor(r["fingerprint"], dtype=torch.float32).unsqueeze(0),
            torch.tensor(curr_sc, dtype=torch.float32),
            torch.tensor(prev["fingerprint"], dtype=torch.float32).unsqueeze(0),
            torch.tensor(prev_sc, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            r["days"],
        )


class PIUAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 3, 1, 1), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, 1, 1), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Flatten(), nn.Linear(32 * 25, 64), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 32 * 25), nn.ReLU(), nn.Unflatten(1, (32, 25)),
            nn.Upsample(scale_factor=2), nn.Conv1d(32, 16, 3, 1, 1), nn.ReLU(),
            nn.Upsample(scale_factor=2), nn.Conv1d(16, 1, 3, 1, 1), nn.Sigmoid()
        )
        self.regressor = nn.Sequential(
            nn.Linear(64 + 2, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, fp, sc):
        feat = self.encoder(fp)
        pred = self.regressor(torch.cat((feat, sc), dim=1))
        recon = self.decoder(feat)
        return pred, recon


def train_and_eval(vehicle_frames: Dict[str, pd.DataFrame], cfg: Config, output_dir: str) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_metrics = []
    all_soc_export = []
    all_point_export = []

    for veh, frame in vehicle_frames.items():
        if len(frame) < 30:
            continue

        scaler_path = os.path.join(output_dir, f"{veh}_scaler.npz")
        model_path = os.path.join(output_dir, f"{veh}_pi_uae.pth")

        train_ds = SOHDataset(frame, is_train=True, split_mod=cfg.train_split_mod, scaler_path=scaler_path)
        test_ds = SOHDataset(frame, is_train=False, split_mod=cfg.train_split_mod, scaler_path=scaler_path)
        if len(train_ds) == 0 or len(test_ds) == 0:
            continue

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

        model = PIUAE().to(device)
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
        mse = nn.MSELoss()

        for _ in range(cfg.epochs):
            model.train()
            for c_fp, c_sc, p_fp, p_sc, y, _ in train_loader:
                c_fp, c_sc, p_fp, p_sc, y = c_fp.to(device), c_sc.to(device), p_fp.to(device), p_sc.to(device), y.to(device)
                y_pred, recon = model(c_fp, c_sc)
                y_prev, _ = model(p_fp, p_sc)
                loss = mse(y_pred.squeeze(), y) + cfg.lambda_recon * mse(recon, c_fp) + cfg.alpha_physics * torch.relu(y_pred.squeeze() - y_prev.squeeze()).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        torch.save(model.state_dict(), model_path)

        model.eval()
        days, y_true, y_pred = [], [], []
        with torch.no_grad():
            for c_fp, c_sc, _, _, y, d in test_loader:
                p, _ = model(c_fp.to(device), c_sc.to(device))
                y_pred.extend((p.cpu().numpy().flatten() * 100).tolist())
                y_true.extend((y.numpy() * 100).tolist())
                days.extend(d.numpy().tolist())

        idx = np.argsort(days)
        days = np.array(days)[idx]
        y_true = np.array(y_true)[idx]
        y_pred = np.array(y_pred)[idx]
        y_pred_filtered = pd.Series(y_pred).rolling(window=cfg.smooth_window, min_periods=1, center=True).mean().values

        metrics = {
            "Vehicle": veh,
            "N_test": len(y_true),
            "RMSE_raw": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE_raw": float(mean_absolute_error(y_true, y_pred)),
            "R2_raw": float(r2_score(y_true, y_pred)),
            "RMSE_filtered": float(np.sqrt(mean_squared_error(y_true, y_pred_filtered))),
            "MAE_filtered": float(mean_absolute_error(y_true, y_pred_filtered)),
            "R2_filtered": float(r2_score(y_true, y_pred_filtered)),
        }
        all_metrics.append(metrics)

        for d, yt, yr, yf in zip(days, y_true, y_pred, y_pred_filtered):
            all_point_export.append({
                "Vehicle": veh,
                "Days": int(d),
                "SOH_true": float(yt),
                "SOH_pred_raw": float(yr),
                "SOH_pred_filtered": float(yf),
            })
            all_soc_export.append({"Vehicle": veh, "Days": int(d), "Pred_SOH": float(yf / 100.0)})

    metric_df = pd.DataFrame(all_metrics)
    if not metric_df.empty:
        metric_df.to_csv(os.path.join(output_dir, "soh_metrics_vehicle.csv"), index=False)
        summary = {
            "Metric": ["RMSE_raw", "MAE_raw", "R2_raw", "RMSE_filtered", "MAE_filtered", "R2_filtered"],
            "Mean": [metric_df[c].mean() for c in ["RMSE_raw", "MAE_raw", "R2_raw", "RMSE_filtered", "MAE_filtered", "R2_filtered"]],
            "Std": [metric_df[c].std(ddof=1) if len(metric_df) > 1 else 0.0 for c in ["RMSE_raw", "MAE_raw", "R2_raw", "RMSE_filtered", "MAE_filtered", "R2_filtered"]],
        }
        pd.DataFrame(summary).to_csv(os.path.join(output_dir, "soh_metrics_summary.csv"), index=False)

    point_df = pd.DataFrame(all_point_export)
    if not point_df.empty:
        point_df.to_csv(os.path.join(output_dir, "soh_predictions_points.csv"), index=False)

    soc_df = pd.DataFrame(all_soc_export)
    if not soc_df.empty:
        soc_df.to_csv(os.path.join(output_dir, "SOH_Predictions_For_SOC.csv"), index=False)

    return metric_df


def build_vehicle_frames(files: List[str], cfg: Config, output_dir: str) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for f in files:
        veh = os.path.splitext(os.path.basename(f))[0]
        try:
            df = read_vehicle_csv(f)
            segs = extract_segments(df, cfg)
            labeled = build_pseudo_labels(segs, robust_linear=True)
            if not labeled.empty:
                frames[veh] = labeled
                labeled[["days", "raw_cap", "raw_cap_clip", "baseline_cap", "soh_true"]].to_csv(
                    os.path.join(output_dir, f"{veh}_pseudo_labels.csv"), index=False
                )
        except Exception as e:
            print(f"[WARN] {veh} 处理失败: {e}")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="SOH终极路线复现实验脚本")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smooth-window", type=int, default=15)
    args = parser.parse_args()

    cfg = Config(epochs=args.epochs, seed=args.seed, smooth_window=args.smooth_window)
    set_seed(cfg.seed)

    os.makedirs(args.output, exist_ok=True)
    files = collect_files(DATA_DIRS)
    if not files:
        print("❌ 未找到CSV数据，请将文件放入 data/ 或 data1/")
        return

    frames = build_vehicle_frames(files, cfg, args.output)
    if not frames:
        print("❌ 没有可用车辆样本，建议调低筛选阈值或调整电压窗口。")
        return

    metrics = train_and_eval(frames, cfg, args.output)
    if metrics.empty:
        print("❌ 训练/评估阶段无有效输出。")
        return

    print("✅ 实验完成，结果已导出：")
    print(f"- {args.output}/soh_metrics_vehicle.csv")
    print(f"- {args.output}/soh_metrics_summary.csv")
    print(f"- {args.output}/SOH_Predictions_For_SOC.csv")
    print(f"- {args.output}/soh_predictions_points.csv")


if __name__ == "__main__":
    main()

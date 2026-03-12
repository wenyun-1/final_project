"""SOH 终极路线：单脚本可复现实验管线

核心思想：
1) 充电片段按 SOC 75%→90% 截取
2) 每个充电片段计算出的 SOH 都直接作为监督标签
3) 固定测试集 EV1/EV8，导出 SOC 接口文件
"""

from __future__ import annotations

import glob
import os
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import interp1d
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, Dataset

TIME_FORMAT = "mixed"
DATA_DIRS = ["data"]
DEFAULT_OUTPUT_DIR = "outputs_final"
SEGMENT_CACHE_TAG = "soc75_90_v1"

SOC_WINDOW_START = 75.0
SOC_WINDOW_END = 90.0


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
    min_soc_delta: float = 10.0
    fixed_test_vehicles: List[str] = field(default_factory=lambda: ["LFP604EV1", "LFP604EV8"])
    read_chunk_size: int = 200000
    use_segment_cache: bool = True
    refresh_segment_cache: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_files(data_dirs: List[str]) -> List[str]:
    files: List[str] = []
    for d in data_dirs:
        files.extend(glob.glob(os.path.join(d, "*.csv")))
    return sorted(set(files))


def normalize_vehicle_name(file_stem: str) -> str:
    name = file_stem.strip().upper()
    m = re.search(r"(LFP\d+EV\d+)", name)
    if m:
        return m.group(1)
    return name


def _extract_from_one_segment(records: List[Dict], cfg: Config) -> Dict | None:
    if len(records) <= cfg.min_seg_points:
        return None
    df_seg = pd.DataFrame(records)

    soc_series = df_seg["SOC"].to_numpy(dtype=float)
    idx_s = int(np.argmin(np.abs(soc_series - SOC_WINDOW_START)))
    idx_e = int(np.argmin(np.abs(soc_series - SOC_WINDOW_END)))
    if idx_e <= idx_s:
        return None

    df_sub = df_seg.iloc[idx_s:idx_e + 1].copy()
    if len(df_sub) <= 10:
        return None

    dt = df_sub["DATA_TIME"].diff().dt.total_seconds().fillna(10)
    if dt.max() > cfg.max_gap_seconds:
        return None

    soc_delta = float(df_sub["SOC"].iloc[-1] - df_sub["SOC"].iloc[0])
    if soc_delta <= cfg.min_soc_delta:
        return None

    curr_abs = df_sub["totalCurrent"].abs()
    ah = (curr_abs * dt).sum() / 3600.0
    raw_cap = ah / (soc_delta / 100.0)

    v_seq = df_sub["totalVoltage"].to_numpy(dtype=float)
    f_interp = interp1d(np.linspace(0, 1, len(v_seq)), v_seq, kind="linear")
    fp_raw = f_interp(np.linspace(0, 1, 100))
    fp_min, fp_max = float(np.min(fp_raw)), float(np.max(fp_raw))
    fingerprint = (fp_raw - fp_min) / max(fp_max - fp_min, 1e-6)

    sub_duration_h = float(dt.sum() / 3600.0)
    voltage_rise_rate = float((df_sub["totalVoltage"].iloc[-1] - df_sub["totalVoltage"].iloc[0]) / max(sub_duration_h, 1e-6))

    q_inc = (df_sub["totalCurrent"].abs().to_numpy(dtype=float) * dt.to_numpy(dtype=float)) / 3600.0
    q_cum = np.cumsum(q_inc)
    dv = np.diff(df_sub["totalVoltage"].to_numpy(dtype=float))
    dq = np.diff(q_cum)
    valid = np.abs(dv) > 1e-4
    ic_peak = float(np.nanmax(np.clip(dq[valid] / dv[valid], -500, 500))) if np.any(valid) else 0.0

    return {
        "days": int((df_seg["DATA_TIME"].iloc[0] - pd.Timestamp("2020-01-01")).days),
        "raw_cap": float(raw_cap),
        "sub_duration_h": sub_duration_h,
        "soc_delta": float(soc_delta),
        "voltage_rise_rate": voltage_rise_rate,
        "ic_peak": ic_peak,
        "fingerprint": fingerprint,
        "avg_curr": float(df_sub["totalCurrent"].abs().mean()),
        "avg_temp": float(df_sub["maxTemperature"].mean()),
    }


def extract_segments_from_file(path: str, cfg: Config) -> List[Dict]:
    sample = pd.read_csv(path, nrows=5)
    use_cols = ["DATA_TIME", "totalCurrent", "totalVoltage", "SOC"]
    if "maxTemperature" in sample.columns:
        use_cols.append("maxTemperature")

    out, curr_seg = [], []
    reader = pd.read_csv(path, usecols=use_cols, low_memory=False, chunksize=cfg.read_chunk_size, on_bad_lines="skip")
    for chunk in reader:
        chunk["DATA_TIME"] = pd.to_datetime(chunk["DATA_TIME"], format=TIME_FORMAT, errors="coerce")
        for c in ["totalCurrent", "totalVoltage", "SOC"]:
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
        chunk["maxTemperature"] = pd.to_numeric(chunk["maxTemperature"], errors="coerce") if "maxTemperature" in chunk.columns else 25.0
        chunk = chunk.dropna(subset=["DATA_TIME", "totalCurrent", "totalVoltage", "SOC", "maxTemperature"])

        for row in chunk.itertuples(index=False):
            rec = {
                "DATA_TIME": row.DATA_TIME,
                "totalCurrent": float(row.totalCurrent),
                "totalVoltage": float(row.totalVoltage),
                "SOC": float(row.SOC),
                "maxTemperature": float(row.maxTemperature),
            }
            if rec["totalCurrent"] < -1.0:
                curr_seg.append(rec)
            else:
                seg = _extract_from_one_segment(curr_seg, cfg)
                if seg:
                    out.append(seg)
                curr_seg = []
    seg = _extract_from_one_segment(curr_seg, cfg)
    if seg:
        out.append(seg)
    return out


def _save_segments_cache(path: str, segs: List[Dict]) -> None:
    if not segs:
        return
    rows, fps = [], []
    for seg in segs:
        rows.append({k: v for k, v in seg.items() if k != "fingerprint"})
        fps.append(np.asarray(seg["fingerprint"], dtype=np.float32))
    np.savez_compressed(path, meta=np.array(rows, dtype=object), fingerprint=np.stack(fps, axis=0))


def _load_segments_cache(path: str) -> List[Dict]:
    z = np.load(path, allow_pickle=True)
    meta, fps = list(z["meta"]), z["fingerprint"]
    out = []
    for i, m in enumerate(meta):
        d = dict(m)
        d["fingerprint"] = fps[i]
        out.append(d)
    return out


def load_all_raw_segments(files: List[str], cfg: Config, cache_dir: str) -> Dict[str, List[Dict]]:
    os.makedirs(cache_dir, exist_ok=True)
    seg_by_vehicle: Dict[str, List[Dict]] = {}
    for i, f in enumerate(files, 1):
        veh_file = os.path.splitext(os.path.basename(f))[0]
        veh = normalize_vehicle_name(veh_file)
        cache_path = os.path.join(cache_dir, f"{veh}.npz")
        if cfg.use_segment_cache and (not cfg.refresh_segment_cache) and os.path.exists(cache_path):
            segs = _load_segments_cache(cache_path)
            print(f"[Data] {i}/{len(files)} {veh_file} -> 从缓存读取片段数 {len(segs)}")
        else:
            segs = extract_segments_from_file(f, cfg)
            if cfg.use_segment_cache:
                _save_segments_cache(cache_path, segs)
            print(f"[Data] {i}/{len(files)} {veh_file} -> 提取片段数 {len(segs)}")
        if segs:
            seg_by_vehicle.setdefault(veh, []).extend(segs)
    return seg_by_vehicle


def build_segment_labels(rows: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("days").reset_index(drop=True)
    q1, q99 = df["raw_cap"].quantile([0.01, 0.99])
    df["raw_cap_clip"] = df["raw_cap"].clip(lower=q1, upper=q99)

    n_head = max(5, int(len(df) * 0.2))
    baseline_cap = float(np.percentile(df["raw_cap_clip"].iloc[:n_head], 85))

    soh_true = (df["raw_cap_clip"] / max(baseline_cap, 1e-6)) * 100.0
    df["baseline_cap"] = baseline_cap
    df["soh_true"] = np.clip(soh_true, 0.0, 100.0)
    return df


class SOHDataset(Dataset):
    def __init__(self, rows: List[Dict], mean: np.ndarray, std: np.ndarray):
        self.rows = rows
        self.mean = np.asarray(mean)
        self.std = np.asarray(std) + 1e-9

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        curr_sc = (np.array(r["curr_sc_raw"], dtype=float) - self.mean) / self.std
        prev_sc = (np.array(r["prev_sc_raw"], dtype=float) - self.mean) / self.std
        y = r["soh_true"] / 100.0
        return (
            torch.tensor(r["curr_fp"], dtype=torch.float32).unsqueeze(0),
            torch.tensor(curr_sc, dtype=torch.float32),
            torch.tensor(r["prev_fp"], dtype=torch.float32).unsqueeze(0),
            torch.tensor(prev_sc, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            r["days"],
            r["Vehicle"],
        )


def build_rows_for_vehicles(vehicle_frames: Dict[str, pd.DataFrame], vehicles: List[str]) -> List[Dict]:
    rows = []
    for veh in vehicles:
        records = vehicle_frames[veh].sort_values("days").reset_index(drop=True).to_dict("records")
        for i, r in enumerate(records):
            prev = records[i - 1] if i > 0 else records[i]
            rows.append({
                "Vehicle": veh,
                "days": int(r["days"]),
                "soh_true": float(r["soh_true"]),
                "curr_fp": r["fingerprint"],
                "curr_sc_raw": [float(r["avg_curr"]), 0.0],
                "prev_fp": prev["fingerprint"],
                "prev_sc_raw": [float(prev["avg_curr"]), 0.0],
            })
    return rows


def split_vehicles(vehicle_frames: Dict[str, pd.DataFrame], cfg: Config) -> Tuple[List[str], List[str]]:
    vehicles = sorted(vehicle_frames.keys())
    fixed = [normalize_vehicle_name(v) for v in cfg.fixed_test_vehicles]
    test_vehicles = sorted([v for v in fixed if v in vehicles])
    train_vehicles = sorted([v for v in vehicles if v not in set(test_vehicles)])
    return train_vehicles, test_vehicles


class PIUAE(nn.Module):
    def __init__(self, num_scalars=2):
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
            nn.Linear(64 + num_scalars, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, fp, sc):
        feat = self.encoder(fp)
        pred = self.regressor(torch.cat((feat, sc), dim=1))
        recon = self.decoder(feat)
        return pred, recon


def train_and_eval(vehicle_frames: Dict[str, pd.DataFrame], cfg: Config, output_dir: str) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_vehicles, test_vehicles = split_vehicles(vehicle_frames, cfg)
    if not test_vehicles:
        return pd.DataFrame()

    train_rows = build_rows_for_vehicles(vehicle_frames, train_vehicles)
    if len(train_rows) == 0:
        return pd.DataFrame()

    scalars = np.array([r["curr_sc_raw"] for r in train_rows])
    mean, std = scalars.mean(0), scalars.std(0) + 1e-6

    train_loader = DataLoader(SOHDataset(train_rows, mean=mean, std=std), batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    model = PIUAE(num_scalars=len(train_rows[0]["curr_sc_raw"])).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    mse = nn.MSELoss()

    print("[Train] 开始训练 PI-UAE 模型...")
    for _ in range(cfg.epochs):
        model.train()
        for c_fp, c_sc, p_fp, p_sc, y, _, _ in train_loader:
            c_fp, c_sc, p_fp, p_sc, y = c_fp.to(device), c_sc.to(device), p_fp.to(device), p_sc.to(device), y.to(device)
            y_pred, recon = model(c_fp, c_sc)
            y_prev, _ = model(p_fp, p_sc)

            supervised_loss = mse(y_pred.squeeze(), y)
            loss = (
                supervised_loss
                + cfg.lambda_recon * mse(recon, c_fp)
                + cfg.alpha_physics * torch.relu(y_pred.squeeze() - y_prev.squeeze()).mean()
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    all_metrics, all_soc_export, error_plot_cache = [], [], {}
    test_set = set(test_vehicles)

    for veh in sorted(vehicle_frames.keys()):
        veh_rows = build_rows_for_vehicles(vehicle_frames, [veh])
        if len(veh_rows) < 5:
            continue
        test_loader = DataLoader(SOHDataset(veh_rows, mean=mean, std=std), batch_size=cfg.batch_size, shuffle=False)

        model.eval()
        days, y_true, y_pred = [], [], []
        with torch.no_grad():
            for c_fp, c_sc, _, _, y, d, _ in test_loader:
                p, _ = model(c_fp.to(device), c_sc.to(device))
                y_pred.extend((p.cpu().numpy().flatten() * 100).tolist())
                y_true.extend((y.numpy() * 100).tolist())
                days.extend(d.numpy().tolist())

        idx = np.argsort(days)
        days, y_true, y_pred = np.array(days)[idx], np.array(y_true)[idx], np.array(y_pred)[idx]
        y_pred_filtered = pd.Series(y_pred).rolling(window=cfg.smooth_window, min_periods=1, center=True).mean().values

        if veh in test_set:
            all_metrics.append({
                "Vehicle": veh,
                "N_test": len(y_true),
                "RMSE_raw": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "MAE_raw": float(mean_absolute_error(y_true, y_pred)),
            })
            error_plot_cache[veh] = {"days": days, "y_true": y_true, "y_pred_raw": y_pred, "y_pred_filtered": y_pred_filtered}

        for d, yf in zip(days, y_pred_filtered):
            all_soc_export.append({"Vehicle": veh, "Days": int(d), "Pred_SOH": float(yf / 100.0)})

    metric_df = pd.DataFrame(all_metrics)
    if not metric_df.empty:
        metric_df.to_csv(os.path.join(output_dir, "soh_metrics_vehicle.csv"), index=False)
    pd.DataFrame(all_soc_export).to_csv(os.path.join(output_dir, "SOH_Predictions_For_SOC.csv"), index=False)

    target_order = [v for v in cfg.fixed_test_vehicles if v in error_plot_cache]
    if target_order:
        fig, axes = plt.subplots(len(target_order), 2, figsize=(14, 4.2 * len(target_order)))
        axes = np.array(axes).reshape(len(target_order), 2)
        for i, veh in enumerate(target_order):
            dat = error_plot_cache[veh]
            ax_l, ax_r = axes[i, 0], axes[i, 1]
            ax_l.scatter(dat["days"], dat["y_pred_raw"], s=12, c="#4e79a7", alpha=0.5)
            ax_l.plot(dat["days"], dat["y_pred_filtered"], c="#e15759", lw=2.0, label="Pred Trend")
            ax_l.scatter(dat["days"], dat["y_true"], s=12, c="#9e9e9e", alpha=0.9, label="Label")
            ax_l.set_title(f"{veh} SOH")
            ax_l.legend()
            err = dat["y_pred_filtered"] - dat["y_true"]
            ax_r.axhline(0, color="gray", lw=1.0)
            ax_r.plot(dat["days"], err, c="#f28e2b", lw=1.6)
            ax_r.set_title(f"{veh} Error")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "soh_error.png"), dpi=260)
        plt.close(fig)

    return metric_df


def main() -> None:
    cfg = Config()
    set_seed(cfg.seed)

    files = collect_files(DATA_DIRS)
    if not files:
        print("❌ 未找到CSV数据")
        return

    cache_dir = os.path.join(DEFAULT_OUTPUT_DIR, f"segment_cache_{SEGMENT_CACHE_TAG}")
    raw_segs_by_vehicle = load_all_raw_segments(files, cfg, cache_dir)
    if not raw_segs_by_vehicle:
        print("❌ 数据清洗后无可用片段")
        return

    frames: Dict[str, pd.DataFrame] = {}
    for veh, segs in raw_segs_by_vehicle.items():
        labeled = build_segment_labels(segs)
        if not labeled.empty:
            frames[veh] = labeled

    if frames:
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        train_and_eval(frames, cfg, DEFAULT_OUTPUT_DIR)
        print("✅ 实验完成！")


if __name__ == "__main__":
    main()

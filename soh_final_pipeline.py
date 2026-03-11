"""SOH 终极路线：单脚本可复现实验管线（7天检修真值注入）

核心思想：
1) 伪标签：采用 weekly_inspection（模拟每7天检修）
2) 模型：提升了物理惩罚系数 (alpha_physics=2.0)，强迫模型学习真正的退化趋势，拒绝随温度震荡
3) 评估：自动按方法分目录导出结果，直接生成 SOC 接口文件
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import interp1d, PchipInterpolator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.isotonic import IsotonicRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from torch.utils.data import DataLoader, Dataset

TIME_FORMAT = "mixed"
DATA_DIRS = ["data"]
DEFAULT_OUTPUT_DIR = "outputs_final"

V_START = 538.0
V_END = 558.0


@dataclass
class Config:
    batch_size: int = 32
    epochs: int = 120
    learning_rate: float = 5e-4
    lambda_recon: float = 0.5
    # 【核心修改 1】：加大了物理惩罚！如果红线还乱跳，就把这里改成 5.0
    alpha_physics: float = 2.0
    smooth_window: int = 15
    seed: int = 42
    min_seg_points: int = 30
    max_gap_seconds: int = 60
    min_soc_delta: float = 20.0
    split_mode: str = "cross_vehicle"
    test_vehicle_ratio: float = 0.3
    train_vehicle_count: int = 10
    test_vehicle_count: int = 2
    # 固定测试集
    fixed_test_vehicles: List[str] = field(default_factory=lambda: ["LFP604EV11", "LFP604EV12"])
    read_chunk_size: int = 200000
    use_segment_cache: bool = True
    refresh_segment_cache: bool = False
    
    # [7-DAY ITERATION] 7天检修周期
    inspect_interval_days: int = 7
    # [7-DAY ITERATION] 真值注入模式：none / weekly
    truth_injection_mode: str = "weekly"
    # [7-DAY ITERATION] 真值监督损失权重
    truth_supervision_weight: float = 1.0
    # [7-DAY ITERATION] 自洽监督损失权重
    self_consistency_weight: float = 0.35
    # [7-DAY ITERATION] 检修评估窗口长度（7天：第0天真值，后6天预测）
    weekly_eval_horizon: int = 7


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
    if m: return m.group(1)
    return name

def _extract_from_one_segment(records: List[Dict], cfg: Config) -> Dict | None:
    if len(records) <= cfg.min_seg_points: return None
    df_seg = pd.DataFrame(records)
    if df_seg["totalVoltage"].min() >= V_START or df_seg["totalVoltage"].max() <= V_END: return None
    idx_s = (df_seg["totalVoltage"] - V_START).abs().idxmin()
    idx_e = (df_seg["totalVoltage"] - V_END).abs().idxmin()
    if idx_e <= idx_s: return None
    df_sub = df_seg.loc[idx_s:idx_e].copy()
    if len(df_sub) <= 10: return None
    dt = df_sub["DATA_TIME"].diff().dt.total_seconds().fillna(10)
    if dt.max() > cfg.max_gap_seconds: return None
    soc_delta = df_seg["SOC"].iloc[-1] - df_seg["SOC"].iloc[0]
    if soc_delta <= cfg.min_soc_delta: return None

    curr_abs = df_seg["totalCurrent"].abs()
    dt_full = df_seg["DATA_TIME"].diff().dt.total_seconds().fillna(10)
    
    # 这里正是利用了 SOC 差值计算出的每次充电片段的“观测容量”
    ah = (curr_abs * dt_full).sum() / 3600
    raw_cap = ah / (soc_delta / 100.0)

    v_seq = df_sub["totalVoltage"].values
    f_interp = interp1d(np.linspace(0, 1, len(v_seq)), v_seq, kind="linear")
    fingerprint = (f_interp(np.linspace(0, 1, 100)) - V_START) / (V_END - V_START)

    dt_sub = df_sub["DATA_TIME"].diff().dt.total_seconds().fillna(10)
    charge_duration_h = float(dt_full.sum() / 3600.0)
    sub_duration_h = float(dt_sub.sum() / 3600.0)
    voltage_rise_rate = float((V_END - V_START) / max(sub_duration_h, 1e-6))

    q_inc = (df_sub["totalCurrent"].abs().values * dt_sub.values) / 3600.0
    q_cum = np.cumsum(q_inc)
    dv = np.diff(df_sub["totalVoltage"].values)
    dq = np.diff(q_cum)
    valid = np.abs(dv) > 1e-4
    ic_peak = float(np.nanmax(np.clip(dq[valid] / dv[valid], -500, 500))) if np.any(valid) else 0.0

    return {
        "days": int((df_seg["DATA_TIME"].iloc[0] - pd.Timestamp("2020-01-01")).days),
        "raw_cap": float(raw_cap),
        "charge_duration_h": charge_duration_h,
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
    if "maxTemperature" in sample.columns: use_cols.append("maxTemperature")

    out, curr_seg = [], []
    reader = pd.read_csv(path, usecols=use_cols, low_memory=False, chunksize=cfg.read_chunk_size, on_bad_lines="skip")
    for chunk in reader:
        chunk["DATA_TIME"] = pd.to_datetime(chunk["DATA_TIME"], format=TIME_FORMAT, errors="coerce")
        for c in ["totalCurrent", "totalVoltage", "SOC"]: chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
        chunk["maxTemperature"] = pd.to_numeric(chunk["maxTemperature"], errors="coerce") if "maxTemperature" in chunk.columns else 25.0
        chunk = chunk.dropna(subset=["DATA_TIME", "totalCurrent", "totalVoltage", "SOC", "maxTemperature"])

        for row in chunk.itertuples(index=False):
            rec = {"DATA_TIME": row.DATA_TIME, "totalCurrent": float(row.totalCurrent), "totalVoltage": float(row.totalVoltage), "SOC": float(row.SOC), "maxTemperature": float(row.maxTemperature)}
            if rec["totalCurrent"] < -1.0: curr_seg.append(rec)
            else:
                seg = _extract_from_one_segment(curr_seg, cfg)
                if seg: out.append(seg)
                curr_seg = []
    seg = _extract_from_one_segment(curr_seg, cfg)
    if seg: out.append(seg)
    return out

def _save_segments_cache(path: str, segs: List[Dict]) -> None:
    if not segs: return
    rows, fps = [], []
    for s in segs:
        rows.append({k: v for k, v in s.items() if k != "fingerprint"})
        fps.append(np.asarray(s["fingerprint"], dtype=np.float32))
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
        if cfg.use_segment_cache and not cfg.refresh_segment_cache and os.path.exists(cache_path):
            segs = _load_segments_cache(cache_path)
            print(f"[Data] {i}/{len(files)} {veh_file} -> 从缓存读取片段数 {len(segs)}")
        else:
            segs = extract_segments_from_file(f, cfg)
            if cfg.use_segment_cache: _save_segments_cache(cache_path, segs)
            print(f"[Data] {i}/{len(files)} {veh_file} -> 提取片段数 {len(segs)}")
        if segs: seg_by_vehicle.setdefault(veh, []).extend(segs)
    return seg_by_vehicle

def build_pseudo_labels(rows: List[Dict], method: str, cfg: Config = Config()) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty: return df

    df = df.sort_values("days").reset_index(drop=True)
    q1, q99 = df["raw_cap"].quantile([0.01, 0.99])
    df["raw_cap_clip"] = df["raw_cap"].clip(lower=q1, upper=q99)

    n_head = max(5, int(len(df) * 0.2))
    baseline_cap = float(np.percentile(df["raw_cap_clip"].iloc[:n_head], 85))

    days = df["days"].to_numpy(dtype=float)
    raw_cap_clip = df["raw_cap_clip"].to_numpy(dtype=float)

    # [7-DAY ITERATION] 预置真值掩码（默认无真值注入）
    df["truth_mask"] = False

    # [7-DAY ITERATION] 使用较大窗口平滑 + 保序回归，避免阶梯状死区
    smooth_window = 21
    rolling_median = pd.Series(raw_cap_clip).rolling(window=smooth_window, center=True, min_periods=1).median().bfill().ffill().to_numpy()
    iso = IsotonicRegression(increasing=False)
    monotone_fit = iso.fit_transform(days, rolling_median)

    if method == "weekly_inspection":
        unique_days, indices = np.unique(days, return_index=True)
        unique_monotone = monotone_fit[indices]
        if len(unique_days) >= 3:
            pchip = PchipInterpolator(unique_days, unique_monotone)
            trend_cap = pchip(days)
        else:
            trend_cap = monotone_fit

        # 周期性真值日（每7天锚点 + 首末点）
        min_day = int(np.min(days))
        interval = max(1, int(cfg.inspect_interval_days))
        normalized_days = np.round(days - min_day).astype(int)
        truth_mask = (normalized_days % interval) == 0
        truth_mask[0] = True
        truth_mask[-1] = True
        df["truth_mask"] = truth_mask

    elif method == "pchip_smooth":
        unique_days, indices = np.unique(days, return_index=True)
        unique_monotone = monotone_fit[indices]
        if len(unique_days) >= 4:
            pchip = PchipInterpolator(unique_days, unique_monotone)
            trend_cap = pchip(days)
        else:
            trend_cap = monotone_fit

    else:
        raise ValueError(f"未知伪标签方法: {method}")

    # [7-DAY ITERATION] 插值后再次做保序回归，确保平滑单调递减
    trend_cap = iso.fit_transform(days, trend_cap)

    soh_true = (trend_cap / baseline_cap) * 100.0
    df["baseline_cap"] = baseline_cap
    df["soh_true"] = np.clip(soh_true, 0.0, 100.0)
    return df

class SOHDataset(Dataset):
    def __init__(self, rows: List[Dict], mean: np.ndarray, std: np.ndarray):
        self.rows = rows
        self.mean_full, self.std_full = np.asarray(mean), np.asarray(std)
        self.mean, self.std = self.mean_full, self.std_full + 1e-9

    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        curr_sc = (np.array(r["curr_sc_raw"], dtype=float) - self.mean) / self.std
        prev_sc = (np.array(r["prev_sc_raw"], dtype=float) - self.mean) / self.std
        y = r["soh_true"] / 100.0
        # [7-DAY ITERATION] 显式真值日标记
        is_truth_day = float(r.get("truth_mask", False))
        return (
            torch.tensor(r["curr_fp"], dtype=torch.float32).unsqueeze(0),
            torch.tensor(curr_sc, dtype=torch.float32),
            torch.tensor(r["prev_fp"], dtype=torch.float32).unsqueeze(0),
            torch.tensor(prev_sc, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(is_truth_day, dtype=torch.float32),
            r["days"], r["Vehicle"],
        )


# [7-DAY ITERATION] 训练时周期性真值注入器（可关闭，保持向后兼容）
class WeeklyTruthInjector:
    def __init__(self, mode: str = "weekly", interval_days: int = 7):
        self.mode = mode
        self.interval_days = max(1, int(interval_days))

    def inject(self, y_target: torch.Tensor, is_truth_day: torch.Tensor, days: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode != "weekly":
            return y_target, is_truth_day > 0.5
        day0 = days.min()
        week_mask = ((days - day0) % self.interval_days) == 0
        truth_mask = torch.logical_or(is_truth_day > 0.5, week_mask)
        return y_target, truth_mask

def build_rows_for_vehicles(vehicle_frames: Dict[str, pd.DataFrame], vehicles: List[str]) -> List[Dict]:
    rows = []
    for veh in vehicles:
        records = vehicle_frames[veh].sort_values("days").reset_index(drop=True).to_dict("records")
        for i, r in enumerate(records):
            prev = records[i - 1] if i > 0 else records[i]
            rows.append({
                "Vehicle": veh, "days": int(r["days"]), "soh_true": float(r["soh_true"]),
                # [7-DAY ITERATION] 向样本传递真值日标签
                "truth_mask": bool(r.get("truth_mask", False)),
                "curr_fp": r["fingerprint"], "curr_sc_raw": [float(r["avg_curr"]), 0.0],# 这里的第二个标量特征暂时占位为0.0，后续可以替换成 avg_temp 或其他有意义的特征，屏蔽温度特征
                "prev_fp": prev["fingerprint"], "prev_sc_raw": [float(prev["avg_curr"]), 0.0],
            })
    return rows

def split_vehicles(vehicle_frames: Dict[str, pd.DataFrame], cfg: Config) -> Tuple[List[str], List[str]]:
    vehicles = sorted(vehicle_frames.keys())
    if cfg.fixed_test_vehicles:
        fixed = [normalize_vehicle_name(v) for v in cfg.fixed_test_vehicles]
        test_vehicles = sorted([v for v in fixed if v in vehicles])
        train_vehicles = sorted([v for v in vehicles if v not in set(test_vehicles)])
        return train_vehicles, test_vehicles
    return vehicles, vehicles

class PIUAE(nn.Module):
    def __init__(self,num_scalars=2):
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
            nn.Linear(64 + num_scalars, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, fp, sc):
        feat = self.encoder(fp)
        pred = self.regressor(torch.cat((feat, sc), dim=1))
        recon = self.decoder(feat)
        return pred, recon

def train_and_eval(vehicle_frames: Dict[str, pd.DataFrame], cfg: Config, output_dir: str, method_name: str) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_metrics, all_soc_export, all_point_export = [], [], []

    train_vehicles, test_vehicles = split_vehicles(vehicle_frames, cfg)
    if not test_vehicles: return pd.DataFrame()

    train_rows = build_rows_for_vehicles(vehicle_frames, train_vehicles)
    if len(train_rows) == 0: return pd.DataFrame()

    scalars = np.array([r["curr_sc_raw"] for r in train_rows])
    mean, std = scalars.mean(0), scalars.std(0) + 1e-6

    train_ds = SOHDataset(train_rows, mean=mean, std=std)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    
    model = PIUAE(num_scalars=len(train_rows[0]["curr_sc_raw"])).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    mse = nn.MSELoss()
    # [7-DAY ITERATION] 周期性真值注入模块
    truth_injector = WeeklyTruthInjector(mode=cfg.truth_injection_mode, interval_days=cfg.inspect_interval_days)
    
    print(f"[{method_name}] 开始训练 PI-UAE 模型...")
    for epoch in range(cfg.epochs):
        model.train()
        for c_fp, c_sc, p_fp, p_sc, y, is_truth_day, d, _ in train_loader:
            c_fp, c_sc, p_fp, p_sc, y = c_fp.to(device), c_sc.to(device), p_fp.to(device), p_sc.to(device), y.to(device)
            is_truth_day = is_truth_day.to(device)
            d = d.to(device)
            y_pred, recon = model(c_fp, c_sc)
            y_prev, _ = model(p_fp, p_sc)
            y_target, truth_mask = truth_injector.inject(y, is_truth_day, d)

            # [7-DAY ITERATION] 显式拆分：真值监督日 vs 预测自洽日
            truth_mask_f = truth_mask.float()
            pred_mask_f = 1.0 - truth_mask_f
            truth_count = truth_mask_f.sum().clamp(min=1.0)
            pred_count = pred_mask_f.sum().clamp(min=1.0)
            supervised_loss = (((y_pred.squeeze() - y_target) ** 2) * truth_mask_f).sum() / truth_count
            self_consistency_loss = (((y_pred.squeeze() - y_target) ** 2) * pred_mask_f).sum() / pred_count

            # 物理约束的惩罚核心：如果今天的预测值大于昨天的预测值，就给予惩罚！
            loss = (
                cfg.truth_supervision_weight * supervised_loss
                + cfg.self_consistency_weight * self_consistency_loss
                + cfg.lambda_recon * mse(recon, c_fp)
                + cfg.alpha_physics * torch.relu(y_pred.squeeze() - y_prev.squeeze()).mean()
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    eval_vehicles = sorted(vehicle_frames.keys())
    test_set = set(test_vehicles)
    error_plot_cache = {}

    for veh in eval_vehicles:
        veh_rows = build_rows_for_vehicles(vehicle_frames, [veh])
        if len(veh_rows) < 5: continue
        test_loader = DataLoader(SOHDataset(veh_rows, mean=mean, std=std), batch_size=cfg.batch_size, shuffle=False)

        model.eval()
        days, y_true, y_pred = [], [], []
        with torch.no_grad():
            for c_fp, c_sc, _, _, y, _, d, _ in test_loader:
                p, _ = model(c_fp.to(device), c_sc.to(device))
                y_pred.extend((p.cpu().numpy().flatten() * 100).tolist())
                y_true.extend((y.numpy() * 100).tolist())
                days.extend(d.numpy().tolist())

        idx = np.argsort(days)
        days, y_true, y_pred = np.array(days)[idx], np.array(y_true)[idx], np.array(y_pred)[idx]
        y_pred_filtered = pd.Series(y_pred).rolling(window=cfg.smooth_window, min_periods=1, center=True).mean().values

        if veh in test_set:
            metrics = {
                "Vehicle": veh, "N_test": len(y_true),
                "RMSE_raw": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "MAE_raw": float(mean_absolute_error(y_true, y_pred)),
            }
            all_metrics.append(metrics)
            error_plot_cache[veh] = {"days": days, "y_true": y_true, "y_pred_raw": y_pred, "y_pred_filtered": y_pred_filtered}

        for d, yt, yr, yf in zip(days, y_true, y_pred, y_pred_filtered):
            all_soc_export.append({"Vehicle": veh, "Days": int(d), "Pred_SOH": float(yf / 100.0)})

    metric_df = pd.DataFrame(all_metrics)
    if not metric_df.empty: metric_df.to_csv(os.path.join(output_dir, "soh_metrics_vehicle.csv"), index=False)
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
            ax_l.plot(dat["days"], dat["y_true"], c="black", lw=1.4, alpha=0.8, label="Pseudo True")
            ax_l.set_title(f"[{method_name}] {veh} SOH")
            ax_l.legend()
            err = dat["y_pred_filtered"] - dat["y_true"]
            ax_r.axhline(0, color="gray", lw=1.0)
            ax_r.plot(dat["days"], err, c="#f28e2b", lw=1.6)
            ax_r.set_title(f"[{method_name}] {veh} Error")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"soh_error_{method_name}.png"), dpi=260)
        plt.close(fig)

    return metric_df


# [7-DAY ITERATION] 检修场景评估器：第0天真值已知，后6天漂移
def simulate_weekly_inspection(vehicle_frames: Dict[str, pd.DataFrame], cfg: Config, output_dir: str) -> pd.DataFrame:
    horizon = max(2, int(cfg.weekly_eval_horizon))
    rows = []
    for veh, df in sorted(vehicle_frames.items()):
        if df.empty or "soh_true" not in df.columns:
            continue
        dfx = df.sort_values("days").reset_index(drop=True)
        if len(dfx) < horizon:
            continue

        day0 = int(dfx["days"].iloc[0])
        rel = (dfx["days"] - day0).astype(int)
        truth_days = dfx[rel % max(1, cfg.inspect_interval_days) == 0].index.tolist()
        if not truth_days:
            truth_days = [0]

        for idx in truth_days:
            end_idx = min(idx + horizon, len(dfx))
            if end_idx - idx < 2:
                continue
            base = float(dfx.loc[idx, "soh_true"])
            chunk = dfx.iloc[idx:end_idx].copy()
            drift = (chunk["soh_true"].astype(float) - base).abs()
            pred_window_drift_mae = float(drift.iloc[1:].mean()) if len(drift) > 1 else 0.0
            rows.append({
                "Vehicle": veh,
                "anchor_day": int(chunk["days"].iloc[0]),
                "window_len": int(len(chunk)),
                "known_day0_soh": base,
                "pred_window_drift_mae": pred_window_drift_mae,
            })

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df.to_csv(os.path.join(output_dir, "weekly_inspection_eval.csv"), index=False)
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser()
    # 默认跑7天检修真值注入 + 平滑方法作对比
    parser.add_argument("--pseudo-label-methods", nargs="+", default=["weekly_inspection", "pchip_smooth"])
    args = parser.parse_args()

    cfg = Config()
    set_seed(cfg.seed)

    files = collect_files(DATA_DIRS)
    if not files: return print("❌ 未找到CSV数据")

    cache_dir = os.path.join(DEFAULT_OUTPUT_DIR, "segment_cache")
    raw_segs_by_vehicle = load_all_raw_segments(files, cfg, cache_dir)
    if not raw_segs_by_vehicle: return print("❌ 数据清洗后无可用片段")

    for method in args.pseudo_label_methods:
        print(f"\n================ 正在运行实验: {method} ================")
        method_out_dir = os.path.join(DEFAULT_OUTPUT_DIR, method)
        os.makedirs(method_out_dir, exist_ok=True)

        frames: Dict[str, pd.DataFrame] = {}
        for veh, segs in raw_segs_by_vehicle.items():
            labeled = build_pseudo_labels(segs, method=method, cfg=cfg)
            if not labeled.empty: frames[veh] = labeled
        
        if frames:
            train_and_eval(frames, cfg, method_out_dir, method)
            if method == "weekly_inspection":
                simulate_weekly_inspection(frames, cfg, method_out_dir)
            print(f"✅ {method} 实验完成！")

if __name__ == "__main__":
    main()

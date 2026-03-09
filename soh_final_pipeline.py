"""SOH 终极路线：单脚本可复现实验管线

核心思想：
1) 伪标签：前20%数据的85分位容量作为基准 + 一阶稳健拟合 + SOH<=100%物理上限
2) 模型：PI-UAE(电压指纹 + 温度/电流标量)
3) 评估：raw/filtered 双指标 + 自动导出论文表格 + SOC接口文件
"""

from __future__ import annotations

import argparse
import glob
import json
import inspect
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
from scipy.interpolate import interp1d
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset

TIME_FORMAT = "mixed"
DATA_DIRS = ["data"]
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
    split_mode: str = "cross_vehicle"
    test_vehicle_ratio: float = 0.3
    train_vehicle_count: int = 9
    test_vehicle_count: int = 3
    fixed_test_vehicles: List[str] = field(default_factory=lambda: ["LFP604EV3", "LFP604EV10", "LFP604EV9"])
    read_chunk_size: int = 200000
    log_every_epoch: int = 10
    use_segment_cache: bool = True
    refresh_segment_cache: bool = False
    reuse_if_same_trainset: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_files(data_dirs: List[str]) -> List[str]:
    files: List[str] = []
    for d in data_dirs:
        files.extend(glob.glob(os.path.join(d, "*.csv")))
    return sorted(set(files))


def sanity_check_source_structure() -> None:
    """检测本地文件是否因手工合并冲突出现重复函数定义。"""
    try:
        src = inspect.getsource(inspect.getmodule(sanity_check_source_structure))
    except Exception:
        return
    dup_targets = ["def split_vehicles(", "def build_rows_for_vehicles("]
    bad = {k: src.count(k) for k in dup_targets if src.count(k) > 1}
    if bad:
        raise RuntimeError(
            f"检测到代码冲突残留（重复定义）: {bad}。请用仓库最新版覆盖本地 soh_final_pipeline.py。"
        )


def normalize_vehicle_name(file_stem: str) -> str:
    """统一命名场景下的车辆键提取：优先使用 LFPxxxxEVx。"""
    name = file_stem.strip().upper()
    m = re.search(r"(LFP\d+EV\d+)", name)
    if m:
        return m.group(1)
    return name


def _extract_from_one_segment(records: List[Dict], cfg: Config) -> Dict | None:
    if len(records) <= cfg.min_seg_points:
        return None

    df_seg = pd.DataFrame(records)
    if df_seg["totalVoltage"].min() >= V_START or df_seg["totalVoltage"].max() <= V_END:
        return None

    idx_s = (df_seg["totalVoltage"] - V_START).abs().idxmin()
    idx_e = (df_seg["totalVoltage"] - V_END).abs().idxmin()
    if idx_e <= idx_s:
        return None

    df_sub = df_seg.loc[idx_s:idx_e].copy()
    if len(df_sub) <= 10:
        return None

    dt = df_sub["DATA_TIME"].diff().dt.total_seconds().fillna(10)
    if dt.max() > cfg.max_gap_seconds:
        return None

    soc_delta = df_seg["SOC"].iloc[-1] - df_seg["SOC"].iloc[0]
    if soc_delta <= cfg.min_soc_delta:
        return None

    curr_abs = df_seg["totalCurrent"].abs()
    dt_full = df_seg["DATA_TIME"].diff().dt.total_seconds().fillna(10)
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
    if np.any(valid):
        ic_curve = dq[valid] / dv[valid]
        ic_peak = float(np.nanmax(np.clip(ic_curve, -500, 500)))
    else:
        ic_peak = 0.0

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
    if "maxTemperature" in sample.columns:
        use_cols.append("maxTemperature")

    out: List[Dict] = []
    curr_seg: List[Dict] = []

    reader = pd.read_csv(
        path,
        usecols=use_cols,
        low_memory=False,
        chunksize=cfg.read_chunk_size,
        on_bad_lines="skip",
    )
    for chunk in reader:
        chunk["DATA_TIME"] = pd.to_datetime(chunk["DATA_TIME"], format=TIME_FORMAT, errors="coerce")
        for c in ["totalCurrent", "totalVoltage", "SOC"]:
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
        if "maxTemperature" in chunk.columns:
            chunk["maxTemperature"] = pd.to_numeric(chunk["maxTemperature"], errors="coerce")
        else:
            chunk["maxTemperature"] = 25.0
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
                if seg is not None:
                    out.append(seg)
                curr_seg = []

    seg = _extract_from_one_segment(curr_seg, cfg)
    if seg is not None:
        out.append(seg)
    return out


def _save_segments_cache(path: str, segs: List[Dict]) -> None:
    if len(segs) == 0:
        return
    rows = []
    fps = []
    for s in segs:
        rows.append(
            {
                "days": int(s["days"]),
                "raw_cap": float(s["raw_cap"]),
                "charge_duration_h": float(s["charge_duration_h"]),
                "sub_duration_h": float(s["sub_duration_h"]),
                "soc_delta": float(s["soc_delta"]),
                "voltage_rise_rate": float(s["voltage_rise_rate"]),
                "ic_peak": float(s["ic_peak"]),
                "avg_curr": float(s["avg_curr"]),
                "avg_temp": float(s["avg_temp"]),
            }
        )
        fps.append(np.asarray(s["fingerprint"], dtype=np.float32))
    np.savez_compressed(path, meta=np.array(rows, dtype=object), fingerprint=np.stack(fps, axis=0))


def _load_segments_cache(path: str) -> List[Dict]:
    z = np.load(path, allow_pickle=True)
    meta = list(z["meta"])
    fps = z["fingerprint"]
    out: List[Dict] = []
    for i, m in enumerate(meta):
        d = dict(m)
        d["fingerprint"] = fps[i]
        out.append(d)
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
    days = df["days"].to_numpy(dtype=float)
    raw_cap_clip = df["raw_cap_clip"].to_numpy(dtype=float)
    if robust_linear:
        # 先做中心化，降低病态拟合概率；同时约束衰减斜率不为正
        if np.unique(days).size >= 2:
            x = days - days.mean()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", np.RankWarning)
                    z = np.polyfit(x, raw_cap_clip, 1)
                z[0] = min(z[0], 0.0)
                trend_cap = np.poly1d(z)(x)
            except np.RankWarning:
                # 退化方案：滚动中位数 + 单调衰减约束
                trend_cap = pd.Series(raw_cap_clip).rolling(window=7, min_periods=1, center=True).median().values
                trend_cap = np.minimum.accumulate(trend_cap)
        else:
            trend_cap = np.full_like(raw_cap_clip, float(np.median(raw_cap_clip)))
    else:
        trend_cap = raw_cap_clip

    soh_true = (trend_cap / baseline_cap) * 100.0
    soh_true = np.clip(soh_true, 0.0, 100.0)

    df["baseline_cap"] = baseline_cap
    df["soh_true"] = soh_true
    return df


def _save_corr_heatmap(corr_df: pd.DataFrame, out_png: str, title: str) -> None:
    labels = corr_df.columns.tolist()
    arr = corr_df.values
    fig, ax = plt.subplots(figsize=(8, 6.8))
    im = ax.imshow(arr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=260)
    plt.close(fig)


def export_feature_correlation(vehicle_frames: Dict[str, pd.DataFrame], output_dir: str) -> None:
    merged = []
    for veh, frame in vehicle_frames.items():
        tmp = frame.copy()
        tmp["Vehicle"] = veh
        merged.append(tmp)
    if not merged:
        return

    df = pd.concat(merged, ignore_index=True)
    hi_cols = [
        "raw_cap", "charge_duration_h", "sub_duration_h", "soc_delta",
        "avg_curr", "avg_temp", "voltage_rise_rate", "ic_peak", "soh_true"
    ]
    hi_cols = [c for c in hi_cols if c in df.columns]
    if len(hi_cols) < 3:
        return

    hi_df = df[hi_cols].copy()
    hi_df.to_csv(os.path.join(output_dir, "hi_features_all_samples.csv"), index=False)

    pear = hi_df.corr(method="pearson")
    spear = hi_df.corr(method="spearman")
    pear.to_csv(os.path.join(output_dir, "hi_corr_pearson.csv"))
    spear.to_csv(os.path.join(output_dir, "hi_corr_spearman.csv"))
    _save_corr_heatmap(pear, os.path.join(output_dir, "hi_corr_pearson_heatmap.png"), "Pearson Correlation Heatmap")
    _save_corr_heatmap(spear, os.path.join(output_dir, "hi_corr_spearman_heatmap.png"), "Spearman Correlation Heatmap")


class SOHDataset(Dataset):
    def __init__(self, rows: List[Dict], mean: np.ndarray, std: np.ndarray):
        self.rows = rows
        self.mean, self.std = mean, std

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        curr_sc = (np.array(r["curr_sc_raw"]) - self.mean) / self.std
        prev_sc = (np.array(r["prev_sc_raw"]) - self.mean) / self.std
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


def split_vehicles(vehicle_frames: Dict[str, pd.DataFrame], cfg: Config) -> Tuple[List[str], List[str]]:
    vehicles = sorted(vehicle_frames.keys())
    if len(vehicles) < 2:
        return vehicles, []
    if cfg.split_mode == "intra_vehicle":
        return vehicles, vehicles

    rng = random.Random(cfg.seed)
    shuffled = vehicles[:]
    rng.shuffle(shuffled)
    if cfg.fixed_test_vehicles:
        fixed = [normalize_vehicle_name(v) for v in cfg.fixed_test_vehicles]
        miss = [v for v in fixed if v not in vehicles]
        if miss:
            raise ValueError(f"指定测试车辆不存在: {miss}，当前车辆: {vehicles}")
        test_vehicles = sorted(fixed)
        train_vehicles = sorted([v for v in vehicles if v not in set(test_vehicles)])
        if cfg.train_vehicle_count > 0 and len(train_vehicles) != cfg.train_vehicle_count:
            raise ValueError(f"训练车辆数异常: 期望 {cfg.train_vehicle_count}, 实际 {len(train_vehicles)}")
        return train_vehicles, test_vehicles

    if cfg.test_vehicle_count > 0:
        n_test = cfg.test_vehicle_count
    else:
        n_test = max(1, int(len(shuffled) * cfg.test_vehicle_ratio))
        n_test = min(len(shuffled) - 1, n_test)

    n_train = cfg.train_vehicle_count if cfg.train_vehicle_count > 0 else (len(shuffled) - n_test)
    if n_train <= 0:
        raise ValueError("train_vehicle_count 必须 > 0，或保证至少留有训练车辆。")
    if len(shuffled) < n_train + n_test:
        raise ValueError(
            f"车辆数量不足：当前仅 {len(shuffled)} 辆，但要求严格划分为 train={n_train} + test={n_test}。"
        )

    test_vehicles = sorted(shuffled[:n_test])
    remain = [v for v in shuffled if v not in set(test_vehicles)]
    remain = remain[:n_train]
    train_vehicles = sorted(remain)
    return train_vehicles, test_vehicles


def build_rows_for_vehicles(vehicle_frames: Dict[str, pd.DataFrame], vehicles: List[str]) -> List[Dict]:
    rows: List[Dict] = []
    for veh in vehicles:
        frame = vehicle_frames[veh].sort_values("days").reset_index(drop=True)
        records = frame.to_dict("records")
        for i, r in enumerate(records):
            prev = records[i - 1] if i > 0 else records[i]
            rows.append(
                {
                    "Vehicle": veh,
                    "days": int(r["days"]),
                    "soh_true": float(r["soh_true"]),
                    "curr_fp": r["fingerprint"],
                    "curr_sc_raw": [float(r["avg_curr"]), float(r["avg_temp"])],
                    "prev_fp": prev["fingerprint"],
                    "prev_sc_raw": [float(prev["avg_curr"]), float(prev["avg_temp"])],
                }
            )
    return rows


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

    train_vehicles, test_vehicles = split_vehicles(vehicle_frames, cfg)
    if cfg.split_mode == "cross_vehicle":
        if cfg.train_vehicle_count > 0 and len(train_vehicles) != cfg.train_vehicle_count:
            raise ValueError(f"训练车辆数异常: 期望 {cfg.train_vehicle_count}, 实际 {len(train_vehicles)}")
        if cfg.test_vehicle_count > 0 and len(test_vehicles) != cfg.test_vehicle_count:
            raise ValueError(f"测试车辆数异常: 期望 {cfg.test_vehicle_count}, 实际 {len(test_vehicles)}")
    if not test_vehicles:
        print("❌ 可用车辆数不足，无法执行跨车测试。")
        return pd.DataFrame()

    train_rows = build_rows_for_vehicles(vehicle_frames, train_vehicles)
    test_rows = build_rows_for_vehicles(vehicle_frames, test_vehicles)
    if len(train_rows) == 0 or len(test_rows) == 0:
        print("❌ 训练或测试样本为空。")
        return pd.DataFrame()

    scalars = np.array([r["curr_sc_raw"] for r in train_rows])
    mean, std = scalars.mean(0), scalars.std(0) + 1e-6
    np.savez(os.path.join(output_dir, "global_scaler.npz"), mean=mean, std=std)

    train_ds = SOHDataset(train_rows, mean=mean, std=std)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    print(f"[Data] 训练样本数: {len(train_rows)} | 测试样本数: {len(test_rows)}")

    signature = {
        "train_vehicles": train_vehicles,
        "train_rows": len(train_rows),
        "split_mode": cfg.split_mode,
        "train_vehicle_count": cfg.train_vehicle_count,
        "test_vehicle_count": cfg.test_vehicle_count,
    }
    sig_path = os.path.join(output_dir, "trainset_signature.json")
    model_path = os.path.join(output_dir, "global_pi_uae.pth")
    scaler_path = os.path.join(output_dir, "global_scaler.npz")
    can_reuse = False
    if cfg.reuse_if_same_trainset and os.path.exists(sig_path) and os.path.exists(model_path) and os.path.exists(scaler_path):
        with open(sig_path, "r", encoding="utf-8") as f:
            old_sig = json.load(f)
        can_reuse = old_sig == signature

    model = PIUAE().to(device)
    if can_reuse:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("[Train] 训练集未变化，直接复用已有模型并跳过训练。")
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
        mse = nn.MSELoss()
        for epoch in range(cfg.epochs):
            model.train()
            train_loss_sum = 0.0
            train_steps = 0
            for c_fp, c_sc, p_fp, p_sc, y, _, _ in train_loader:
                c_fp, c_sc, p_fp, p_sc, y = c_fp.to(device), c_sc.to(device), p_fp.to(device), p_sc.to(device), y.to(device)
                y_pred, recon = model(c_fp, c_sc)
                y_prev, _ = model(p_fp, p_sc)
                loss = mse(y_pred.squeeze(), y) + cfg.lambda_recon * mse(recon, c_fp) + cfg.alpha_physics * torch.relu(y_pred.squeeze() - y_prev.squeeze()).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_sum += float(loss.item())
                train_steps += 1

            if (epoch + 1) % max(1, cfg.log_every_epoch) == 0 or epoch == 0 or (epoch + 1) == cfg.epochs:
                avg_loss = train_loss_sum / max(1, train_steps)
                print(f"[Train] Epoch {epoch+1}/{cfg.epochs} | AvgLoss={avg_loss:.6f}")

        torch.save(model.state_dict(), model_path)
        with open(sig_path, "w", encoding="utf-8") as f:
            json.dump(signature, f, ensure_ascii=False, indent=2)

    print(f"[Split] 训练车辆 {len(train_vehicles)}: {train_vehicles}")
    print(f"[Split] 测试车辆 {len(test_vehicles)}: {test_vehicles}")
    split_rows = ([{"Vehicle": v, "Role": "train"} for v in train_vehicles] +
                  [{"Vehicle": v, "Role": "test"} for v in test_vehicles])
    pd.DataFrame(split_rows).to_csv(os.path.join(output_dir, "vehicle_split.csv"), index=False)

    eval_vehicles = sorted(vehicle_frames.keys())
    test_set = set(test_vehicles)
    error_plot_cache: Dict[str, Dict[str, np.ndarray]] = {}

    for veh in eval_vehicles:
        veh_rows = build_rows_for_vehicles(vehicle_frames, [veh])
        if len(veh_rows) < 5:
            continue
        test_ds = SOHDataset(veh_rows, mean=mean, std=std)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

        model.eval()
        days, y_true, y_pred = [], [], []
        with torch.no_grad():
            for c_fp, c_sc, _, _, y, d, _ in test_loader:
                p, _ = model(c_fp.to(device), c_sc.to(device))
                y_pred.extend((p.cpu().numpy().flatten() * 100).tolist())
                y_true.extend((y.numpy() * 100).tolist())
                days.extend(d.numpy().tolist())

        idx = np.argsort(days)
        days = np.array(days)[idx]
        y_true = np.array(y_true)[idx]
        y_pred = np.array(y_pred)[idx]
        y_pred_filtered = pd.Series(y_pred).rolling(window=cfg.smooth_window, min_periods=1, center=True).mean().values

        if veh in test_set:
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
            error_plot_cache[veh] = {
                "days": days,
                "y_true": y_true,
                "y_pred_raw": y_pred,
                "y_pred_filtered": y_pred_filtered,
            }

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

    target_order = [normalize_vehicle_name(v) for v in cfg.fixed_test_vehicles] if cfg.fixed_test_vehicles else sorted(test_set)
    target_order = [v for v in target_order if v in error_plot_cache]
    if target_order:
        fig, axes = plt.subplots(len(target_order), 2, figsize=(14, 4.2 * len(target_order)), sharex=False)
        axes = np.array(axes).reshape(len(target_order), 2)
        for i, veh in enumerate(target_order):
            dat = error_plot_cache[veh]
            days = dat["days"]
            y_true = dat["y_true"]
            y_pred_raw = dat["y_pred_raw"]
            y_pred_f = dat["y_pred_filtered"]

            ax_l = axes[i, 0]
            ax_l.scatter(days, y_pred_raw, s=12, c="#4e79a7", alpha=0.5, label="Estimated points")
            ax_l.plot(days, y_pred_f, c="#e15759", lw=2.0, label="SOH degradation trend")
            ax_l.plot(days, y_true, c="black", lw=1.4, alpha=0.8, label="Pseudo True SOH")
            ax_l.set_title(f"{veh} SOH Estimation")
            ax_l.set_ylabel("SOH (%)")
            ax_l.grid(alpha=0.25)
            ax_l.legend(loc="best")

            err = y_pred_f - y_true
            ax_r = axes[i, 1]
            ax_r.axhline(0, color="gray", lw=1.0)
            ax_r.plot(days, err, c="#f28e2b", lw=1.6)
            ax_r.fill_between(days, 0, err, color="#f28e2b", alpha=0.2)
            ax_r.set_title(f"{veh} Error (Pred-True)")
            ax_r.set_ylabel("Error (%)")
            ax_r.grid(alpha=0.25)

            axes[i, 0].set_xlabel("Days")
            axes[i, 1].set_xlabel("Days")

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "soh_test_vehicles_with_error.png"), dpi=260)
        plt.close(fig)

    return metric_df


def build_vehicle_frames(files: List[str], cfg: Config, output_dir: str) -> Dict[str, pd.DataFrame]:
    cache_dir = os.path.join(output_dir, "segment_cache")
    os.makedirs(cache_dir, exist_ok=True)
    seg_by_vehicle: Dict[str, List[Dict]] = {}
    for i, f in enumerate(files, 1):
        veh_file = os.path.splitext(os.path.basename(f))[0]
        veh = normalize_vehicle_name(veh_file)
        cache_path = os.path.join(cache_dir, f"{veh}.npz")
        try:
            if cfg.use_segment_cache and (not cfg.refresh_segment_cache) and os.path.exists(cache_path):
                segs = _load_segments_cache(cache_path)
                print(f"[LoadCache] {i}/{len(files)} {veh_file} -> 片段数 {len(segs)}")
            else:
                segs = extract_segments_from_file(f, cfg)
                if cfg.use_segment_cache:
                    _save_segments_cache(cache_path, segs)
                print(f"[Load] {i}/{len(files)} {veh_file} -> 片段数 {len(segs)}")
            if len(segs) > 0:
                seg_by_vehicle.setdefault(veh, []).extend(segs)
        except Exception as e:
            print(f"[WARN] {veh_file} 处理失败: {e}")

    frames: Dict[str, pd.DataFrame] = {}
    for veh, segs in seg_by_vehicle.items():
        labeled = build_pseudo_labels(segs, robust_linear=True)
        if labeled.empty:
            continue
        frames[veh] = labeled
        labeled[["days", "raw_cap", "raw_cap_clip", "baseline_cap", "soh_true"]].to_csv(
            os.path.join(output_dir, f"{veh}_pseudo_labels.csv"), index=False
        )
    return frames


def main() -> None:
    sanity_check_source_structure()
    parser = argparse.ArgumentParser(description="SOH终极路线复现实验脚本")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smooth-window", type=int, default=15)
    parser.add_argument("--split-mode", choices=["cross_vehicle", "intra_vehicle"], default="cross_vehicle")
    parser.add_argument("--test-vehicle-ratio", type=float, default=0.3)
    parser.add_argument("--train-vehicle-count", type=int, default=9, help="跨车训练车辆数（<=0 表示不限制）")
    parser.add_argument("--test-vehicle-count", type=int, default=3, help="跨车测试车辆数（<=0 表示按比例）")
    parser.add_argument("--test-vehicles", nargs="*", default=["LFP604EV3", "LFP604EV10", "LFP604EV9"], help="固定测试车辆列表")
    parser.add_argument("--data-dirs", nargs="+", default=DATA_DIRS, help="SOH原始数据目录列表（默认: data data1）")
    parser.add_argument("--read-chunk-size", type=int, default=200000, help="分块读取行数，内存不足时可调小")
    parser.add_argument("--log-every-epoch", type=int, default=10, help="每隔多少个epoch打印训练进度")
    parser.add_argument("--no-segment-cache", action="store_true", help="不使用充电片段缓存")
    parser.add_argument("--refresh-segment-cache", action="store_true", help="强制重建充电片段缓存")
    parser.add_argument("--reuse-if-same-trainset", action="store_true", help="训练集不变时复用模型并跳过训练")
    args = parser.parse_args()

    cfg = Config(
        epochs=args.epochs,
        seed=args.seed,
        smooth_window=args.smooth_window,
        split_mode=args.split_mode,
        test_vehicle_ratio=args.test_vehicle_ratio,
        train_vehicle_count=args.train_vehicle_count,
        test_vehicle_count=args.test_vehicle_count,
        fixed_test_vehicles=args.test_vehicles,
        read_chunk_size=args.read_chunk_size,
        log_every_epoch=args.log_every_epoch,
        use_segment_cache=(not args.no_segment_cache),
        refresh_segment_cache=args.refresh_segment_cache,
        reuse_if_same_trainset=args.reuse_if_same_trainset,
    )
    set_seed(cfg.seed)

    os.makedirs(args.output, exist_ok=True)
    files = collect_files(args.data_dirs)
    if not files:
        print(f"❌ 未找到CSV数据，请检查目录: {args.data_dirs}")
        return

    frames = build_vehicle_frames(files, cfg, args.output)
    if not frames:
        print("❌ 没有可用车辆样本，建议调低筛选阈值或调整电压窗口。")
        return

    export_feature_correlation(frames, args.output)

    metrics = train_and_eval(frames, cfg, args.output)
    if metrics.empty:
        print("❌ 训练/评估阶段无有效输出。")
        return

    print("✅ 实验完成，结果已导出：")
    print(f"- {args.output}/soh_metrics_vehicle.csv")
    print(f"- {args.output}/soh_metrics_summary.csv")
    print(f"- {args.output}/SOH_Predictions_For_SOC.csv")
    print(f"- {args.output}/soh_predictions_points.csv")
    print(f"- {args.output}/vehicle_split.csv")
    print(f"- {args.output}/hi_features_all_samples.csv")
    print(f"- {args.output}/hi_corr_pearson.csv")
    print(f"- {args.output}/hi_corr_spearman.csv")
    print(f"- {args.output}/hi_corr_pearson_heatmap.png")
    print(f"- {args.output}/hi_corr_spearman_heatmap.png")
    print(f"- {args.output}/soh_test_vehicles_with_error.png")


if __name__ == "__main__":
    main()

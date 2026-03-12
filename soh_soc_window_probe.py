#!/usr/bin/env python3
"""SOH 充电片段筛选画图探针。

按车辆执行：
1) 提取完整充电片段（电流<0 且 SOC 单调上升），筛选 SOC 跨度>=25%。
2) 在全车片段上做 25% SOC 滑窗覆盖统计，找最高频窗口。
3) 选取一个覆盖该窗口的片段，精确裁切窗口区间并绘制 2x2 诊断图。

输出：
- <output>/window_hist/<vehicle>_window_frequency.png
- <output>/probe_plots/<vehicle>_probe_2x2.png
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TIME_FORMAT = "mixed"


@dataclass
class ProbeConfig:
    min_seg_points: int = 30
    max_gap_seconds: int = 60
    min_soc_span: float = 25.0
    window_size: float = 25.0
    window_step: float = 1.0
    read_chunk_size: int = 200000


def normalize_vehicle_name(file_stem: str) -> str:
    name = file_stem.strip().upper()
    return name


def collect_files(data_dirs: List[str]) -> List[str]:
    files: List[str] = []
    for d in data_dirs:
        files.extend(glob.glob(os.path.join(d, "*.csv")))
    return sorted(set(files))


def _finalize_segment(records: List[Dict], cfg: ProbeConfig) -> pd.DataFrame | None:
    if len(records) < cfg.min_seg_points:
        return None
    seg = pd.DataFrame(records).sort_values("DATA_TIME").reset_index(drop=True)
    dt = seg["DATA_TIME"].diff().dt.total_seconds().fillna(0)
    if (dt > cfg.max_gap_seconds).any():
        return None

    seg["SOC"] = pd.to_numeric(seg["SOC"], errors="coerce")
    seg = seg.dropna(subset=["SOC"]).copy()
    if seg.empty:
        return None

    soc_start = float(seg["SOC"].iloc[0])
    soc_end = float(seg["SOC"].iloc[-1])
    if soc_end - soc_start < cfg.min_soc_span:
        return None

    # 允许轻微抖动，但整体需上升
    if seg["SOC"].diff().fillna(0).lt(-0.5).sum() > max(3, int(0.03 * len(seg))):
        return None

    return seg


def extract_charge_segments(path: str, cfg: ProbeConfig) -> List[pd.DataFrame]:
    sample = pd.read_csv(path, nrows=5)
    use_cols = ["DATA_TIME", "totalCurrent", "totalVoltage", "SOC"]
    if "maxTemperature" in sample.columns:
        use_cols.append("maxTemperature")

    segments: List[pd.DataFrame] = []
    curr: List[Dict] = []

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
                curr.append(rec)
            else:
                seg = _finalize_segment(curr, cfg)
                if seg is not None:
                    segments.append(seg)
                curr = []

    seg = _finalize_segment(curr, cfg)
    if seg is not None:
        segments.append(seg)
    return segments


def sliding_window_frequency(segments: List[pd.DataFrame], cfg: ProbeConfig) -> pd.DataFrame:
    starts = np.arange(0.0, 100.0 - cfg.window_size + 1e-9, cfg.window_step)
    counts = []
    for ws in starts:
        we = ws + cfg.window_size
        c = 0
        for seg in segments:
            s0, s1 = float(seg["SOC"].iloc[0]), float(seg["SOC"].iloc[-1])
            if s0 <= ws and s1 >= we:
                c += 1
        counts.append(c)
    return pd.DataFrame({"soc_start": starts, "soc_end": starts + cfg.window_size, "count": counts})


def trim_segment_by_soc(seg: pd.DataFrame, soc_start: float, soc_end: float) -> pd.DataFrame:
    asc = seg.sort_values("DATA_TIME").copy()
    asc = asc[(asc["SOC"] >= soc_start) & (asc["SOC"] <= soc_end)].copy()
    return asc.reset_index(drop=True)


def compute_ic_curve(seg: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    if len(seg) < 5:
        return np.array([]), np.array([])
    dt_h = seg["DATA_TIME"].diff().dt.total_seconds().fillna(0).to_numpy() / 3600.0
    curr_a = np.abs(seg["totalCurrent"].to_numpy())
    dq = curr_a * dt_h
    q = np.cumsum(dq)
    v = seg["totalVoltage"].to_numpy()

    dv = np.diff(v)
    dq_step = np.diff(q)
    ok = np.abs(dv) > 1e-4
    if ok.sum() < 3:
        return np.array([]), np.array([])
    x = (v[:-1] + v[1:]) / 2.0
    y = np.zeros_like(x)
    y[ok] = dq_step[ok] / dv[ok]
    # 简单平滑，避免噪点
    y = pd.Series(y).rolling(7, center=True, min_periods=1).mean().to_numpy()
    return x[ok], y[ok]


def plot_window_hist(dfw: pd.DataFrame, veh: str, out_path: str) -> None:
    plt.figure(figsize=(10, 4.2))
    plt.plot(dfw["soc_start"], dfw["count"], color="#1f77b4", lw=1.8)
    idx = int(dfw["count"].idxmax())
    best = dfw.loc[idx]
    plt.axvspan(best["soc_start"], best["soc_end"], color="#ff7f0e", alpha=0.25, label="Top-1 window")
    plt.xlabel("SOC window start (%)")
    plt.ylabel("Coverage count")
    plt.title(f"{veh} - 25% SOC window coverage frequency")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_probe(seg: pd.DataFrame, veh: str, soc_start: float, soc_end: float, out_path: str) -> None:
    t = (seg["DATA_TIME"] - seg["DATA_TIME"].iloc[0]).dt.total_seconds().to_numpy() / 60.0

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(t, seg["SOC"], color="#2ca02c")
    axes[0, 0].set_title("SOC vs Time")
    axes[0, 0].set_xlabel("Time (min)")
    axes[0, 0].set_ylabel("SOC (%)")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(t, seg["totalVoltage"], color="#d62728")
    axes[0, 1].set_title("Voltage vs Time")
    axes[0, 1].set_xlabel("Time (min)")
    axes[0, 1].set_ylabel("Voltage (V)")
    axes[0, 1].grid(alpha=0.25)

    ax_l = axes[1, 0]
    ax_r = ax_l.twinx()
    ax_l.plot(t, seg["totalCurrent"], color="#1f77b4", label="Current")
    ax_r.plot(t, seg["maxTemperature"], color="#ff7f0e", label="Temp")
    ax_l.set_title("Current & Temp vs Time")
    ax_l.set_xlabel("Time (min)")
    ax_l.set_ylabel("Current (A)", color="#1f77b4")
    ax_r.set_ylabel("Temp (°C)", color="#ff7f0e")
    ax_l.grid(alpha=0.25)

    xv, yic = compute_ic_curve(seg)
    if len(xv) > 0:
        axes[1, 1].plot(xv, yic, color="#9467bd")
    axes[1, 1].set_title("dQ/dV vs Voltage (IC)")
    axes[1, 1].set_xlabel("Voltage (V)")
    axes[1, 1].set_ylabel("dQ/dV (Ah/V)")
    axes[1, 1].grid(alpha=0.25)

    fig.suptitle(f"{veh} probe @ SOC [{soc_start:.1f}%, {soc_end:.1f}%]", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def process_vehicle(csv_path: str, cfg: ProbeConfig, output_dir: str) -> Dict:
    veh = normalize_vehicle_name(os.path.splitext(os.path.basename(csv_path))[0])
    segs = extract_charge_segments(csv_path, cfg)
    if not segs:
        return {"vehicle": veh, "segments": 0, "status": "no_valid_segment"}

    freq = sliding_window_frequency(segs, cfg)
    if freq["count"].max() <= 0:
        return {"vehicle": veh, "segments": len(segs), "status": "no_window_hit"}

    os.makedirs(os.path.join(output_dir, "window_hist"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "probe_plots"), exist_ok=True)

    hist_path = os.path.join(output_dir, "window_hist", f"{veh}_window_frequency.png")
    plot_window_hist(freq, veh, hist_path)

    best = freq.loc[freq["count"].idxmax()]
    ws, we = float(best["soc_start"]), float(best["soc_end"])
    candidates = [s for s in segs if float(s["SOC"].iloc[0]) <= ws and float(s["SOC"].iloc[-1]) >= we]
    # 选取最平滑的一段（SOC负跳最少）
    pick = min(candidates, key=lambda s: int(s["SOC"].diff().fillna(0).lt(-0.2).sum()))
    cut = trim_segment_by_soc(pick, ws, we)
    if len(cut) < 5:
        return {"vehicle": veh, "segments": len(segs), "status": "trim_too_short", "window": (ws, we)}

    probe_path = os.path.join(output_dir, "probe_plots", f"{veh}_probe_2x2.png")
    plot_probe(cut, veh, ws, we, probe_path)

    return {
        "vehicle": veh,
        "segments": len(segs),
        "status": "ok",
        "best_window_start": ws,
        "best_window_end": we,
        "best_count": int(best["count"]),
        "hist_path": hist_path,
        "probe_path": probe_path,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="按车辆绘制 SOH 片段筛选 25%SOC 诊断图")
    p.add_argument("--data-dirs", nargs="+", default=["data"], help="车辆CSV目录列表")
    p.add_argument("--output-dir", default="outputs_soc_probe", help="输出目录")
    p.add_argument("--window-size", type=float, default=25.0, help="SOC窗口宽度(%)")
    p.add_argument("--window-step", type=float, default=1.0, help="SOC窗口步长(%)")
    p.add_argument("--min-soc-span", type=float, default=25.0, help="充电片段最小SOC跨度(%)")
    p.add_argument("--max-vehicles", type=int, default=0, help="仅处理前N辆车，0表示全部")
    args = p.parse_args()

    cfg = ProbeConfig(
        window_size=args.window_size,
        window_step=args.window_step,
        min_soc_span=args.min_soc_span,
    )

    files = collect_files(args.data_dirs)
    if args.max_vehicles > 0:
        files = files[: args.max_vehicles]

    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    for i, f in enumerate(files, 1):
        res = process_vehicle(f, cfg, args.output_dir)
        rows.append(res)
        print(f"[{i}/{len(files)}] {res['vehicle']}: {res['status']}")

    out_csv = os.path.join(args.output_dir, "probe_summary.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Done. Summary -> {out_csv}")


if __name__ == "__main__":
    main()

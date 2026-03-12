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
from scipy.signal import savgol_filter

TIME_FORMAT = "mixed"


@dataclass
class ProbeConfig:
    min_seg_points: int = 30
    max_gap_seconds: int = 60
    min_soc_span: float = 25.0
    window_size: float = 25.0
    window_step: float = 1.0
    read_chunk_size: int = 200000
    cc_current_tol: float = 2.0
    cc_step_tol: float = 2.0
    cc_min_points: int = 20
    tail_cut_seconds: float = 40.0
    ic_voltage_step: float = 0.02
    ic_savgol_window: int = 7
    ic_savgol_polyorder: int = 2
    ic_outlier_quantile: float = 0.995


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


def extract_cc_subsegment(seg: pd.DataFrame, cfg: ProbeConfig) -> pd.DataFrame:
    """从片段中提取最稳定的恒流(CC)子段；遇到电流阶跃/波动自动截断。"""
    if seg.empty:
        return seg
    s = seg.sort_values("DATA_TIME").reset_index(drop=True).copy()
    cur = s["totalCurrent"].to_numpy()
    base = float(np.median(cur))
    # 阈值1：偏离主电流幅值过大（>±2A）
    stable_amp = np.abs(cur - base) <= cfg.cc_current_tol
    # 阈值2：相邻采样阶跃过大（>±2A）
    dcur = np.diff(cur, prepend=cur[0])
    stable_step = np.abs(dcur) <= cfg.cc_step_tol
    stable = stable_amp & stable_step

    best_start, best_len = -1, 0
    start = None
    for i, flag in enumerate(stable):
        if flag and start is None:
            start = i
        if (not flag or i == len(stable) - 1) and start is not None:
            end = i if not flag else i + 1
            ln = end - start
            if ln > best_len:
                best_start, best_len = start, ln
            start = None
    if best_start < 0 or best_len < cfg.cc_min_points:
        return s.iloc[0:0].copy()
    out = s.iloc[best_start : best_start + best_len].copy().reset_index(drop=True)
    return out


def compute_ic_curve(seg: pd.DataFrame, cfg: ProbeConfig) -> Tuple[np.ndarray, np.ndarray]:
    if len(seg) < 8:
        return np.array([]), np.array([])
    work = seg.sort_values("DATA_TIME").reset_index(drop=True).copy()

    # 切掉末端几十秒，避开CV尾段/断开造成的反向尾巴
    if cfg.tail_cut_seconds > 0:
        t_sec = (work["DATA_TIME"] - work["DATA_TIME"].iloc[0]).dt.total_seconds().to_numpy()
        keep = t_sec <= max(0.0, t_sec[-1] - cfg.tail_cut_seconds)
        if keep.sum() >= 8:
            work = work.loc[keep].reset_index(drop=True)
        if len(work) < 8:
            return np.array([]), np.array([])

    dt_h = work["DATA_TIME"].diff().dt.total_seconds().fillna(0).to_numpy() / 3600.0
    dt_h = np.clip(dt_h, 0.0, None)
    curr_a = np.abs(work["totalCurrent"].to_numpy(dtype=float))
    q = np.cumsum(curr_a * dt_h).astype(float)
    v = work["totalVoltage"].to_numpy(dtype=float)

    # 时间序列上先轻微平滑V，再强制单调非降，避免量化台阶导致的导数异常
    if len(v) >= 7:
        v_smooth = savgol_filter(v, window_length=7, polyorder=2)
    else:
        v_smooth = v
    v_mono = np.maximum.accumulate(v_smooth)

    # 构建单值Q(V): 对重复电压使用“最后一次”容量，保持随时间递增关系
    uniq_v, rev_idx = np.unique(v_mono[::-1], return_index=True)
    take_idx = len(v_mono) - 1 - rev_idx
    order = np.argsort(uniq_v)
    uniq_v = uniq_v[order]
    uniq_q = q[take_idx][order]
    if len(uniq_v) < 8 or (uniq_v.max() - uniq_v.min()) < cfg.ic_voltage_step * 6:
        return np.array([]), np.array([])

    # 固定电压间隔重采样（默认20mV，可改10mV）
    step = max(1e-4, float(cfg.ic_voltage_step))
    edges = np.arange(uniq_v.min(), uniq_v.max() + step * 0.5, step)
    if len(edges) < 8:
        return np.array([]), np.array([])
    q_edges = np.interp(edges, uniq_v, uniq_q)

    # 对Q(V)做适度S-G平滑后，再按固定ΔV求dQ/dV
    win = int(cfg.ic_savgol_window)
    if win % 2 == 0:
        win += 1
    win = min(win, len(q_edges) if len(q_edges) % 2 == 1 else len(q_edges) - 1)
    if win >= 5:
        q_edges = savgol_filter(q_edges, window_length=win, polyorder=min(cfg.ic_savgol_polyorder, win - 2))

    dqdv = np.diff(q_edges) / step
    v_mid = (edges[:-1] + edges[1:]) / 2.0

    # 仅做轻量去噪与极端值抑制，尽量保留峰值特征
    dqdv = pd.Series(dqdv).rolling(3, center=True, min_periods=1).median().to_numpy()
    qhi = np.quantile(dqdv, cfg.ic_outlier_quantile)
    dqdv = np.clip(dqdv, 0.0, qhi)
    return v_mid, dqdv


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


def plot_probe(seg: pd.DataFrame, veh: str, soc_start: float, soc_end: float, out_path: str, cfg: ProbeConfig) -> None:
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

    xv, yic = compute_ic_curve(seg, cfg)
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

    cc_cut = extract_cc_subsegment(cut, cfg)
    if len(cc_cut) < cfg.cc_min_points:
        return {
            "vehicle": veh,
            "segments": len(segs),
            "status": "no_cc_subsegment",
            "window": (ws, we),
        }

    probe_path = os.path.join(output_dir, "probe_plots", f"{veh}_probe_2x2.png")
    plot_probe(cc_cut, veh, ws, we, probe_path, cfg)

    return {
        "vehicle": veh,
        "segments": len(segs),
        "status": "ok",
        "best_window_start": ws,
        "best_window_end": we,
        "best_count": int(best["count"]),
        "cc_points": int(len(cc_cut)),
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
    p.add_argument("--cc-current-tol", type=float, default=2.0, help="恒流段电流偏差阈值(A)")
    p.add_argument("--cc-step-tol", type=float, default=2.0, help="恒流段相邻采样阶跃阈值(A)")
    p.add_argument("--tail-cut-seconds", type=float, default=40.0, help="IC计算前裁掉末端秒数")
    p.add_argument("--ic-voltage-step", type=float, default=0.02, help="IC计算电压重采样步长(V)")
    p.add_argument("--ic-savgol-window", type=int, default=7, help="S-G平滑窗口(奇数, 建议5-9)")
    args = p.parse_args()

    cfg = ProbeConfig(
        window_size=args.window_size,
        window_step=args.window_step,
        min_soc_span=args.min_soc_span,
        cc_current_tol=args.cc_current_tol,
        cc_step_tol=args.cc_step_tol,
        tail_cut_seconds=args.tail_cut_seconds,
        ic_voltage_step=args.ic_voltage_step,
        ic_savgol_window=args.ic_savgol_window,
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

#!/usr/bin/env python3
"""数据集基础可视化：按车辆、按过程输出充放电 V-SOC 与 IC 曲线图。

输出结构：
<output_dir>/
  <vehicle>/
    charge_00001.png
    discharge_00001.png
    ...
  dataset_basic_summary.csv

说明：
- 充电段判定：totalCurrent < -1.0 A
- 放电段判定：totalCurrent > +1.0 A
- 段分割：电流符号变化或时间间隔超过 max_gap_seconds
- IC 曲线：对每个充/放电过程按固定电压步长计算 dQ/dV（轻量平滑）
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

TIME_FORMAT = "mixed"


@dataclass
class Config:
    read_chunk_size: int = 200000
    min_points: int = 25
    max_gap_seconds: int = 60
    current_threshold: float = 1.0
    ic_voltage_step: float = 0.02
    tail_trim_seconds: float = 30.0


def collect_files(data_dirs: List[str]) -> List[str]:
    files: List[str] = []
    for d in data_dirs:
        files.extend(glob.glob(os.path.join(d, "*.csv")))
    return sorted(set(files))


def normalize_vehicle_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0].upper()


def _phase_of_current(i: float, th: float) -> str:
    if i < -th:
        return "charge"
    if i > th:
        return "discharge"
    return "idle"


def _finalize_segment(records: List[Dict], cfg: Config, phase: str) -> pd.DataFrame | None:
    if phase not in {"charge", "discharge"}:
        return None
    if len(records) < cfg.min_points:
        return None
    seg = pd.DataFrame(records).sort_values("DATA_TIME").reset_index(drop=True)
    dt = seg["DATA_TIME"].diff().dt.total_seconds().fillna(0)
    if (dt > cfg.max_gap_seconds).any():
        return None
    if seg["SOC"].isna().any() or seg["totalVoltage"].isna().any():
        return None
    return seg


def extract_segments(path: str, cfg: Config) -> Dict[str, List[pd.DataFrame]]:
    sample = pd.read_csv(path, nrows=5)
    use_cols = ["DATA_TIME", "totalCurrent", "totalVoltage", "SOC"]
    if "maxTemperature" in sample.columns:
        use_cols.append("maxTemperature")

    segments = {"charge": [], "discharge": []}
    curr_records: List[Dict] = []
    curr_phase = "idle"
    prev_t = None

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
        if "maxTemperature" not in chunk.columns:
            chunk["maxTemperature"] = 25.0
        else:
            chunk["maxTemperature"] = pd.to_numeric(chunk["maxTemperature"], errors="coerce")
        chunk = chunk.dropna(subset=["DATA_TIME", "totalCurrent", "totalVoltage", "SOC", "maxTemperature"])

        for row in chunk.itertuples(index=False):
            t = row.DATA_TIME
            i = float(row.totalCurrent)
            phase = _phase_of_current(i, cfg.current_threshold)
            rec = {
                "DATA_TIME": t,
                "totalCurrent": i,
                "totalVoltage": float(row.totalVoltage),
                "SOC": float(row.SOC),
                "maxTemperature": float(row.maxTemperature),
            }

            gap_break = prev_t is not None and (t - prev_t).total_seconds() > cfg.max_gap_seconds
            phase_break = phase != curr_phase
            if gap_break or phase_break:
                seg = _finalize_segment(curr_records, cfg, curr_phase)
                if seg is not None:
                    segments[curr_phase].append(seg)
                curr_records = []

            if phase in {"charge", "discharge"}:
                curr_records.append(rec)
            curr_phase = phase
            prev_t = t

    seg = _finalize_segment(curr_records, cfg, curr_phase)
    if seg is not None:
        segments[curr_phase].append(seg)

    return segments


def compute_ic_curve(seg: pd.DataFrame, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    if len(seg) < 8:
        return np.array([]), np.array([])

    work = seg.sort_values("DATA_TIME").reset_index(drop=True).copy()
    if cfg.tail_trim_seconds > 0:
        t_sec = (work["DATA_TIME"] - work["DATA_TIME"].iloc[0]).dt.total_seconds().to_numpy()
        keep = t_sec <= max(0.0, t_sec[-1] - cfg.tail_trim_seconds)
        if keep.sum() >= 8:
            work = work.loc[keep].reset_index(drop=True)

    if len(work) < 8:
        return np.array([]), np.array([])

    dt_h = work["DATA_TIME"].diff().dt.total_seconds().fillna(0).to_numpy() / 3600.0
    dt_h = np.clip(dt_h, 0.0, None)
    q = np.cumsum(np.abs(work["totalCurrent"].to_numpy()) * dt_h)
    v = work["totalVoltage"].to_numpy(dtype=float)
    v = np.maximum.accumulate(v)

    uniq_v, rev_idx = np.unique(v[::-1], return_index=True)
    idx = len(v) - 1 - rev_idx
    order = np.argsort(uniq_v)
    uniq_v = uniq_v[order]
    uniq_q = q[idx][order]

    step = max(1e-4, float(cfg.ic_voltage_step))
    if len(uniq_v) < 8 or (uniq_v.max() - uniq_v.min()) < step * 6:
        return np.array([]), np.array([])

    edges = np.arange(uniq_v.min(), uniq_v.max() + step * 0.5, step)
    if len(edges) < 8:
        return np.array([]), np.array([])

    q_edges = np.interp(edges, uniq_v, uniq_q)
    dqdv = np.diff(q_edges) / step
    dqdv = pd.Series(dqdv).rolling(3, center=True, min_periods=1).median().to_numpy()
    v_mid = (edges[:-1] + edges[1:]) / 2.0
    return v_mid, dqdv


def plot_segment(seg: pd.DataFrame, phase: str, veh: str, idx: int, out_path: str, cfg: Config) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].plot(seg["SOC"], seg["totalVoltage"], color="#1f77b4", lw=1.5)
    axes[0].set_title(f"{phase.capitalize()} Voltage vs SOC")
    axes[0].set_xlabel("SOC (%)")
    axes[0].set_ylabel("Voltage (V)")
    axes[0].grid(alpha=0.25)

    t_min = (seg["DATA_TIME"] - seg["DATA_TIME"].iloc[0]).dt.total_seconds() / 60.0
    axes[1].plot(t_min, seg["totalVoltage"], color="#d62728", lw=1.2, label="Voltage")
    ax2 = axes[1].twinx()
    ax2.plot(t_min, seg["SOC"], color="#2ca02c", lw=1.2, label="SOC")
    axes[1].set_title("Voltage & SOC vs Time")
    axes[1].set_xlabel("Time (min)")
    axes[1].set_ylabel("Voltage (V)")
    ax2.set_ylabel("SOC (%)")
    axes[1].grid(alpha=0.25)

    xv, yv = compute_ic_curve(seg, cfg)
    if len(xv) > 0:
        axes[2].plot(xv, yv, color="#9467bd", lw=1.4)
    else:
        axes[2].text(0.5, 0.5, "IC unavailable", ha="center", va="center", transform=axes[2].transAxes)
    axes[2].set_title("IC: dQ/dV vs Voltage")
    axes[2].set_xlabel("Voltage (V)")
    axes[2].set_ylabel("dQ/dV (Ah/V)")
    axes[2].grid(alpha=0.25)

    fig.suptitle(f"{veh} - {phase} segment #{idx:05d}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="按车辆输出充放电过程 V-SOC 与 IC 曲线图片")
    parser.add_argument("--data-dirs", nargs="+", default=["data"], help="输入CSV目录")
    parser.add_argument("--output-dir", default="outputs_dataset_basic_plots", help="输出目录")
    parser.add_argument("--max-vehicles", type=int, default=0, help="仅处理前N辆车，0=全部")
    parser.add_argument("--min-points", type=int, default=25, help="片段最少点数")
    parser.add_argument("--max-gap-seconds", type=int, default=60, help="片段内最大时间间隔")
    parser.add_argument("--current-threshold", type=float, default=1.0, help="充放电阈值电流(A)")
    parser.add_argument("--ic-voltage-step", type=float, default=0.02, help="IC电压步长(V)")
    args = parser.parse_args()

    cfg = Config(
        min_points=args.min_points,
        max_gap_seconds=args.max_gap_seconds,
        current_threshold=args.current_threshold,
        ic_voltage_step=args.ic_voltage_step,
    )

    files = collect_files(args.data_dirs)
    if args.max_vehicles > 0:
        files = files[: args.max_vehicles]

    os.makedirs(args.output_dir, exist_ok=True)
    summary_rows = []

    for i, fp in enumerate(files, 1):
        veh = normalize_vehicle_name(fp)
        veh_dir = os.path.join(args.output_dir, veh)
        os.makedirs(veh_dir, exist_ok=True)

        segs = extract_segments(fp, cfg)
        charge_list = segs["charge"]
        discharge_list = segs["discharge"]

        for j, seg in enumerate(charge_list, 1):
            out = os.path.join(veh_dir, f"charge_{j:05d}.png")
            plot_segment(seg, "charge", veh, j, out, cfg)
        for j, seg in enumerate(discharge_list, 1):
            out = os.path.join(veh_dir, f"discharge_{j:05d}.png")
            plot_segment(seg, "discharge", veh, j, out, cfg)

        summary_rows.append(
            {
                "vehicle": veh,
                "charge_segments": len(charge_list),
                "discharge_segments": len(discharge_list),
                "output_dir": veh_dir,
            }
        )
        print(f"[{i}/{len(files)}] {veh}: charge={len(charge_list)}, discharge={len(discharge_list)}")

    summary_csv = os.path.join(args.output_dir, "dataset_basic_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"Done -> {summary_csv}")


if __name__ == "__main__":
    main()

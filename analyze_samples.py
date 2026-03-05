"""轻量级 CSV 数据体检脚本（无需 pandas/torch）。"""

import csv
import datetime as dt
import glob
import os
from statistics import fmean

V_START, V_END = 540.0, 552.0


def iter_files():
    files = []
    for d in ("data", "data1", "samples"):
        files.extend(glob.glob(os.path.join(d, "*.csv")))
    return sorted(set(files))


def parse_file(path):
    recs = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = dt.datetime.fromisoformat(row["DATA_TIME"])
                curr = float(row["totalCurrent"])
                volt = float(row["totalVoltage"])
            except Exception:
                continue
            recs.append((ts, curr, volt))
    return recs


def count_segments(records):
    segs, buf = [], []
    for rec in records:
        if rec[1] < -1:
            buf.append(rec)
        else:
            if len(buf) > 30:
                segs.append(buf)
            buf = []
    if len(buf) > 30:
        segs.append(buf)
    return segs


def report(path):
    records = parse_file(path)
    if not records:
        return f"{path}: 无有效记录"

    currents = [r[1] for r in records]
    volts = [r[2] for r in records]
    segs = count_segments(records)

    cc_candidates = 0
    for seg in segs:
        v = [x[2] for x in seg]
        if min(v) < V_START and max(v) > V_END:
            cc_candidates += 1

    return (
        f"{path}\n"
        f"  记录数={len(records)}\n"
        f"  电流范围=[{min(currents):.1f}, {max(currents):.1f}] A, 充电点均值={fmean(abs(c) for c in currents if c < -1):.1f} A\n"
        f"  电压范围=[{min(volts):.1f}, {max(volts):.1f}] V\n"
        f"  充电片段数={len(segs)}, 覆盖{V_START:.0f}-{V_END:.0f}V窗口的片段={cc_candidates}"
    )


if __name__ == "__main__":
    files = iter_files()
    if not files:
        print("未找到 CSV 文件，请先将数据放入 data/ 或 data1/")
    else:
        print("SOH 数据体检结果：")
        for f in files:
            print(report(f))

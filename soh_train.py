import os
import glob
import random
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader, Dataset

# ================= 核心配置 =================
DATA_DIRS = ["./data", "./data1", "./samples"]
TIME_FORMAT = "mixed"

# 固定物理电压窗口 (提取区间容量的绝对标尺)
V_START, V_END = 540.0, 552.0
MIN_SEGMENT_POINTS = 30
MAX_TIME_GAP_SECONDS = 60
MAX_CURRENT_FLUCTUATION = 15.0

BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 0.0005

LAMBDA_RECON = 0.5
ALPHA_PHYSICS = 0.01

# 该电压段内最大充入电量估计上限
AH_MAX_SCALE = 50.0


def collect_raw_files() -> List[str]:
    files: List[str] = []
    for data_dir in DATA_DIRS:
        files.extend(glob.glob(os.path.join(data_dir, "*.csv")))
    return sorted(set(files))


RAW_FILES = collect_raw_files()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(42)


class PIDataset(Dataset):
    def __init__(self, raw_files: List[str], is_train: bool = True):
        self.data = []
        self.vehicle_names: Dict[int, str] = {}
        print(f"正在构建{'训练' if is_train else '测试'}集 (离散区间容量范式)...")

        vehicle_idx = 0
        all_raw_samples = []

        for f in raw_files:
            try:
                base_name = os.path.basename(f).split(".")[0]
                self.vehicle_names[vehicle_idx] = base_name
                print(f"⏳ 处理车辆: {base_name} ...")

                sample_df = pd.read_csv(f, nrows=5)
                has_temp = "maxTemperature" in sample_df.columns
                use_cols = ["DATA_TIME", "totalCurrent", "totalVoltage"]
                if has_temp:
                    use_cols.append("maxTemperature")

                df = pd.read_csv(f, usecols=use_cols, low_memory=False)
                df["DATA_TIME"] = pd.to_datetime(
                    df["DATA_TIME"], format=TIME_FORMAT, dayfirst=False, errors="coerce"
                )
                num_cols = ["totalCurrent", "totalVoltage"]
                if has_temp:
                    num_cols.append("maxTemperature")
                for col in num_cols:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna().sort_values("DATA_TIME").reset_index(drop=True)

                records = df.to_dict("records")
                candidate_segments = []
                curr_seg = []

                for row in records:
                    if row["totalCurrent"] < -1.0:
                        curr_seg.append(row)
                    else:
                        self._collect_candidate(curr_seg, candidate_segments, has_temp)
                        curr_seg = []

                # 文件以充电片段结尾时，补一次 flush
                self._collect_candidate(curr_seg, candidate_segments, has_temp)

                if len(candidate_segments) <= 20:
                    vehicle_idx += 1
                    continue

                # 自适应恒流窗口：避免硬编码 110~130A 导致跨车型样本损失
                avg_currs = np.array([seg["avg_curr"] for seg in candidate_segments])
                q10, q90 = np.percentile(avg_currs, [10, 90])
                curr_low, curr_high = q10 - 3.0, q90 + 3.0

                vehicle_segments = []
                for seg in candidate_segments:
                    if seg["curr_fluctuation"] > MAX_CURRENT_FLUCTUATION:
                        continue
                    if not (curr_low <= seg["avg_curr"] <= curr_high):
                        continue
                    vehicle_segments.append(seg)

                df_veh = pd.DataFrame(vehicle_segments)
                if df_veh.empty or len(df_veh) <= 20:
                    vehicle_idx += 1
                    continue

                df_veh = df_veh.sort_values("days").reset_index(drop=True)
                veh_samples = []
                for i in range(len(df_veh)):
                    row = df_veh.iloc[i]
                    prev_row = df_veh.iloc[i - 1] if i > 0 else df_veh.iloc[i]
                    y_norm = row["interval_ah"] / AH_MAX_SCALE
                    veh_samples.append(
                        {
                            "x_curr_fp": row["fingerprint"],
                            "x_curr_sc": row["scalars"],
                            "x_prev_fp": prev_row["fingerprint"],
                            "x_prev_sc": prev_row["scalars"],
                            "y_curr": y_norm,
                            "days": row["days"],
                            "vid": vehicle_idx,
                        }
                    )

                all_raw_samples.extend(veh_samples)
                print(
                    f"   ✅ 候选片段 {len(candidate_segments)} -> 过滤后 {len(veh_samples)} "
                    f"(电流窗口 {curr_low:.1f}~{curr_high:.1f}A)"
                )
                vehicle_idx += 1
            except Exception as e:
                print(f"⚠️ 处理文件失败: {f} | {e}")

        if len(all_raw_samples) > 0:
            if is_train:
                train_scalars = np.array(
                    [s["x_curr_sc"] for i, s in enumerate(all_raw_samples) if i % 5 != 0]
                )
                sc_mean = np.mean(train_scalars, axis=0)
                sc_std = np.std(train_scalars, axis=0) + 1e-6
                np.savez("scaler_stats.npz", mean=sc_mean, std=sc_std)
            else:
                stats = np.load("scaler_stats.npz")
                sc_mean, sc_std = stats["mean"], stats["std"]

            for s in all_raw_samples:
                s["x_curr_sc"] = (np.array(s["x_curr_sc"]) - sc_mean) / sc_std
                s["x_prev_sc"] = (np.array(s["x_prev_sc"]) - sc_mean) / sc_std
                self.data.append(s)

        if is_train:
            self.data = [d for i, d in enumerate(self.data) if i % 5 != 0]
        else:
            self.data = [d for i, d in enumerate(self.data) if i % 5 == 0]

    def _collect_candidate(self, curr_seg, candidate_segments, has_temp: bool):
        if len(curr_seg) <= MIN_SEGMENT_POINTS:
            return

        df_seg = pd.DataFrame(curr_seg)
        v_min, v_max = df_seg["totalVoltage"].min(), df_seg["totalVoltage"].max()
        if not (v_min < V_START and v_max > V_END):
            return

        idx_s = (df_seg["totalVoltage"] - V_START).abs().idxmin()
        idx_e = (df_seg["totalVoltage"] - V_END).abs().idxmin()
        if idx_e <= idx_s:
            return

        df_sub = df_seg.loc[idx_s:idx_e].copy()
        v_seq = df_sub["totalVoltage"].values
        if len(v_seq) <= 10:
            return

        dt_series = df_sub["DATA_TIME"].diff().dt.total_seconds().fillna(10)
        if dt_series.max() > MAX_TIME_GAP_SECONDS:
            return

        curr_array = df_sub["totalCurrent"].abs()
        avg_curr = curr_array.mean()
        curr_fluctuation = curr_array.max() - curr_array.min()
        interval_ah = (curr_array * dt_series).sum() / 3600
        if not (1.0 < interval_ah < AH_MAX_SCALE):
            return

        f_interp = interp1d(np.linspace(0, 1, len(v_seq)), v_seq, kind="linear")
        fp_norm = (f_interp(np.linspace(0, 1, 100)) - V_START) / (V_END - V_START)

        avg_temp = df_sub["maxTemperature"].mean() if has_temp else 25.0
        days = (df_seg["DATA_TIME"].iloc[0] - pd.Timestamp("2020-01-01")).days

        candidate_segments.append(
            {
                "days": days,
                "interval_ah": interval_ah,
                "fingerprint": fp_norm,
                "scalars": [avg_curr, avg_temp],
                "avg_curr": avg_curr,
                "curr_fluctuation": curr_fluctuation,
            }
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            torch.tensor(item["x_curr_fp"], dtype=torch.float32).unsqueeze(0),
            torch.tensor(item["x_curr_sc"], dtype=torch.float32),
            torch.tensor(item["x_prev_fp"], dtype=torch.float32).unsqueeze(0),
            torch.tensor(item["x_prev_sc"], dtype=torch.float32),
            torch.tensor(item["y_curr"], dtype=torch.float32),
            item["vid"],
            item["days"],
        )


class MultiModal_PI_UAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 3, 1, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, 1, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(32 * 25, 64),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 32 * 25),
            nn.ReLU(),
            nn.Unflatten(1, (32, 25)),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(32, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(16, 1, 3, 1, 1),
            nn.Sigmoid(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(64 + 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, fp, sc):
        shape_feat = self.encoder(fp)
        return self.regressor(torch.cat((shape_feat, sc), dim=1)), self.decoder(shape_feat)


def train_model():
    if len(RAW_FILES) == 0:
        print("❌ 没有找到可用 CSV，请将数据放入 data/ 或 data1/")
        return

    train_ds = PIDataset(RAW_FILES, is_train=True)
    if len(train_ds) == 0:
        print("❌ 数据过滤后没有训练样本，请先运行数据分析脚本检查过滤条件。")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModal_PI_UAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_mse = nn.MSELoss()

    print(f"\n🚀 开始模型训练 (设备: {device})...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for curr_fp, curr_sc, prev_fp, prev_sc, y_curr, _, _ in train_loader:
            curr_fp, curr_sc, y_curr = curr_fp.to(device), curr_sc.to(device), y_curr.to(device)
            prev_fp, prev_sc = prev_fp.to(device), prev_sc.to(device)

            pred_curr, recon_curr = model(curr_fp, curr_sc)
            pred_prev, _ = model(prev_fp, prev_sc)

            loss = (
                criterion_mse(pred_curr.squeeze(), y_curr)
                + LAMBDA_RECON * criterion_mse(recon_curr, curr_fp)
                + ALPHA_PHYSICS * torch.relu(pred_curr.squeeze() - pred_prev.squeeze()).mean()
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d}/{EPOCHS}, Loss: {epoch_loss / len(train_loader):.6f}")

    torch.save(model.state_dict(), "pi_uae_model_weights.pth")
    print("\n💾 训练结束！模型已保存。请运行 soh_eval.py")


if __name__ == "__main__":
    train_model()

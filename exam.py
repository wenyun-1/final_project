import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d
from sklearn.isotonic import IsotonicRegression # ⚡️ 核心大杀器：保序回归
import random

# ================= 核心配置 =================
DATA_DIR = './data'
RAW_FILES = glob.glob(os.path.join(DATA_DIR, '*.csv'))
TIME_FORMAT = 'mixed'
V_START, V_END = 540.0, 552.0
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 0.0005
LAMBDA_RECON = 0.5   
ALPHA_PHYSICS = 0.02 
SOH_MIN, SOH_MAX = 60.0, 110.0

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(42)

# ================= 数据集类 (保序回归提取真值) =================
class PIDataset(Dataset):
    def __init__(self, raw_files, is_train=True):
        self.data = []
        self.vehicle_names = {} 
        print(f"正在构建{'训练' if is_train else '测试'}集 (保序回归范式)...")
        
        vehicle_idx = 0
        all_raw_samples = [] 
        
        for f in raw_files:
            try:
                base_name = os.path.basename(f).split('.')[0]
                self.vehicle_names[vehicle_idx] = base_name
                print(f"⏳ 处理车辆: {base_name} ...")
                
                sample_df = pd.read_csv(f, nrows=5)
                has_temp = 'maxTemperature' in sample_df.columns
                use_cols = ['DATA_TIME', 'totalCurrent', 'totalVoltage', 'SOC']
                if has_temp: use_cols.append('maxTemperature')

                df = pd.read_csv(f, usecols=use_cols, low_memory=False)
                df['DATA_TIME'] = pd.to_datetime(df['DATA_TIME'], format=TIME_FORMAT, dayfirst=False, errors='coerce')
                num_cols = ['totalCurrent', 'totalVoltage', 'SOC']
                if has_temp: num_cols.append('maxTemperature')
                for col in num_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna()
                
                records = df.to_dict('records')
                vehicle_segments, curr_seg = [], []
                
                for row in records:
                    if row['totalCurrent'] < -1.0: 
                        curr_seg.append(row)
                    else:
                        if len(curr_seg) > 30:
                            df_seg = pd.DataFrame(curr_seg)
                            v_min, v_max = df_seg['totalVoltage'].min(), df_seg['totalVoltage'].max()
                            if v_min < V_START and v_max > V_END:
                                idx_s = (df_seg['totalVoltage'] - V_START).abs().idxmin()
                                idx_e = (df_seg['totalVoltage'] - V_END).abs().idxmin()
                                if idx_e > idx_s:
                                    df_sub = df_seg.loc[idx_s:idx_e]
                                    v_seq = df_sub['totalVoltage'].values
                                    if len(v_seq) > 10: 
                                        f_interp = interp1d(np.linspace(0, 1, len(v_seq)), v_seq, kind='linear')
                                        fp_norm = (f_interp(np.linspace(0, 1, 100)) - V_START) / (V_END - V_START)
                                        
                                        dt = df_sub['DATA_TIME'].diff().dt.total_seconds().fillna(10)
                                        seg_ah = (df_sub['totalCurrent'].abs() * dt).sum() / 3600
                                        avg_curr = df_sub['totalCurrent'].abs().mean()
                                        avg_temp = df_sub['maxTemperature'].mean() if has_temp else 25.0
                                        
                                        soc_delta = df_seg['SOC'].iloc[-1] - df_seg['SOC'].iloc[0]
                                        # 基础过滤：剔除浅充片段以减少极大误差
                                        if soc_delta > 20.0:
                                            tot_dt = df_seg['DATA_TIME'].diff().dt.total_seconds().fillna(10)
                                            tot_ah = (df_seg['totalCurrent'].abs() * tot_dt).sum() / 3600
                                            raw_cap = tot_ah / (soc_delta / 100.0)
                                            days = (df_seg['DATA_TIME'].iloc[0] - pd.Timestamp("2020-01-01")).days
                                            
                                            vehicle_segments.append({
                                                'days': days, 'raw_cap': raw_cap,
                                                'fingerprint': fp_norm,
                                                'scalars': [seg_ah, avg_curr, avg_temp] 
                                            })
                            curr_seg = []
                
                df_veh = pd.DataFrame(vehicle_segments)
                if not df_veh.empty:
                    df_veh = df_veh[(df_veh['raw_cap'] > 300) & (df_veh['raw_cap'] < 1500)]
                    
                    if len(df_veh) > 20:
                        # 1. 强制按时间排序
                        df_veh = df_veh.sort_values('days').reset_index(drop=True)
                        
                        # 2. 截断极值 (去噪)：滤除上下 10% 的极端测算误差
                        q_low = df_veh['raw_cap'].quantile(0.10)
                        q_high = df_veh['raw_cap'].quantile(0.90)
                        df_veh['raw_cap_clipped'] = df_veh['raw_cap'].clip(lower=q_low, upper=q_high)
                        
                        # ================= ⚡️ 核心：保序回归 (Isotonic Regression) ⚡️ =================
                        # increasing=False 强制要求生成的曲线只能下降或平移，绝不允许上升，完美契合 SOH 物理定义！
                        iso_reg = IsotonicRegression(increasing=False, out_of_bounds='clip')
                        iso_cap = iso_reg.fit_transform(df_veh['days'], df_veh['raw_cap_clipped'])
                        
                        # 保序回归起点的容量必定是全局最高水平，直接作为 100% 物理基准
                        cap_start_true = iso_cap[0]
                        cap_end_true = iso_cap[-1]
                        deg_ratio = (cap_start_true - cap_end_true) / cap_start_true
                        
                        # 只要确有物理衰减即可保留
                        if deg_ratio > 0.01: 
                            # 3. 生成严格单调下降的 SOH 真值标签，并截断上限于 100%
                            df_veh['SOH_True'] = (iso_cap / cap_start_true) * 100
                            df_veh['SOH_True'] = df_veh['SOH_True'].clip(upper=100.0)
                            
                            veh_samples = []
                            for i in range(len(df_veh)):
                                row = df_veh.iloc[i]
                                prev_row = df_veh.iloc[i-1] if i > 0 else df_veh.iloc[i]
                                if 60 <= row['SOH_True'] <= 100:
                                    y_norm = (row['SOH_True'] - SOH_MIN) / (SOH_MAX - SOH_MIN)
                                    veh_samples.append({
                                        'x_curr_fp': row['fingerprint'], 'x_curr_sc': row['scalars'],
                                        'x_prev_fp': prev_row['fingerprint'], 'x_prev_sc': prev_row['scalars'],
                                        'y_curr': y_norm, 'days': row['days'], 'vid': vehicle_idx
                                    })
                            all_raw_samples.extend(veh_samples)
                            print(f"   ✅ 保序回归提取成功! 拟合衰减率: {deg_ratio*100:.2f}%")
                        else:
                            print(f"   ⚠️ 容量无宏观衰减趋势，已剔除。")
                vehicle_idx += 1
            except Exception as e: 
                print(f"出错: {e}")

        # 归一化处理
        if len(all_raw_samples) > 0:
            if is_train:
                train_scalars = np.array([s['x_curr_sc'] for i, s in enumerate(all_raw_samples) if i % 5 != 0])
                sc_mean = np.mean(train_scalars, axis=0)
                sc_std = np.std(train_scalars, axis=0) + 1e-6
                np.savez('scaler_stats.npz', mean=sc_mean, std=sc_std)
                print("\n✅ 归一化参数已计算并保存至 scaler_stats.npz")
            else:
                stats = np.load('scaler_stats.npz')
                sc_mean, sc_std = stats['mean'], stats['std']

            for s in all_raw_samples:
                s['x_curr_sc'] = (np.array(s['x_curr_sc']) - sc_mean) / sc_std
                s['x_prev_sc'] = (np.array(s['x_prev_sc']) - sc_mean) / sc_std
                self.data.append(s)

        # 交错抽样
        if is_train: self.data = [d for i, d in enumerate(self.data) if i % 5 != 0] 
        else:        self.data = [d for i, d in enumerate(self.data) if i % 5 == 0] 

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return (torch.tensor(item['x_curr_fp'], dtype=torch.float32).unsqueeze(0),
                torch.tensor(item['x_curr_sc'], dtype=torch.float32),
                torch.tensor(item['x_prev_fp'], dtype=torch.float32).unsqueeze(0),
                torch.tensor(item['x_prev_sc'], dtype=torch.float32),
                torch.tensor(item['y_curr'], dtype=torch.float32),
                item['vid'], item['days'])

# ================= 模型结构 (带有物理约束的多模态网络) =================
class MultiModal_PI_UAE(nn.Module):
    def __init__(self):
        super(MultiModal_PI_UAE, self).__init__()
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
            nn.Linear(64 + 3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() 
        )

    def forward(self, fp, sc):
        shape_feat = self.encoder(fp)
        return self.regressor(torch.cat((shape_feat, sc), dim=1)), self.decoder(shape_feat)

# ================= 训练流程 =================
def train_model():
    train_ds = PIDataset(RAW_FILES, is_train=True)
    if len(train_ds) == 0: return
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModal_PI_UAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_mse = nn.MSELoss()
    
    print(f"\n🚀 开始 模型训练 (设备: {device})...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for curr_fp, curr_sc, prev_fp, prev_sc, y_curr, _, _ in train_loader:
            curr_fp, curr_sc, y_curr = curr_fp.to(device), curr_sc.to(device), y_curr.to(device)
            prev_fp, prev_sc = prev_fp.to(device), prev_sc.to(device)
            
            latent_curr, recon_curr = model(curr_fp, curr_sc)
            latent_prev, _ = model(prev_fp, prev_sc)
            
            # 损失包含：预测误差 + 重构误差 + 单调物理约束惩罚
            loss = criterion_mse(latent_curr.squeeze(), y_curr) + \
                   LAMBDA_RECON * criterion_mse(recon_curr, curr_fp) + \
                   ALPHA_PHYSICS * torch.relu(latent_curr.squeeze() - latent_prev.squeeze()).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch+1) % 10 == 0: print(f"Epoch {epoch+1:3d}/{EPOCHS}, Loss: {epoch_loss / len(train_loader):.6f}")

    torch.save(model.state_dict(), 'pi_uae_model_weights.pth')
    print(f"\n💾 训练结束！保序回归模型已保存。请运行 soh_eval.py")

if __name__ == "__main__":
    train_model()
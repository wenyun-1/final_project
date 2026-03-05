import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error, r2_score
import random

# ================= 核心配置 =================
DATA_DIR = './data'
RAW_FILES = glob.glob(os.path.join(DATA_DIR, '*.csv'))
print(f"📁 在 {DATA_DIR} 目录下共找到 {len(RAW_FILES)} 个 CSV 文件。")

TIME_FORMAT = 'mixed'

# 物理窗口
V_START = 538.0
V_END = 558.0

# 训练参数
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 0.0005

# ⚡️ 损失函数权重
LAMBDA_RECON = 0.5   
ALPHA_PHYSICS = 0.05 # 降低一点物理约束，防止过拟合工况噪声

# SOH 归一化范围
SOH_MIN = 60.0
SOH_MAX = 110.0

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
# =========================================

# === 1. 多模态物理约束数据集类 ===
class PIDataset(Dataset):
    def __init__(self, raw_files, is_train=True):
        self.data = []
        self.vehicle_names = {} 
        
        print(f"正在构建{'训练' if is_train else '测试'}集 (引入多模态特征)...")
        
        vehicle_idx = 0
        for f in raw_files:
            try:
                base_name = os.path.basename(f).split('.')[0]
                self.vehicle_names[vehicle_idx] = base_name
                print(f"⏳ 处理车辆: {base_name} ...")
                
                # ⚡️ 恢复引入 maxTemperature (极度重要!)
                use_cols = ['DATA_TIME', 'totalCurrent', 'totalVoltage', 'SOC']
                # 如果有温度列就读，没有就不用（兼容性处理）
                sample_df = pd.read_csv(f, nrows=5)
                has_temp = 'maxTemperature' in sample_df.columns
                if has_temp:
                    use_cols.append('maxTemperature')

                # 1. 加上 low_memory=False 消除 DtypeWarning 警告
                df = pd.read_csv(f, usecols=use_cols, low_memory=False)
                
                # 2. 转换时间，解析失败的直接变 NaT
                df['DATA_TIME'] = pd.to_datetime(df['DATA_TIME'], format=TIME_FORMAT, dayfirst=False, errors='coerce')
                
                # 3. ⚡️ 强力清洗：把数值列强制转为数字，遇到那个“超级字符串”直接变成 NaN
                num_cols = ['totalCurrent', 'totalVoltage', 'SOC']
                if has_temp:
                    num_cols.append('maxTemperature')
                    
                for col in num_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # 4. 把那些被我们变成 NaN 的乱码行，以及原本就空缺的行，统统删掉
                df = df.dropna()
                records = df.to_dict('records')
                
                vehicle_segments = []
                curr_seg = []
                
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
                                        # 1. 形状指纹 (绝对电压归一化，保留起伏特征)
                                        f_interp = interp1d(np.linspace(0, 1, len(v_seq)), v_seq, kind='linear')
                                        fingerprint = f_interp(np.linspace(0, 1, 100))
                                        fingerprint_norm = (fingerprint - V_START) / (V_END - V_START)
                                        
                                        # 2. ⚡️ 尺度特征 (最重要的修补！)
                                        dt = df_sub['DATA_TIME'].diff().dt.total_seconds().fillna(10)
                                        seg_ah = (df_sub['totalCurrent'].abs() * dt).sum() / 3600
                                        avg_curr = df_sub['totalCurrent'].abs().mean()
                                        avg_temp = df_sub['maxTemperature'].mean() if has_temp else 25.0
                                        
                                        # 标量特征归一化 (经验范围)
                                        # 局部容量通常在 0-20Ah, 电流 0-150A, 温度 -20-60度
                                        scalars = [
                                            seg_ah / 20.0, 
                                            avg_curr / 150.0, 
                                            (avg_temp + 20.0) / 80.0
                                        ]
                                        
                                        # 3. 标签生成参数
                                        soc_delta = df_seg['SOC'].iloc[-1] - df_seg['SOC'].iloc[0]
                                        if soc_delta > 20.0:
                                            total_dt = df_seg['DATA_TIME'].diff().dt.total_seconds().fillna(10)
                                            total_ah = (df_seg['totalCurrent'].abs() * total_dt).sum() / 3600
                                            raw_cap = total_ah / (soc_delta / 100.0)
                                            days = (df_seg['DATA_TIME'].iloc[0] - pd.Timestamp("2020-01-01")).days
                                            
                                            vehicle_segments.append({
                                                'days': days,
                                                'raw_cap': raw_cap,
                                                'fingerprint': fingerprint_norm,
                                                'scalars': scalars
                                            })
                            curr_seg = []
                
                # 拟合真值并构造时序对
                df_veh = pd.DataFrame(vehicle_segments)
                if not df_veh.empty:
                    df_veh = df_veh[(df_veh['raw_cap'] > 400) & (df_veh['raw_cap'] < 1500)]
                    
                    if len(df_veh) > 20:
                        z = np.polyfit(df_veh['days'], df_veh['raw_cap'], 1)
                        if z[0] < 0: # 必须衰减
                            p = np.poly1d(z)
                            cap_init = p(df_veh['days'].min())
                            df_veh['SOH_True'] = (p(df_veh['days']) / cap_init) * 100
                            
                            df_veh = df_veh.sort_values('days').reset_index(drop=True)
                            
                            veh_samples = []
                            for i in range(len(df_veh)):
                                row = df_veh.iloc[i]
                                prev_row = df_veh.iloc[i-1] if i > 0 else df_veh.iloc[i]
                                
                                if 60 <= row['SOH_True'] <= 110:
                                    y_curr_norm = (row['SOH_True'] - SOH_MIN) / (SOH_MAX - SOH_MIN)
                                    veh_samples.append({
                                        'x_curr_fp': row['fingerprint'],
                                        'x_curr_sc': row['scalars'],
                                        'x_prev_fp': prev_row['fingerprint'],
                                        'x_prev_sc': prev_row['scalars'],
                                        'y_curr': y_curr_norm,
                                        'days': row['days'],
                                        'vid': vehicle_idx
                                    })
                            
                            print(f"   ✅ 提取成功: 生成 {len(veh_samples)} 对多模态时序数据")
                            self.data.extend(veh_samples)
                        else:
                            print(f"   ⚠️ 跳过: 车辆容量无衰减")
                vehicle_idx += 1
                
            except Exception as e:
                print(f"读取出错 {f}: {e}")

        # 划分数据集 (以车辆内的时间间隔划分，确保分布一致)
        if is_train:
            self.data = [d for i, d in enumerate(self.data) if i % 5 != 0] 
        else:
            self.data = [d for i, d in enumerate(self.data) if i % 5 == 0] 
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            torch.tensor(item['x_curr_fp'], dtype=torch.float32).unsqueeze(0),
            torch.tensor(item['x_curr_sc'], dtype=torch.float32),
            torch.tensor(item['x_prev_fp'], dtype=torch.float32).unsqueeze(0),
            torch.tensor(item['x_prev_sc'], dtype=torch.float32),
            torch.tensor(item['y_curr'], dtype=torch.float32),
            item['vid'],
            item['days']
        )

# === 2. 网络架构 (多模态 PI-UAE) ===
class MultiModal_PI_UAE(nn.Module):
    def __init__(self):
        super(MultiModal_PI_UAE, self).__init__()
        # 编码器 (处理形状)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(32 * 25, 64), nn.ReLU() # 降维到64
        )
        # 解码器 (重构任务)
        self.decoder = nn.Sequential(
            nn.Linear(64, 32 * 25), nn.ReLU(), nn.Unflatten(1, (32, 25)),
            nn.Upsample(scale_factor=2), nn.Conv1d(32, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2), nn.Conv1d(16, 1, kernel_size=3, padding=1), nn.Sigmoid() 
        )
        # ⚡️ 融合回归器 (处理 形状64维 + 标量3维 = 67维)
        self.regressor = nn.Sequential(
            nn.Linear(64 + 3, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Sigmoid() 
        )

    def forward(self, fp, sc):
        # 1. 提取形状特征
        shape_feat = self.encoder(fp)
        # 2. 重构形状
        fp_recon = self.decoder(shape_feat)
        # 3. 特征融合: 形状特征拼接尺度环境特征 [batch, 64] concat [batch, 3]
        combined_feat = torch.cat((shape_feat, sc), dim=1)
        # 4. 预测SOH
        latent_z = self.regressor(combined_feat)
        return latent_z, fp_recon

# === 3. 训练与评估 ===
def train_and_eval():
    if len(RAW_FILES) == 0: return

    train_ds = PIDataset(RAW_FILES, is_train=True)
    test_ds = PIDataset(RAW_FILES, is_train=False)
    
    if len(train_ds) == 0: return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModal_PI_UAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_mse = nn.MSELoss()
    
    print(f"\n🚀 开始 多模态PI-UAE 训练 (融合 Ah、Current、Temp)...")
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for curr_fp, curr_sc, prev_fp, prev_sc, y_curr, _, _ in train_loader:
            curr_fp, curr_sc = curr_fp.to(device), curr_sc.to(device)
            prev_fp, prev_sc = prev_fp.to(device), prev_sc.to(device)
            y_curr = y_curr.to(device)
            
            # 当前时刻与前一时刻的前向传播
            latent_curr, recon_curr = model(curr_fp, curr_sc)
            latent_prev, _ = model(prev_fp, prev_sc)
            
            # 损失计算
            loss_soh = criterion_mse(latent_curr.squeeze(), y_curr)
            loss_recon = criterion_mse(recon_curr, curr_fp)
            
            # 物理约束: 若今天的SOH(curr)比昨天(prev)高，产生惩罚
            diff = latent_curr.squeeze() - latent_prev.squeeze()
            loss_physics = torch.relu(diff).mean()
            
            loss = loss_soh + LAMBDA_RECON * loss_recon + ALPHA_PHYSICS * loss_physics
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS}, Loss: {epoch_loss / len(train_loader):.6f}")

    # === 4. 分车独立输出评估 ===
    model.eval()
    results_by_veh = {}
    
    print("\n正在进行分车测试集评估...")
    with torch.no_grad():
        for curr_fp, curr_sc, _, _, y_curr, vids, days in test_loader:
            curr_fp, curr_sc = curr_fp.to(device), curr_sc.to(device)
            latent_curr, _ = model(curr_fp, curr_sc)
            
            preds = latent_curr.cpu().numpy().flatten() * (SOH_MAX - SOH_MIN) + SOH_MIN
            trues = y_curr.numpy() * (SOH_MAX - SOH_MIN) + SOH_MIN
            vids = vids.numpy()
            days = days.numpy()
            
            for i in range(len(vids)):
                vid = vids[i]
                if vid not in results_by_veh:
                    results_by_veh[vid] = {'days':[], 'trues':[], 'preds':[]}
                results_by_veh[vid]['days'].append(days[i])
                results_by_veh[vid]['trues'].append(trues[i])
                results_by_veh[vid]['preds'].append(preds[i])

    # === 5. 可视化 ===
    num_vehicles = len(results_by_veh)
    if num_vehicles == 0: return
        
    fig, axes = plt.subplots(num_vehicles, 1, figsize=(10, 4 * num_vehicles))
    if num_vehicles == 1: axes = [axes]
    
    overall_rmse_list = []
    
    for i, (vid, res) in enumerate(results_by_veh.items()):
        sorted_indices = np.argsort(res['days'])
        days_sorted = np.array(res['days'])[sorted_indices]
        trues_sorted = np.array(res['trues'])[sorted_indices]
        preds_sorted = np.array(res['preds'])[sorted_indices]
        
        rmse = np.sqrt(mean_squared_error(trues_sorted, preds_sorted))
        r2 = r2_score(trues_sorted, preds_sorted)
        overall_rmse_list.append(rmse)
        
        veh_name = test_ds.vehicle_names.get(vid, f"Vehicle_{vid}")
        print(f"🚗 {veh_name} -> RMSE: {rmse:.3f}%, R2: {r2:.3f}")
        
        ax = axes[i]
        ax.plot(days_sorted, trues_sorted, 'k-', linewidth=2, label='True SOH (Trend)')
        ax.scatter(days_sorted, preds_sorted, c='crimson', alpha=0.8, s=30, label='PI-UAE Prediction')
        
        # 为了视觉清晰，使用滑动平均画一条拟合线
        if len(preds_sorted) > 10:
            smoothed_preds = pd.Series(preds_sorted).rolling(window=5, min_periods=1).mean()
            ax.plot(days_sorted, smoothed_preds, 'r-', alpha=0.6, linewidth=2, label='Smoothed Prediction')

        ax.set_title(f"Degradation Tracking: {veh_name} (RMSE={rmse:.3f}%, R2={r2:.3f})")
        ax.set_xlabel("Days since 2020-01-01")
        ax.set_ylabel("SOH (%)")
        ax.grid(True, alpha=0.4)
        ax.legend()

    print(f"\n🏆 所有车辆的平均 RMSE: {np.mean(overall_rmse_list):.3f}%")
    plt.tight_layout()
    plt.savefig('PI_UAE_Vehicle_Tracking_Fixed.png')
    plt.show()

if __name__ == "__main__":
    train_and_eval()
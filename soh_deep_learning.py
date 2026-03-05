import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# ================= 配置区 =================
# 只用好数据 (EV3已剔除)
RAW_FILES = ['LFP604EV2.csv', 'LFP604EV1.csv']
TIME_FORMAT = 'mixed'

# 物理窗口 (分类成功的那个窗口)
V_START = 538.0
V_END = 558.0

# 训练参数
BATCH_SIZE = 32
EPOCHS = 100 # 深度学习需要多跑几轮
LEARNING_RATE = 0.001
LAMBDA_RECON = 0.5 # 重构损失的权重 (0.5表示一半看形状，一半看SOH)

# 随机种子复现
torch.manual_seed(42)
np.random.seed(42)
# =========================================

# === 1. 数据集类 (PyTorch Dataset) ===
class BatteryDataset(Dataset):
    def __init__(self, raw_files, is_train=True):
        self.x_fingerprints = []
        self.y_soh = []
        
        # 临时存储用于拟合的数据
        all_segments = []
        
        print(f"正在读取数据并构建{'训练' if is_train else '测试'}集...")
        
        for f in raw_files:
            try:
                # 读取关键列
                df = pd.read_csv(f, usecols=['DATA_TIME', 'totalCurrent', 'totalVoltage', 'SOC'])
                df['DATA_TIME'] = pd.to_datetime(df['DATA_TIME'], format=TIME_FORMAT, dayfirst=False)
                df = df.dropna()
                
                # 分段处理
                curr_seg = []
                for _, row in df.iterrows():
                    if row['totalCurrent'] < -1.0: # 充电状态
                        curr_seg.append(row)
                    else:
                        if len(curr_seg) > 30:
                            df_seg = pd.DataFrame(curr_seg)
                            v_min, v_max = df_seg['totalVoltage'].min(), df_seg['totalVoltage'].max()
                            
                            # 截取电压片段
                            if v_min < V_START and v_max > V_END:
                                idx_s = (df_seg['totalVoltage'] - V_START).abs().idxmin()
                                idx_e = (df_seg['totalVoltage'] - V_END).abs().idxmin()
                                
                                if idx_e > idx_s:
                                    df_sub = df_seg.loc[idx_s:idx_e]
                                    
                                    # --- A. 提取电压指纹 (输入 X) ---
                                    # 强制重采样到 100 个点
                                    v_seq = df_sub['totalVoltage'].values
                                    if len(v_seq) > 10: # 只有点数够才处理
                                        f_interp = interp1d(np.linspace(0, 1, len(v_seq)), v_seq, kind='linear')
                                        fingerprint = f_interp(np.linspace(0, 1, 100))
                                        
                                        # 归一化指纹到 0-1 之间 (帮助神经网络收敛)
                                        f_min, f_max = fingerprint.min(), fingerprint.max()
                                        fingerprint_norm = (fingerprint - f_min) / (f_max - f_min + 1e-6)
                                        
                                        # --- B. 计算原始容量 (用于拟合真值) ---
                                        soc_delta = df_seg['SOC'].iloc[-1] - df_seg['SOC'].iloc[0]
                                        if soc_delta > 20.0:
                                            dt = df_seg['DATA_TIME'].diff().dt.total_seconds().fillna(10)
                                            ah = (df_seg['totalCurrent'].abs() * dt).sum() / 3600
                                            raw_cap = ah / (soc_delta / 100.0)
                                            
                                            days = (df_seg['DATA_TIME'].iloc[0] - pd.Timestamp("2020-01-01")).days
                                            
                                            all_segments.append({
                                                'file': f,
                                                'days': days,
                                                'raw_cap': raw_cap,
                                                'fingerprint': fingerprint_norm
                                            })
                            curr_seg = []
            except Exception as e:
                print(f"读取文件 {f} 出错: {e}")

        # === 2. 生成伪真值 (Pseudo-Label Generation) ===
        # 这是“无标签”学习的核心：用拟合曲线当真值
        df_all = pd.DataFrame(all_segments)
        # 清洗离群值
        df_all = df_all[(df_all['raw_cap'] > 400) & (df_all['raw_cap'] < 1500)]
        
        final_samples = []
        
        # 对每辆车单独拟合
        for f_name, group in df_all.groupby('file'):
            # 线性拟合 Cap = a*t + b
            z = np.polyfit(group['days'], group['raw_cap'], 1)
            p = np.poly1d(z)
            
            # 如果斜率为负(衰减)，则是有效数据
            if z[0] < 0:
                cap_init = p(group['days'].min())
                group['SOH_True'] = (p(group['days']) / cap_init) * 100
                
                # 存入最终列表
                for _, row in group.iterrows():
                    if 70 <= row['SOH_True'] <= 105:
                        final_samples.append({
                            'x': row['fingerprint'],
                            'y': row['SOH_True']
                        })
                print(f"   车辆 {f_name}: 生成 {len(group)} 条伪标签数据")

        # 划分训练/测试集 (简单的按比例切分)
        # 实际操作中，最好按时间或车辆切分，这里为了代码简单随机切
        import random
        random.shuffle(final_samples)
        split_idx = int(len(final_samples) * 0.8)
        
        if is_train:
            self.data = final_samples[:split_idx]
        else:
            self.data = final_samples[split_idx:]
            
        print(f"   数据集构建完成: {len(self.data)} 样本")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 转换为 Tensor
        x = torch.tensor(self.data[idx]['x'], dtype=torch.float32).unsqueeze(0) # (1, 100)
        y = torch.tensor(self.data[idx]['y'], dtype=torch.float32)
        return x, y

# === 3. 深度多任务网络 (Autoencoder + Regressor) ===
class SOH_Net(nn.Module):
    def __init__(self):
        super(SOH_Net, self).__init__()
        
        # --- 编码器 (Encoder): 提取形状特征 ---
        # 输入: (Batch, 1, 100)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2), # -> (16, 50)
            
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2), # -> (32, 25)
            
            nn.Flatten(),
            nn.Linear(32 * 25, 128),
            nn.ReLU()
        ) # 输出核心特征: 128维向量
        
        # --- 解码器 (Decoder): 重构曲线 (辅助任务) ---
        self.decoder = nn.Sequential(
            nn.Linear(128, 32 * 25),
            nn.ReLU(),
            nn.Unflatten(1, (32, 25)),
            
            nn.Upsample(scale_factor=2),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2),
            nn.Conv1d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid() # 输出归一化后的曲线 (0-1)
        )
        
        # --- 回归头 (Regressor): 预测 SOH (主任务) ---
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2), # 防止过拟合
            nn.Linear(64, 1) # 输出 SOH
        )

    def forward(self, x):
        features = self.encoder(x)
        x_recon = self.decoder(features)
        soh_pred = self.regressor(features)
        return soh_pred, x_recon

# === 4. 训练流程 ===
def train_model():
    # 准备数据
    train_ds = BatteryDataset(RAW_FILES, is_train=True)
    test_ds = BatteryDataset(RAW_FILES, is_train=False)
    
    if len(train_ds) == 0:
        print("没有有效数据！请检查 CSV 文件路径或内容。")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model = SOH_Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_mse = nn.MSELoss()
    
    print("\n🚀 开始深度学习训练 (Multi-Task Learning)...")
    train_losses = []
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for x, y_true in train_loader:
            x, y_true = x.to(device), y_true.to(device)
            
            # 前向传播
            soh_pred, x_recon = model(x)
            
            # 计算损失
            loss_soh = criterion_mse(soh_pred.squeeze(), y_true) # 任务1: SOH 准不准
            loss_recon = criterion_mse(x_recon, x)               # 任务2: 曲线像不像
            
            # ⭐️ 核心公式: 总损失 = SOH损失 + λ * 重构损失
            loss = loss_soh + LAMBDA_RECON * loss_recon * 100 # *100是为了平衡量级
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(train_loader):.4f}")
            train_losses.append(epoch_loss)

    # === 5. 测试与评估 ===
    model.eval()
    y_trues = []
    y_preds = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            soh_pred, _ = model(x)
            y_preds.extend(soh_pred.cpu().numpy().flatten())
            y_trues.extend(y.numpy())
            
    rmse = np.sqrt(mean_squared_error(y_trues, y_preds))
    r2 = r2_score(y_trues, y_preds)
    
    print("\n" + "="*40)
    print(f"🔥 Deep Learning 最终结果:")
    print(f"   RMSE : {rmse:.4f} %")
    print(f"   R²   : {r2:.4f}")
    print("="*40)
    
    # 画图
    plt.figure(figsize=(10, 5))
    
    # 结果散点图
    plt.subplot(1, 2, 1)
    plt.scatter(y_trues, y_preds, alpha=0.6, color='purple', s=40)
    plt.plot([min(y_trues), max(y_trues)], [min(y_trues), max(y_trues)], 'r--')
    plt.xlabel("True SOH (Trend)")
    plt.ylabel("Predicted SOH (Deep Net)")
    plt.title(f"Deep MTL Result\nR2={r2:.3f}")
    plt.grid(True, alpha=0.3)
    
    # 训练Loss曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_losses)
    plt.xlabel("Epochs (x10)")
    plt.ylabel("Loss")
    plt.title("Training Convergence")
    
    plt.tight_layout()
    plt.savefig('Deep_Learning_Result.png')
    plt.show()

if __name__ == "__main__":
    train_model()
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
# 只用好数据 (EV3已剔除)
RAW_FILES = ['LFP604EV2.csv', 'LFP604EV1.csv']
TIME_FORMAT = 'mixed'

# 物理窗口 (分类成功的那个窗口)
V_START = 538.0
V_END = 558.0

# 训练参数
BATCH_SIZE = 32
EPOCHS = 200         # 稍微多跑几轮
LEARNING_RATE = 0.0005 # ⚡️ 调小学习率，防止震荡
LAMBDA_RECON = 0.5   # 重构损失权重

# ⚡️ SOH 归一化范围 (关键修改!)
SOH_MIN = 60.0
SOH_MAX = 110.0

# 随机种子复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
# =========================================

# === 1. 数据集类 (PyTorch Dataset) ===
class BatteryDataset(Dataset):
    def __init__(self, raw_files, is_train=True):
        self.data = []
        
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
                                    if len(v_seq) > 10: 
                                        f_interp = interp1d(np.linspace(0, 1, len(v_seq)), v_seq, kind='linear')
                                        fingerprint = f_interp(np.linspace(0, 1, 100))
                                        
                                        # 归一化指纹到 0-1
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
        df_all = pd.DataFrame(all_segments)
        # 清洗离群值
        if not df_all.empty:
            df_all = df_all[(df_all['raw_cap'] > 400) & (df_all['raw_cap'] < 1500)]
            
            final_samples = []
            
            # 对每辆车单独拟合
            for f_name, group in df_all.groupby('file'):
                if len(group) > 20:
                    z = np.polyfit(group['days'], group['raw_cap'], 1)
                    p = np.poly1d(z)
                    
                    if z[0] < 0: # 必须衰减
                        cap_init = p(group['days'].min())
                        group['SOH_True'] = (p(group['days']) / cap_init) * 100
                        
                        for _, row in group.iterrows():
                            if 60 <= row['SOH_True'] <= 110:
                                # ⚡️ 关键修改: SOH 归一化到 0-1
                                y_norm = (row['SOH_True'] - SOH_MIN) / (SOH_MAX - SOH_MIN)
                                
                                final_samples.append({
                                    'x': row['fingerprint'],
                                    'y': y_norm 
                                })
                        print(f"   车辆 {f_name}: 生成 {len(group)} 条数据")

            # 划分训练/测试集
            random.shuffle(final_samples)
            split_idx = int(len(final_samples) * 0.8)
            
            if is_train:
                self.data = final_samples[:split_idx]
            else:
                self.data = final_samples[split_idx:]
                
            print(f"   数据集构建完成: {len(self.data)} 样本")
        else:
            print("   未提取到有效数据！")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx]['x'], dtype=torch.float32).unsqueeze(0) # (1, 100)
        y = torch.tensor(self.data[idx]['y'], dtype=torch.float32)
        return x, y

# === 3. 深度多任务网络 (Autoencoder + Regressor) ===
class SOH_Net(nn.Module):
    def __init__(self):
        super(SOH_Net, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Flatten(),
            nn.Linear(32 * 25, 128),
            nn.ReLU()
        )
        
        # 解码器 (自监督分支)
        self.decoder = nn.Sequential(
            nn.Linear(128, 32 * 25),
            nn.ReLU(),
            nn.Unflatten(1, (32, 25)),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )
        
        # 回归头 (SOH预测分支)
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid() # ⚡️ 限制输出在 0-1 之间
        )

    def forward(self, x):
        features = self.encoder(x)
        x_recon = self.decoder(features)
        soh_pred = self.regressor(features)
        return soh_pred, x_recon

# === 4. 训练流程 ===
def train_model():
    train_ds = BatteryDataset(RAW_FILES, is_train=True)
    test_ds = BatteryDataset(RAW_FILES, is_train=False)
    
    if len(train_ds) == 0: return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
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
            
            soh_pred, x_recon = model(x)
            
            # Loss 计算 (都在 0-1 范围内)
            loss_soh = criterion_mse(soh_pred.squeeze(), y_true) 
            loss_recon = criterion_mse(x_recon, x)              
            
            loss = loss_soh + LAMBDA_RECON * loss_recon
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}") # 这里的Loss应该是 0.0xxx

    # === 5. 测试与评估 (自动反归一化) ===
    model.eval()
    y_trues_denorm = []
    y_preds_denorm = []
    
    print("\n正在评估测试集...")
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            soh_pred, _ = model(x)
            
            # ⚡️ 反归一化：把 0-1 还原回 60-110
            pred_val = soh_pred.cpu().numpy().flatten()
            true_val = y.numpy()
            
            pred_denorm = pred_val * (SOH_MAX - SOH_MIN) + SOH_MIN
            true_denorm = true_val * (SOH_MAX - SOH_MIN) + SOH_MIN
            
            y_preds_denorm.extend(pred_denorm)
            y_trues_denorm.extend(true_denorm)
            
    rmse = np.sqrt(mean_squared_error(y_trues_denorm, y_preds_denorm))
    r2 = r2_score(y_trues_denorm, y_preds_denorm)
    
    print("\n" + "="*40)
    print(f"🔥 Deep Learning 最终修正结果:")
    print(f"   RMSE : {rmse:.4f} %")
    print(f"   R²   : {r2:.4f}")
    print("="*40)
    
    # 画图
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_trues_denorm, y_preds_denorm, alpha=0.6, color='purple', s=40)
    plt.plot([SOH_MIN, SOH_MAX], [SOH_MIN, SOH_MAX], 'r--', linewidth=2)
    plt.xlim(70, 105)
    plt.ylim(70, 105)
    plt.xlabel("True SOH")
    plt.ylabel("Predicted SOH")
    plt.title(f"Deep MTL Result (Corrected)\nR2={r2:.3f}")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss (Normalized)")
    plt.title("Training Curve (Should be smooth)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('Deep_Learning_Result_Final2.png')
    plt.show()

if __name__ == "__main__":
    train_model()

# 将此代码追加到 soh_deep_learning_v2.py 的末尾，或者直接替换原来的 main 函数部分
# (前提是你刚刚训练完，变量都在)

def visualize_reconstruction(model, test_loader, device):
    model.eval()
    
    # 从测试集中随机抓取一个 batch
    data_iter = iter(test_loader)
    x, y = next(data_iter)
    x = x.to(device)
    
    # 预测
    with torch.no_grad():
        _, x_recon = model(x)
    
    # 转为 numpy
    x_np = x.cpu().numpy()
    x_recon_np = x_recon.cpu().numpy()
    
    # 画图：展示模型眼中的“电压指纹”
    plt.figure(figsize=(12, 4))
    
    for i in range(3): # 画前3个样本
        plt.subplot(1, 3, i+1)
        plt.plot(x_np[i, 0, :], 'b-', label='Original (Noisy)', linewidth=2, alpha=0.5)
        plt.plot(x_recon_np[i, 0, :], 'r--', label='Reconstructed (Clean)', linewidth=2)
        plt.title(f"Sample {i+1}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    plt.suptitle("Does the Model Understand the Shape? (Autoencoder Result)", fontsize=14)
    plt.tight_layout()
    plt.savefig('Model_Reconstruction_Check.png')
    plt.show()
    print("已生成重构对比图：Model_Reconstruction_Check.png")
    print("如果红线能紧贴蓝线（且更平滑），说明模型成功学到了去噪后的物理特征！")

# 如果你想直接运行可视化，请在 train_model() 的最后调用这个函数：
# visualize_reconstruction(model, test_loader, device)
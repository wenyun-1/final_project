import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time

# 引入我们刚才写的类
from SOC_AttentionGRU_Gated import SOH_Gated_GRU
from SOC_RealVehicleDataset import RealVehicleDataset

# ================= 参数配置 =================
BATCH_SIZE = 512       # 批次大小，显存够大可以改到 1024
LR = 0.001             # 学习率
EPOCHS = 20            # 训练轮数
WINDOW_SIZE = 30       # 时间窗口
SAMPLE_STRIDE = 10     # 重要：每10行取1行，防止内存爆炸
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FILE_PATH = "Processed_All_Bus_Data.csv"
# ===========================================

def train():
    print(f"正在使用设备: {DEVICE}")
    
    # 1. 加载数据
    # sample_stride=10 意味着只加载 1/10 的数据用于训练，极大节省内存
    dataset = RealVehicleDataset(FILE_PATH, window_size=WINDOW_SIZE, sample_stride=SAMPLE_STRIDE)
    
    # 划分训练集和测试集 (80% 训练, 20% 测试)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    
    # 2. 初始化模型
    model = SOH_Gated_GRU().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss() # 回归问题用均方误差
    
    print("开始训练...")
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for i, (x, soh, y) in enumerate(train_loader):
            x, soh, y = x.to(DEVICE), soh.to(DEVICE), y.to(DEVICE)
            
            # === SOH Dropout 策略 ===
            # 30% 概率将 SOH 设为 None (模拟无 SOH 的情况)
            # 这强迫模型既能“盲跑”，又能利用 SOH 提升
            if np.random.rand() < 0.3:
                soh_input = None
            else:
                soh_input = soh
            
            optimizer.zero_grad()
            pred = model(x, soh_input)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.6f}")
        
        avg_train_loss = train_loss / len(train_loader)
        
# === 验证环节 ===
        model.eval()
        val_mae_loss = 0.0 # 改为记录真实的 MAE (百分比)
        with torch.no_grad():
            for x, soh, y in test_loader:
                x, soh, y = x.to(DEVICE), soh.to(DEVICE), y.to(DEVICE)
                pred = model(x, soh)
                
                # 【核心修改：反归一化】
                # 将 0~1 的值乘回 100，还原为 0~100% 的真实物理电量
                pred_real = pred * 100.0
                y_real = y * 100.0
                
                # 计算真实的平均绝对误差 (MAE)
                # torch.abs() 求绝对值，.mean() 求平均
                mae = torch.abs(pred_real - y_real).mean().item()
                val_mae_loss += mae
        
        avg_val_mae = val_mae_loss / len(test_loader)
        
        # 打印时的提示也改了，明确这是真实百分比误差
        print(f"Epoch [{epoch+1}] 完成 | 耗时: {time.time()-start_time:.1f}s | Train Loss(MSE): {avg_train_loss:.6f} | Val MAE (真实电量误差): {avg_val_mae:.2f}%")
        
        # 保存最优模型 (现在根据真实 MAE 来保存)
        if avg_val_mae < best_loss:
            best_loss = avg_val_mae
            torch.save(model.state_dict(), "Best_SOH_Model.pth")
            print(f">>> 模型已保存 (最佳真实误差: {best_loss:.2f}%)")

if __name__ == "__main__":
    train()
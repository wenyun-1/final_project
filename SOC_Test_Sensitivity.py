import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from SOC_AttentionGRU_Gated import SOH_Gated_GRU

# ================= 配置 =================
TEST_FILE_PATH = "D:/研究/1-大论文/数据集/TEG6105BEV13_LFP604/data/TEG6105BEV13_LFP604EV8sample.csv"
MODEL_PATH = "Best_SOH_Model.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WINDOW_SIZE = 30
# =======================================

def load_data_raw(file_path):
    # 只加载基础数据，不设 SOH
    cols = ['totalVoltage', 'totalCurrent', 'maxTemperature', 'minTemperature', 'SOC']
    try:
        df = pd.read_csv(file_path, encoding='gbk', usecols=cols)
    except:
        df = pd.read_csv(file_path, encoding='utf-8', usecols=cols)
        
    df['totalVoltage'] = pd.to_numeric(df['totalVoltage'], errors='coerce')
    df = df[df['totalVoltage'] > 100].reset_index(drop=True)
    
    # 归一化
    df['Norm_Current'] = ((df['totalCurrent'] - (-500)) / (500 - (-500))).clip(-1, 1)
    df['Norm_Voltage'] = ((df['totalVoltage'] - 200) / (800 - 200)).clip(0, 1)
    df['maxTemperature'] = pd.to_numeric(df['maxTemperature'], errors='coerce').fillna(25)
    df['minTemperature'] = pd.to_numeric(df['minTemperature'], errors='coerce').fillna(25)
    temp_mean = (df['maxTemperature'] + df['minTemperature']) / 2
    df['Norm_Temp'] = ((temp_mean - (-30)) / (60 - (-30))).clip(0, 1)
    
    if df['SOC'].max() > 1.5: df['SOC'] = df['SOC'] / 100.0
    return df

def sensitivity_test():
    df = load_data_raw(TEST_FILE_PATH)
    # 取中间一段动态较多的数据
    segment = df.iloc[-3000:-1000].reset_index(drop=True)
    
    inputs = segment[['Norm_Current', 'Norm_Voltage', 'Norm_Temp']].values
    true_soc = segment['SOC'].values[WINDOW_SIZE:]
    
    model = SOH_Gated_GRU().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # === 扫描 SOH 从 0.5 到 1.0 ===
    soh_values = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    mae_results = []
    
    # 先测 Baseline (No SOH)
    pred_base = []
    with torch.no_grad():
        for i in range(len(inputs) - WINDOW_SIZE):
            x = torch.tensor(inputs[i:i+WINDOW_SIZE], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            out = model(x, soh=None)
            pred_base.append(out.item())
    mae_base = np.mean(np.abs(np.array(pred_base) - true_soc))
    print(f"Baseline (No SOH) MAE: {mae_base:.5f}")
    
    # 开始扫描
    print("\n开始 SOH 敏感度扫描...")
    for s_val in soh_values:
        preds = []
        with torch.no_grad():
            s_tensor = torch.tensor([s_val], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            for i in range(len(inputs) - WINDOW_SIZE):
                x = torch.tensor(inputs[i:i+WINDOW_SIZE], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                # 强制输入当前的 s_val
                out = model(x, soh=s_tensor)
                preds.append(out.item())
        
        mae = np.mean(np.abs(np.array(preds) - true_soc))
        mae_results.append(mae)
        print(f"SOH = {s_val:.2f} -> MAE: {mae:.5f}")

    # === 画图 ===
    plt.figure(figsize=(10, 6))
    plt.plot(soh_values, mae_results, 'bo-', label='With SOH (Ours)')
    plt.axhline(y=mae_base, color='r', linestyle='--', label='Baseline (No SOH)')
    
    plt.title("Sensitivity Analysis: Impact of Input SOH on SOC Error")
    plt.xlabel("Input SOH Value")
    plt.ylabel("MAE (Mean Absolute Error)")
    plt.legend()
    plt.grid(True)
    plt.savefig("SOH_Sensitivity.png")
    plt.show()
    print("✅ 分析完成！请看 SOH_Sensitivity.png，找到蓝线最低点对应的 SOH。")

if __name__ == "__main__":
    sensitivity_test()
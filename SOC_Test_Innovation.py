import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from AttentionGRU_Gated import SOH_Gated_GRU

# ================= 配置区域 =================
# 建议挑一辆老化明显的车，比如 EV8
TEST_FILE_PATH = "D:/研究/1-大论文/数据集/TEG6105BEV13_LFP604/data/TEG6105BEV13_LFP604EV8sample.csv" 
SOH_MAPPING_FILE = "SOH_Predictions_For_SOC.csv"
MODEL_PATH = "Best_SOH_Model.pth" 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WINDOW_SIZE = 30
# ===========================================

def run_integrated_test():
    import os
    veh_name = os.path.basename(TEST_FILE_PATH).split('.')[0]
    print(f"1. 正在加载寿命末期测试车辆: {veh_name} ...")
    
    target_cols = ['totalVoltage', 'totalCurrent', 'maxTemperature', 'minTemperature', 'SOC', 'DATA_TIME']
    try: df = pd.read_csv(TEST_FILE_PATH, encoding='gbk', usecols=target_cols)
    except: df = pd.read_csv(TEST_FILE_PATH, encoding='utf-8', usecols=target_cols)

    df['totalVoltage'] = pd.to_numeric(df['totalVoltage'], errors='coerce')
    df = df.dropna(subset=['totalVoltage'])
    df = df[df['totalVoltage'] > 100].reset_index(drop=True)
    df['DATA_TIME'] = pd.to_datetime(df['DATA_TIME'])
    df['Days'] = (df['DATA_TIME'] - pd.Timestamp("2020-01-01")).dt.days
    
    # ⚡️ 截取该车辆寿命最末期的一段动态放电数据 (比如最后一万个点中的一段)
    test_segment = df.iloc[-10000:-8000].reset_index(drop=True)
    
    # 注入第三章的 SOH 预测先验
    soh_mapping = pd.read_csv(SOH_MAPPING_FILE)
    veh_soh = soh_mapping[soh_mapping['Vehicle'] == veh_name].sort_values('Days')
    test_segment = pd.merge_asof(test_segment, veh_soh[['Days', 'Pred_SOH']], on='Days', direction='nearest')
    test_segment['SOH'] = test_segment['Pred_SOH']
    actual_soh_value = test_segment['SOH'].mean()
    print(f"   -> 截取时间段平均老化 SOH = {actual_soh_value*100:.2f}%")

    # 归一化操作
    test_segment['Norm_Current'] = ((test_segment['totalCurrent'] - (-500)) / (500 - (-500))).clip(-1, 1)
    test_segment['Norm_Voltage'] = ((test_segment['totalVoltage'] - 200) / (800 - 200)).clip(0, 1)
    test_segment['maxTemperature'] = pd.to_numeric(test_segment['maxTemperature'], errors='coerce').fillna(25)
    test_segment['minTemperature'] = pd.to_numeric(test_segment['minTemperature'], errors='coerce').fillna(25)
    temp_mean = (test_segment['maxTemperature'] + test_segment['minTemperature']) / 2
    test_segment['Norm_Temp'] = ((temp_mean - (-30)) / (60 - (-30))).clip(0, 1)
    if test_segment['SOC'].max() > 1.5: test_segment['SOC'] = test_segment['SOC'] / 100.0
    
    inputs = test_segment[['Norm_Current', 'Norm_Voltage', 'Norm_Temp']].values
    soh_val = test_segment['SOH'].values
    true_soc = test_segment['SOC'].values
    
    model = SOH_Gated_GRU().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    pred_with_soh, pred_no_soh = [], []
    
    print("2. 启动宏微观协同验证推理...")
    with torch.no_grad():
        for i in range(len(test_segment) - WINDOW_SIZE):
            x_tensor = torch.tensor(inputs[i : i+WINDOW_SIZE], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            # Ous: 有 PI-UAE 老化参数辅助
            s_tensor = torch.tensor([soh_val[i+WINDOW_SIZE-1]], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            pred_with_soh.append(model(x_tensor, soh=s_tensor).item())
            
            # Baseline: 无老化参数辅助 (直接给 None)
            pred_no_soh.append(model(x_tensor, soh=None).item())

    # 结果对比
    true_soc_real = true_soc[WINDOW_SIZE:] * 100.0
    pred_with_soh_real = np.array(pred_with_soh) * 100.0
    pred_no_soh_real = np.array(pred_no_soh) * 100.0
    
    mae_A = np.mean(np.abs(pred_with_soh_real - true_soc_real))
    mae_B = np.mean(np.abs(pred_no_soh_real - true_soc_real))
    
    print(f"\n======== 联合估计成果验证 ========")
    print(f"Ours (联合第3章SOH): MAE = {mae_A:.2f}%")
    print(f"Baseline (仅传统微观GRU): MAE = {mae_B:.2f}%")
    print(f"老化门控机制带来的精度提升: {(mae_B - mae_A)/mae_B * 100:.2f}%")
    
    # 震撼画图
    plt.figure(figsize=(12, 6))
    plt.plot(true_soc_real, 'k-', label='Ground Truth SOC (%)', linewidth=2.5)
    plt.plot(pred_with_soh_real, 'r--', label=f'Ours: Macro-Micro Co-estimation (MAE={mae_A:.2f}%)', linewidth=2)
    plt.plot(pred_no_soh_real, 'g:', label=f'Baseline: Standard GRU (MAE={mae_B:.2f}%)', linewidth=1.5, alpha=0.8)
    
    plt.title(f"SOC Co-estimation Validation at End-of-Life (Actual SOH $\approx$ {actual_soh_value*100:.1f}%)")
    plt.xlabel("Time Step (x10 seconds)")
    plt.ylabel("State of Charge (SOC) [%]")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.savefig("Macro_Micro_Coestimation_Result.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    run_integrated_test()
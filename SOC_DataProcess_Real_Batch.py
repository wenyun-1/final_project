import pandas as pd
import numpy as np
import os
import glob
import gc

# ================= 配置区域 =================
DATA_FOLDER_PATH = "D:/研究/1-大论文/数据集/TEG6105BEV13_LFP604/data/"
PROCESSED_FILE_PATH = "Processed_All_Bus_Data.csv"
# 注意路径里的斜杠最好用正斜杠 / 或者在字符串前加 r
SOH_MAPPING_FILE = "D:/研究/1-大论文/数据集/第三章/SOH_Predictions_For_SOC.csv"

USE_COLS = ['DATA_TIME', 'totalVoltage', 'totalCurrent', 'maxTemperature', 'minTemperature', 'SOC']
MIN_VALID_VOLTAGE = 100.0 
GAP_THRESHOLD_SEC = 1800 
# ===========================================

def process_integrated():
    csv_files = glob.glob(os.path.join(DATA_FOLDER_PATH, "*.csv"))
    print(f"📂 找到 {len(csv_files)} 个高频运行文件，启动【宏微观联合特征对齐】...")

    if os.path.exists(PROCESSED_FILE_PATH): os.remove(PROCESSED_FILE_PATH)
    
    # 1. 加载第三章的 SOH 预测结果
    if not os.path.exists(SOH_MAPPING_FILE):
        raise FileNotFoundError(f"找不到 {SOH_MAPPING_FILE}！请先运行 soh_eval.py 导出 SOH 数据。")
    soh_mapping = pd.read_csv(SOH_MAPPING_FILE)
    
    global_cycle_offset = 0 
    
    for i, file_path in enumerate(csv_files):
        file_name = os.path.basename(file_path)
        veh_name = file_name.split('.')[0] # 获取如 TEG6105BEV13_LFP604EV0sample
        print(f"\n>>> [{i+1}/{len(csv_files)}] 正在处理: {veh_name}")

        try:
            # 读取高频微观数据
            try: df = pd.read_csv(file_path, encoding='gbk', usecols=USE_COLS, low_memory=False)
            except: df = pd.read_csv(file_path, encoding='utf-8', usecols=USE_COLS, low_memory=False)
            
            for col in ['totalVoltage', 'totalCurrent', 'maxTemperature', 'minTemperature', 'SOC']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            df['DATA_TIME'] = pd.to_datetime(df['DATA_TIME'])
            df = df.dropna(subset=['totalVoltage'])
            df = df[df['totalVoltage'] > MIN_VALID_VOLTAGE].copy()
            df = df.sort_values('DATA_TIME').reset_index(drop=True)
            if len(df) == 0: continue

            # 划分行程
            df['time_diff'] = df['DATA_TIME'].diff().dt.total_seconds().fillna(0)
            internal_cycle_ids = (df['time_diff'] > GAP_THRESHOLD_SEC).astype(int).cumsum() + 1
            df['Cycle_ID'] = internal_cycle_ids + global_cycle_offset
            global_cycle_offset = df['Cycle_ID'].max()

            # ================= ⚡️ 核心合龙逻辑：多时间尺度对齐 ⚡️ =================
            # 计算每一行高频数据对应的宏观“天数 (Days)”
            df['Days'] = (df['DATA_TIME'] - pd.Timestamp("2020-01-01")).dt.days
            
            # 从映射表中取出该车专属的老化轨迹
            veh_soh_trajectory = soh_mapping[soh_mapping['Vehicle'] == veh_name].copy()
            
            if len(veh_soh_trajectory) > 0:
                # 按照“天数”进行就近匹配 (因为高频数据是按秒的，SOH是按天的)
                veh_soh_trajectory = veh_soh_trajectory.sort_values('Days')
                df = pd.merge_asof(df, veh_soh_trajectory[['Days', 'Pred_SOH']], on='Days', direction='nearest')
                df['SOH'] = df['Pred_SOH']
                print(f"    ✅ 成功注入第3章 PI-UAE 老化参数! SOH 范围: {df['SOH'].min():.3f} ~ {df['SOH'].max():.3f}")
            else:
                # 如果这辆车没在第三章处理过，赋予默认值 1.0
                df['SOH'] = 1.0
                print(f"    ⚠️ 警告：映射表中没有该车的老化记录，已默认填充 SOH=1.0")
            # ====================================================================

            temp_mean = ((df['maxTemperature'] + df['minTemperature']) / 2).ffill()
            final_df = pd.DataFrame({
                'Current': df['totalCurrent'],  
                'Voltage': df['totalVoltage'],
                'Temperature': temp_mean,
                'SOC': df['SOC'],
                'Cycle_ID': df['Cycle_ID'], 
                'SOH': df['SOH']  # 此时的 SOH 已经是具有高度物理意义的预测值了！
            }).astype('float32')

            write_header = (i == 0)
            final_df.to_csv(PROCESSED_FILE_PATH, mode='a', index=False, header=write_header)
            
            del df; del final_df; gc.collect()
        except Exception as e:
            print(f"    ❌ 处理出错: {e}")

    print(f"\n🎉 宏微观数据合龙完成！联合数据集位于: {PROCESSED_FILE_PATH}")

if __name__ == "__main__":
    process_integrated()
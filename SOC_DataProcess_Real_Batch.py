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
SOH_BASE_DATE = "2020-01-01"
INITIAL_SOH_POLICY = "drop_until_first_valid"  # "drop_until_first_valid" 或 "fill_default"
DEFAULT_INITIAL_SOH = 1.0
# ===========================================


def _prepare_soh_trajectory(soh_mapping_df, veh_name):
    """
    统一整理某辆车 SOH 时间轨迹，供 merge_asof(direction='backward') 前向保持使用。
    优先读取真实时间戳；若只有 Days 列，则退化为基准日+天数。
    """
    veh_soh = soh_mapping_df[soh_mapping_df['Vehicle'] == veh_name].copy()
    if len(veh_soh) == 0:
        return veh_soh

    # 若有“完整充电 / 有效估计”标记，优先过滤，避免碎片充电污染 SOH 更新点
    for valid_col in ['Is_Valid', 'is_valid', 'Valid', 'valid', 'Is_Complete_Charge', 'is_complete_charge']:
        if valid_col in veh_soh.columns:
            veh_soh = veh_soh[veh_soh[valid_col].astype(str).isin(['1', 'True', 'true'])].copy()
            break

    if len(veh_soh) == 0:
        return veh_soh

    if 'Pred_SOH' not in veh_soh.columns:
        raise KeyError("SOH 映射表缺少 Pred_SOH 列，无法注入 SOC 训练数据。")

    if 'Charge_End_Time' in veh_soh.columns:
        veh_soh['SOH_TIME'] = pd.to_datetime(veh_soh['Charge_End_Time'], errors='coerce')
    elif 'DATA_TIME' in veh_soh.columns:
        veh_soh['SOH_TIME'] = pd.to_datetime(veh_soh['DATA_TIME'], errors='coerce')
    elif 'Timestamp' in veh_soh.columns:
        veh_soh['SOH_TIME'] = pd.to_datetime(veh_soh['Timestamp'], errors='coerce')
    elif 'Days' in veh_soh.columns:
        # 兼容仅有“天级索引”的映射文件
        veh_soh['SOH_TIME'] = pd.to_datetime(SOH_BASE_DATE) + pd.to_timedelta(veh_soh['Days'], unit='D')
    else:
        raise KeyError("SOH 映射表缺少可用于时间对齐的列（Charge_End_Time/DATA_TIME/Timestamp/Days）。")

    veh_soh = veh_soh.dropna(subset=['SOH_TIME', 'Pred_SOH']).copy()
    veh_soh = veh_soh.sort_values('SOH_TIME')
    return veh_soh[['SOH_TIME', 'Pred_SOH']]

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

            # ================= ⚡️ 核心合龙逻辑：SOH 前向保持（零阶保持）⚡️ =================
            # 用“最近一次完成充电计算出的 SOH”对后续放电高频数据做方向为 backward 的时间对齐，
            # 从而严格满足在线因果性，避免未来 SOH 泄露。
            df = df.sort_values('DATA_TIME').reset_index(drop=True)
            veh_soh_trajectory = _prepare_soh_trajectory(soh_mapping, veh_name)

            if len(veh_soh_trajectory) > 0:
                df = pd.merge_asof(
                    df,
                    veh_soh_trajectory,
                    left_on='DATA_TIME',
                    right_on='SOH_TIME',
                    direction='backward',
                    allow_exact_matches=True,
                )
                df['SOH'] = df['Pred_SOH']

                if INITIAL_SOH_POLICY == 'fill_default':
                    df['SOH'] = df['SOH'].fillna(DEFAULT_INITIAL_SOH)
                elif INITIAL_SOH_POLICY == 'drop_until_first_valid':
                    df = df[df['SOH'].notna()].copy()
                else:
                    raise ValueError(f"未知 INITIAL_SOH_POLICY: {INITIAL_SOH_POLICY}")

                if len(df) == 0:
                    print("    ⚠️ 当前文件所有数据都早于首个有效 SOH 更新点，已跳过。")
                    continue

                print(f"    ✅ 已按前向保持注入 SOH（backward asof），SOH 范围: {df['SOH'].min():.3f} ~ {df['SOH'].max():.3f}")
            else:
                df['SOH'] = DEFAULT_INITIAL_SOH
                print(f"    ⚠️ 映射表中无该车有效 SOH 轨迹，已默认填充 SOH={DEFAULT_INITIAL_SOH:.3f}")
            # ======================================================================================

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

import pandas as pd
import numpy as np

# ================= 修改区域 =================
# 请将下面这个路径改为你电脑上任意一个大巴车数据文件的真实路径
# 注意：如果是Windows路径，请使用双斜杠 \\ 或反斜杠 /，例如 "D:/MyData/Bus01.csv"
file_path = "TEG6105BEV13_LFP604EV0sample.csv" 
# ===========================================

def inspect_data():
    print(">>> 正在读取数据，请稍候...")
    try:
        # 尝试读取，如果是 Excel 文件请改用 pd.read_excel(file_path)
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path) # 如果有编码报错，尝试加 encoding='gbk'
        
        print("\n" + "="*30)
        print("1. 数据概览 (前5行):")
        print("="*30)
        print(df.head())

        print("\n" + "="*30)
        print("2. 列名列表 (请直接复制给我):")
        print("="*30)
        print(list(df.columns))

        print("\n" + "="*30)
        print("3. 数据统计信息:")
        print("="*30)
        print(f"总行数: {len(df)}")
        
        # 猜测可能的时间列
        time_cols = [c for c in df.columns if 'time' in c.lower() or '时间' in c]
        if time_cols:
            print(f"检测到时间列: {time_cols}")
            
        # 猜测可能的电压电流列
        volt_cols = [c for c in df.columns if 'volt' in c.lower() or '电压' in c]
        curr_cols = [c for c in df.columns if 'curr' in c.lower() or '电流' in c]
        print(f"检测到电压列: {volt_cols}")
        print(f"检测到电流列: {curr_cols}")

        print("\n" + "="*30)
        print("4. 关键问题检查 (这决定了怎么写代码):")
        print("="*30)
        # 检查是否有行程/循环的标记
        cycle_keywords = ['cycle', 'trip', 'segment', '循环', '行程', '片段']
        found_cycle = [c for c in df.columns if any(k in c.lower() for k in cycle_keywords)]
        if found_cycle:
            print(f"[√] 发现行程/循环标记列: {found_cycle} -> 我们可以根据这个来更新SOH")
            print(f"    该列的前几个唯一值: {df[found_cycle[0]].unique()[:5]}")
        else:
            print("[!] 未发现明显的行程/循环标记列。")
            print("    (之后我们需要根据时间间隔或充电状态来手动划分数据段，以便定期更新SOH)")

    except Exception as e:
        print(f"读取出错: {e}")
        print("建议检查：1. 文件路径是否正确？ 2. 文件是不是开着？ 3. 是否需要 encoding='gbk'？")

if __name__ == "__main__":
    inspect_data()
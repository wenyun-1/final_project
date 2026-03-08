import argparse
import gc
import glob
import os
import re

import pandas as pd

# ================= 默认配置（可被命令行覆盖） =================
DATA_FOLDER_PATH = "data"
PROCESSED_FILE_PATH = "Processed_All_Bus_Data.csv"
SOH_MAPPING_FILE = "outputs_final_best/SOH_Predictions_For_SOC.csv"

USE_COLS = ['DATA_TIME', 'totalVoltage', 'totalCurrent', 'maxTemperature', 'minTemperature', 'SOC']
MIN_VALID_VOLTAGE = 100.0
GAP_THRESHOLD_SEC = 1800
SOH_BASE_DATE = "2020-01-01"
INITIAL_SOH_POLICY = "drop_until_first_valid"  # "drop_until_first_valid" 或 "fill_default"
DEFAULT_INITIAL_SOH = 1.0
VEHICLE_SPLIT_FILE = "outputs_final/vehicle_split.csv"
# ===========================================================


def _normalize_vehicle_key(name):
    if not isinstance(name, str):
        return ""
    s = name.strip().upper()
    m = re.search(r'(LFP\d+EV\d+)', s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return s


def _load_allowed_vehicle_keys(split_file, split_role):
    if not split_file:
        return None
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"找不到 vehicle_split 文件: {split_file}")
    split_df = pd.read_csv(split_file)
    if not {"Vehicle", "Role"}.issubset(split_df.columns):
        raise KeyError("vehicle_split.csv 需要包含 Vehicle, Role 两列。")
    if split_role not in {"train", "test", "all"}:
        raise ValueError("split_role 仅支持 train/test/all")
    if split_role == "all":
        target = split_df.copy()
    else:
        target = split_df[split_df["Role"].astype(str).str.lower() == split_role].copy()
    keys = {_normalize_vehicle_key(v) for v in target["Vehicle"].astype(str).tolist()}
    return keys


def _load_allowed_vehicle_keys(split_file, split_role):
    if not split_file:
        return None
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"找不到 vehicle_split 文件: {split_file}")
    split_df = pd.read_csv(split_file)
    if not {"Vehicle", "Role"}.issubset(split_df.columns):
        raise KeyError("vehicle_split.csv 需要包含 Vehicle, Role 两列。")
    if split_role not in {"train", "test", "all"}:
        raise ValueError("split_role 仅支持 train/test/all")
    if split_role == "all":
        target = split_df.copy()
    else:
        target = split_df[split_df["Role"].astype(str).str.lower() == split_role].copy()
    keys = {_normalize_vehicle_key(v) for v in target["Vehicle"].astype(str).tolist()}
    return keys


def _resolve_vehicle_rows(soh_mapping_df, veh_name):
    """支持 Vehicle 命名不一致（如 LFP604EV1 vs TEG...EV1sample）的稳健匹配。"""
    direct = soh_mapping_df[soh_mapping_df['Vehicle'] == veh_name].copy()
    if len(direct) > 0:
        return direct

    target_key = _normalize_vehicle_key(veh_name)
    if not target_key:
        return direct

    tmp = soh_mapping_df.copy()
    tmp['_veh_key'] = tmp['Vehicle'].astype(str).map(_normalize_vehicle_key)
    fuzzy = tmp[tmp['_veh_key'] == target_key].copy()
    if len(fuzzy) > 0:
        print(f"    ℹ️ 车辆名映射: {veh_name} -> key={target_key}，匹配到 {fuzzy['Vehicle'].iloc[0]}")
    return fuzzy


def _prepare_soh_trajectory(soh_mapping_df, veh_name, soh_base_date):
    """
    统一整理某辆车 SOH 时间轨迹，供 merge_asof(direction='backward') 前向保持使用。
    优先读取真实时间戳；若只有 Days 列，则退化为基准日+天数。
    """
    veh_soh = _resolve_vehicle_rows(soh_mapping_df, veh_name)
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
        veh_soh['SOH_TIME'] = pd.to_datetime(soh_base_date) + pd.to_timedelta(veh_soh['Days'], unit='D')
    else:
        raise KeyError("SOH 映射表缺少可用于时间对齐的列（Charge_End_Time/DATA_TIME/Timestamp/Days）。")

    veh_soh = veh_soh.dropna(subset=['SOH_TIME', 'Pred_SOH']).copy()
    veh_soh = veh_soh.sort_values('SOH_TIME')
    return veh_soh[['SOH_TIME', 'Pred_SOH']]


def process_integrated(args):
    csv_files = sorted(glob.glob(os.path.join(args.data_folder, "*.csv")))
    print(f"📂 找到 {len(csv_files)} 个高频运行文件，启动【宏微观联合特征对齐】...")

    if len(csv_files) == 0:
        raise FileNotFoundError(f"数据目录下未找到 CSV: {args.data_folder}")

    if os.path.exists(args.output_csv):
        os.remove(args.output_csv)

    if not os.path.exists(args.soh_mapping):
        raise FileNotFoundError(f"找不到 SOH 映射文件: {args.soh_mapping}")
    soh_mapping = pd.read_csv(args.soh_mapping)

    if 'Vehicle' not in soh_mapping.columns:
        raise KeyError("SOH 映射文件缺少 Vehicle 列。")

    global_cycle_offset = 0
    allowed_keys = _load_allowed_vehicle_keys(args.vehicle_split_file, args.split_role)

    for i, file_path in enumerate(csv_files):
        file_name = os.path.basename(file_path)
        veh_name = file_name.split('.')[0]
        veh_key = _normalize_vehicle_key(veh_name)
        if allowed_keys is not None and veh_key not in allowed_keys:
            print(f"\n>>> [{i+1}/{len(csv_files)}] 跳过: {veh_name} (不在 {args.split_role} 列表)")
            continue
        print(f"\n>>> [{i+1}/{len(csv_files)}] 正在处理: {veh_name}")

        try:
            try:
                df = pd.read_csv(file_path, encoding='gbk', usecols=USE_COLS, low_memory=False)
            except Exception:
                df = pd.read_csv(file_path, encoding='utf-8', usecols=USE_COLS, low_memory=False)

            for col in ['totalVoltage', 'totalCurrent', 'maxTemperature', 'minTemperature', 'SOC']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df['DATA_TIME'] = pd.to_datetime(df['DATA_TIME'], errors='coerce')
            df = df.dropna(subset=['DATA_TIME', 'totalVoltage'])
            df = df[df['totalVoltage'] > args.min_valid_voltage].copy()
            df = df.sort_values('DATA_TIME').reset_index(drop=True)
            if len(df) == 0:
                continue

            df['time_diff'] = df['DATA_TIME'].diff().dt.total_seconds().fillna(0)
            internal_cycle_ids = (df['time_diff'] > args.gap_threshold_sec).astype(int).cumsum() + 1
            df['Cycle_ID'] = internal_cycle_ids + global_cycle_offset
            global_cycle_offset = df['Cycle_ID'].max()

            veh_soh_trajectory = _prepare_soh_trajectory(soh_mapping, veh_name, args.soh_base_date)
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

                if args.initial_soh_policy == 'fill_default':
                    df['SOH'] = df['SOH'].fillna(args.default_initial_soh)
                elif args.initial_soh_policy == 'drop_until_first_valid':
                    df = df[df['SOH'].notna()].copy()
                else:
                    raise ValueError(f"未知 initial_soh_policy: {args.initial_soh_policy}")

                if len(df) == 0:
                    print("    ⚠️ 当前文件所有数据都早于首个有效 SOH 更新点，已跳过。")
                    continue

                print(f"    ✅ 已按前向保持注入 SOH（backward asof），SOH 范围: {df['SOH'].min():.3f} ~ {df['SOH'].max():.3f}")
            else:
                df['SOH'] = args.default_initial_soh
                print(f"    ⚠️ 无该车 SOH 轨迹，已默认填充 SOH={args.default_initial_soh:.3f}")

            temp_mean = ((df['maxTemperature'] + df['minTemperature']) / 2).ffill()
            final_df = pd.DataFrame({
                'Vehicle': veh_key,
                'Current': df['totalCurrent'],
                'Voltage': df['totalVoltage'],
                'Temperature': temp_mean,
                'SOC': df['SOC'],
                'Cycle_ID': df['Cycle_ID'],
                'SOH': df['SOH'],
            })
            for c in ['Current', 'Voltage', 'Temperature', 'SOC', 'Cycle_ID', 'SOH']:
                final_df[c] = pd.to_numeric(final_df[c], errors='coerce').astype('float32')

            write_header = (i == 0)
            final_df.to_csv(args.output_csv, mode='a', index=False, header=write_header)

            del df
            del final_df
            gc.collect()
        except Exception as e:
            print(f"    ❌ 处理出错: {e}")

    print(f"\n🎉 宏微观数据合龙完成！联合数据集位于: {args.output_csv}")


def build_args():
    parser = argparse.ArgumentParser(description="构建 SOC 训练数据：将 SOH 前向保持注入动态工况数据")
    parser.add_argument('--data-folder', default=DATA_FOLDER_PATH, help='实车高频数据目录（按车 CSV）')
    parser.add_argument('--soh-mapping', default=SOH_MAPPING_FILE, help='SOH 估计结果 CSV 路径')
    parser.add_argument('--output-csv', default=PROCESSED_FILE_PATH, help='输出联合训练集 CSV 路径')
    parser.add_argument('--min-valid-voltage', type=float, default=MIN_VALID_VOLTAGE, help='最小有效总压阈值')
    parser.add_argument('--gap-threshold-sec', type=float, default=GAP_THRESHOLD_SEC, help='行程切分时间间隔阈值（秒）')
    parser.add_argument('--soh-base-date', default=SOH_BASE_DATE, help='当 SOH 文件仅有 Days 列时使用的基准日期')
    parser.add_argument(
        '--initial-soh-policy',
        default=INITIAL_SOH_POLICY,
        choices=['drop_until_first_valid', 'fill_default'],
        help='首个有效 SOH 更新前数据处理策略',
    )
    parser.add_argument('--default-initial-soh', type=float, default=DEFAULT_INITIAL_SOH, help='默认 SOH 值')
    parser.add_argument('--vehicle-split-file', default=VEHICLE_SPLIT_FILE, help='SOH阶段输出的车辆划分文件；留空表示不过滤车辆')
    parser.add_argument('--split-role', default='train', choices=['train', 'test', 'all'], help='根据划分文件选择处理哪些车辆')
    return parser.parse_args()


if __name__ == "__main__":
    process_integrated(build_args())

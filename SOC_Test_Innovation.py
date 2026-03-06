import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from SOC_AttentionGRU_Gated import SOH_Gated_GRU


def _normalize_vehicle_key(name):
    if not isinstance(name, str):
        return ""
    s = name.strip()
    m = re.search(r'(EV\d+)', s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return s.upper()


def _resolve_vehicle_rows(soh_mapping, veh_name):
    direct = soh_mapping[soh_mapping['Vehicle'] == veh_name].copy()
    if len(direct) > 0:
        return direct

    target_key = _normalize_vehicle_key(veh_name)
    tmp = soh_mapping.copy()
    tmp['_veh_key'] = tmp['Vehicle'].astype(str).map(_normalize_vehicle_key)
    fuzzy = tmp[tmp['_veh_key'] == target_key].copy()
    if len(fuzzy) > 0:
        print(f"ℹ️ 车辆名映射: {veh_name} -> key={target_key}，匹配到 {fuzzy['Vehicle'].iloc[0]}")
    return fuzzy


def _attach_soh(test_segment, soh_mapping, veh_name, soh_base_date, default_soh):
    veh_soh = _resolve_vehicle_rows(soh_mapping, veh_name)

    if len(veh_soh) == 0:
        test_segment['SOH'] = default_soh
        return test_segment

    if 'Charge_End_Time' in veh_soh.columns:
        veh_soh['SOH_TIME'] = pd.to_datetime(veh_soh['Charge_End_Time'], errors='coerce')
    elif 'DATA_TIME' in veh_soh.columns:
        veh_soh['SOH_TIME'] = pd.to_datetime(veh_soh['DATA_TIME'], errors='coerce')
    elif 'Timestamp' in veh_soh.columns:
        veh_soh['SOH_TIME'] = pd.to_datetime(veh_soh['Timestamp'], errors='coerce')
    elif 'Days' in veh_soh.columns:
        veh_soh['SOH_TIME'] = pd.to_datetime(soh_base_date) + pd.to_timedelta(veh_soh['Days'], unit='D')
    else:
        raise KeyError("SOH 映射文件缺少 Charge_End_Time/DATA_TIME/Timestamp/Days，无法做前向保持注入。")

    veh_soh = veh_soh.dropna(subset=['SOH_TIME', 'Pred_SOH']).sort_values('SOH_TIME')
    test_segment = test_segment.sort_values('DATA_TIME').reset_index(drop=True)
    test_segment = pd.merge_asof(
        test_segment,
        veh_soh[['SOH_TIME', 'Pred_SOH']],
        left_on='DATA_TIME',
        right_on='SOH_TIME',
        direction='backward',
        allow_exact_matches=True,
    )
    test_segment['SOH'] = test_segment['Pred_SOH'].ffill().fillna(default_soh)
    return test_segment


def run_integrated_test(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    veh_name = os.path.basename(args.test_file_path).split('.')[0]
    print(f"1. 正在加载测试车辆: {veh_name} ...")

    target_cols = ['totalVoltage', 'totalCurrent', 'maxTemperature', 'minTemperature', 'SOC', 'DATA_TIME']
    try:
        df = pd.read_csv(args.test_file_path, encoding='gbk', usecols=target_cols)
    except Exception:
        df = pd.read_csv(args.test_file_path, encoding='utf-8', usecols=target_cols)

    df['totalVoltage'] = pd.to_numeric(df['totalVoltage'], errors='coerce')
    df = df.dropna(subset=['totalVoltage'])
    df = df[df['totalVoltage'] > args.min_valid_voltage].reset_index(drop=True)
    df['DATA_TIME'] = pd.to_datetime(df['DATA_TIME'], errors='coerce')
    df = df.dropna(subset=['DATA_TIME']).reset_index(drop=True)

    test_segment = df.iloc[args.segment_start:args.segment_end].reset_index(drop=True)
    if len(test_segment) <= args.window_size:
        raise ValueError("测试片段长度不足，无法形成窗口。请调整 --segment-start/--segment-end")

    soh_mapping = pd.read_csv(args.soh_mapping_file)
    if 'Vehicle' not in soh_mapping.columns:
        raise KeyError('SOH 映射文件缺少 Vehicle 列。')
    if 'Pred_SOH' not in soh_mapping.columns:
        raise KeyError('SOH 映射文件缺少 Pred_SOH 列。')

    test_segment = _attach_soh(test_segment, soh_mapping, veh_name, args.soh_base_date, args.default_soh)
    actual_soh_value = test_segment['SOH'].mean()
    print(f"   -> 测试片段平均 SOH = {actual_soh_value*100:.2f}%")

    test_segment['Norm_Current'] = ((test_segment['totalCurrent'] - (-500)) / (500 - (-500))).clip(-1, 1)
    test_segment['Norm_Voltage'] = ((test_segment['totalVoltage'] - 200) / (800 - 200)).clip(0, 1)
    test_segment['maxTemperature'] = pd.to_numeric(test_segment['maxTemperature'], errors='coerce').fillna(25)
    test_segment['minTemperature'] = pd.to_numeric(test_segment['minTemperature'], errors='coerce').fillna(25)
    temp_mean = (test_segment['maxTemperature'] + test_segment['minTemperature']) / 2
    test_segment['Norm_Temp'] = ((temp_mean - (-30)) / (60 - (-30))).clip(0, 1)
    if test_segment['SOC'].max() > 1.5:
        test_segment['SOC'] = test_segment['SOC'] / 100.0

    inputs = test_segment[['Norm_Current', 'Norm_Voltage', 'Norm_Temp']].values
    soh_val = test_segment['SOH'].values
    true_soc = test_segment['SOC'].values

    model = SOH_Gated_GRU().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    pred_with_soh, pred_no_soh = [], []

    print("2. 启动宏微观协同验证推理...")
    with torch.no_grad():
        for i in range(len(test_segment) - args.window_size):
            x_tensor = torch.tensor(inputs[i:i + args.window_size], dtype=torch.float32).unsqueeze(0).to(device)
            s_tensor = torch.tensor([soh_val[i + args.window_size - 1]], dtype=torch.float32).unsqueeze(0).to(device)
            pred_with_soh.append(model(x_tensor, soh=s_tensor).item())
            pred_no_soh.append(model(x_tensor, soh=None).item())

    true_soc_real = true_soc[args.window_size:] * 100.0
    pred_with_soh_real = np.array(pred_with_soh) * 100.0
    pred_no_soh_real = np.array(pred_no_soh) * 100.0

    mae_with = np.mean(np.abs(pred_with_soh_real - true_soc_real))
    mae_no = np.mean(np.abs(pred_no_soh_real - true_soc_real))

    print("\n======== 联合估计成果验证 ========")
    print(f"Ours (联合 SOH): MAE = {mae_with:.2f}%")
    print(f"Baseline (无 SOH): MAE = {mae_no:.2f}%")
    if mae_no > 1e-12:
        print(f"老化门控机制带来的精度提升: {(mae_no - mae_with)/mae_no * 100:.2f}%")

    plt.figure(figsize=(12, 6))
    plt.plot(true_soc_real, 'k-', label='Ground Truth SOC (%)', linewidth=2.5)
    plt.plot(pred_with_soh_real, 'r--', label=f'Ours (MAE={mae_with:.2f}%)', linewidth=2)
    plt.plot(pred_no_soh_real, 'g:', label=f'Baseline (MAE={mae_no:.2f}%)', linewidth=1.5, alpha=0.8)

    plt.title(f"SOC Validation (Actual SOH ≈ {actual_soh_value*100:.1f}%)")
    plt.xlabel("Time Step")
    plt.ylabel("State of Charge (SOC) [%]")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.savefig(args.figure_out, dpi=300)
    print(f"✅ 图像已保存: {args.figure_out}")


def build_args():
    parser = argparse.ArgumentParser(description='SOC 创新对比测试（有/无 SOH 门控）')
    parser.add_argument('--test-file-path', required=True, help='单车原始高频 CSV 路径')
    parser.add_argument('--soh-mapping-file', default='outputs_final_best/SOH_Predictions_For_SOC.csv', help='SOH 估计结果 CSV')
    parser.add_argument('--model-path', default='Best_SOH_Model.pth', help='训练好的 SOC 模型路径')
    parser.add_argument('--window-size', type=int, default=30)
    parser.add_argument('--segment-start', type=int, default=-10000)
    parser.add_argument('--segment-end', type=int, default=-8000)
    parser.add_argument('--min-valid-voltage', type=float, default=100.0)
    parser.add_argument('--soh-base-date', default='2020-01-01')
    parser.add_argument('--default-soh', type=float, default=1.0)
    parser.add_argument('--figure-out', default='Macro_Micro_Coestimation_Result.png')
    return parser.parse_args()


if __name__ == "__main__":
    run_integrated_test(build_args())

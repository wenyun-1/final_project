import argparse
import glob
import os
import re
import subprocess

import pandas as pd


def run_cmd(cmd):
    print("\n🚀 执行:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _normalize_vehicle_key(name):
    if not isinstance(name, str):
        return ""
    s = name.strip()
    m = re.search(r'LFP\d+(EV\d+)', s, flags=re.IGNORECASE)
    if not m:
        hits = re.findall(r'(EV\d+)', s, flags=re.IGNORECASE)
        m = re.match(r'(EV\d+)', hits[-1], flags=re.IGNORECASE) if hits else None
    if m:
        return f"LFP604{m.group(1).upper()}"
    return s.upper()


def pick_test_file(data_folder, preferred, split_file):
    if preferred and os.path.exists(preferred):
        return preferred
    if split_file and os.path.exists(split_file):
        split_df = pd.read_csv(split_file)
        test_veh = split_df[split_df['Role'].astype(str).str.lower() == 'test']['Vehicle'].astype(str).tolist()
        if test_veh:
            keys = {_normalize_vehicle_key(v) for v in test_veh}
            for p in sorted(glob.glob(os.path.join(data_folder, "*.csv"))):
                stem = os.path.splitext(os.path.basename(p))[0]
                if _normalize_vehicle_key(stem) in keys:
                    return p
    candidates = sorted(glob.glob(os.path.join(data_folder, "*.csv")))
    if not candidates:
        raise FileNotFoundError(f"数据目录下未找到 CSV: {data_folder}")
    return candidates[-1]


def main(args):
    test_file = pick_test_file(args.data_folder, args.test_file, args.vehicle_split_file)
    print(f"📌 测试车辆文件: {test_file}")

    run_cmd([
        "python", "SOC_DataProcess_Real_Batch.py",
        "--data-folder", args.data_folder,
        "--soh-mapping", args.soh_mapping,
        "--output-csv", args.processed_csv,
        "--initial-soh-policy", args.initial_soh_policy,
        "--default-initial-soh", str(args.default_initial_soh),
        "--vehicle-split-file", args.vehicle_split_file,
        "--split-role", "train",
    ])

    run_cmd([
        "python", "SOC_Train_Gated.py",
        "--file-path", args.processed_csv,
        "--epochs", str(args.epochs),
        "--window-size", str(args.window_size),
        "--sample-stride", str(args.sample_stride),
        "--batch-size", str(args.batch_size),
        "--model-out", args.model_out,
    ])

    run_cmd([
        "python", "SOC_Test_Innovation.py",
        "--test-file-path", test_file,
        "--soh-mapping-file", args.soh_mapping,
        "--model-path", args.model_out,
        "--window-size", str(args.window_size),
        "--figure-out", args.figure_out,
    ])

    print("\n✅ 全流程完成！")
    print(f"- 训练数据: {args.processed_csv}")
    print(f"- 最佳模型: {args.model_out}")
    print(f"- 对比图像: {args.figure_out}")


def build_args():
    parser = argparse.ArgumentParser(description="一键运行 SOC 全流程（数据融合→训练→测试）")
    parser.add_argument('--data-folder', default='data', help='实车高频数据目录')
    parser.add_argument('--soh-mapping', default='outputs_final_best/SOH_Predictions_For_SOC.csv', help='SOH 映射结果文件')
    parser.add_argument('--processed-csv', default='Processed_All_Bus_Data.csv', help='生成的 SOC 训练数据路径')
    parser.add_argument('--model-out', default='Best_SOH_Model.pth', help='SOC 最佳模型输出')
    parser.add_argument('--figure-out', default='Macro_Micro_Coestimation_Result.png', help='SOC 对比图输出')
    parser.add_argument('--test-file', default='', help='指定测试车辆 CSV；留空则自动选 data/ 下最后一个')
    parser.add_argument('--vehicle-split-file', default='outputs_final/vehicle_split.csv', help='SOH阶段输出的车辆划分文件')
    parser.add_argument('--initial-soh-policy', default='drop_until_first_valid', choices=['drop_until_first_valid', 'fill_default'])
    parser.add_argument('--default-initial-soh', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--window-size', type=int, default=30)
    parser.add_argument('--sample-stride', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=512)
    return parser.parse_args()


if __name__ == '__main__':
    main(build_args())

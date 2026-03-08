import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from SOC_AttentionGRU_Gated import SOH_Gated_GRU
from SOC_RealVehicleDataset import RealVehicleDataset


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"正在使用设备: {device}")

    dataset = RealVehicleDataset(
        args.file_path,
        window_size=args.window_size,
        sample_stride=args.sample_stride,
        discharge_only=(not args.include_charge),
    )

    if len(dataset) < 10:
        raise ValueError("可用样本过少（可能过滤后为空），请检查数据或关闭放电过滤。")

    train_size = int(args.train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    model = SOH_Gated_GRU().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print("开始训练...")
    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for i, (x, soh, y) in enumerate(train_loader):
            x, soh, y = x.to(device), soh.to(device), y.to(device)

            # SOH Dropout: 训练阶段随机屏蔽 SOH，提高鲁棒性
            soh_input = None if np.random.rand() < args.soh_dropout else soh

            optimizer.zero_grad()
            pred = model(x, soh_input)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.6f}")

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_mae_loss = 0.0
        with torch.no_grad():
            for x, soh, y in test_loader:
                x, soh, y = x.to(device), soh.to(device), y.to(device)
                pred = model(x, soh)

                pred_real = pred * 100.0
                y_real = y * 100.0
                mae = torch.abs(pred_real - y_real).mean().item()
                val_mae_loss += mae

        avg_val_mae = val_mae_loss / len(test_loader)
        print(
            f"Epoch [{epoch+1}] 完成 | 耗时: {time.time()-start_time:.1f}s | "
            f"Train Loss(MSE): {avg_train_loss:.6f} | Val MAE (真实电量误差): {avg_val_mae:.2f}%"
        )

        if avg_val_mae < best_loss:
            best_loss = avg_val_mae
            torch.save(model.state_dict(), args.model_out)
            print(f">>> 模型已保存到 {args.model_out} (最佳真实误差: {best_loss:.2f}%)")


def build_args():
    parser = argparse.ArgumentParser(description='SOC 训练（SOH 门控 Attention-GRU）')
    parser.add_argument('--file-path', default='Processed_All_Bus_Data.csv', help='SOC 训练输入 CSV（由数据处理脚本生成）')
    parser.add_argument('--model-out', default='Best_SOH_Model.pth', help='最佳模型输出路径')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--window-size', type=int, default=30)
    parser.add_argument('--sample-stride', type=int, default=10)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--soh-dropout', type=float, default=0.3, help='训练时 SOH 随机屏蔽概率')
    parser.add_argument('--include-charge', action='store_true', help='若设置则不过滤充电段（默认仅放电段）')
    return parser.parse_args()


if __name__ == "__main__":
    train(build_args())

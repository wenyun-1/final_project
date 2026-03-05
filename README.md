# SOH Battery Dataset

完整数据集未上传到 GitHub（数据过大）。

请将数据放在：

- `data/`
- `data1/`

仓库中的 `samples/` 仅用于展示数据格式。

## 建议的实验流程（先体检，再训练）

1. **先做数据体检**（无依赖版本）
   ```bash
   python analyze_samples.py
   ```
   这个脚本会输出每个 CSV 的记录数、电流/电压范围、可用充电片段数量，帮助你快速判断数据质量。

2. **训练模型**（需要安装 pandas、numpy、torch、scipy）
   ```bash
   python soh_train.py
   ```
   当前训练脚本已经改为：
   - 自动读取 `data/`、`data1/`、`samples/`。
   - 先收集候选充电片段，再按每车电流分布自适应筛选，避免固定 110~130A 导致跨车型样本损失。

3. **离线评估与导出 SOH 特征**
   ```bash
   python soh_eval.py
   ```
   评估脚本会：
   - 输出每车 RMSE / R2；
   - 绘制并保存 SOH 图；
   - 导出 `SOH_Predictions_For_SOC.csv`，用于后续 SOC 模型输入。

## 经验建议

- 如果体检结果显示“覆盖 540-552V 的片段”太少，可尝试把 `V_START` / `V_END` 调整到你数据中更稳定的恒流区。
- 如果训练样本还是不足，可适当放宽 `MAX_CURRENT_FLUCTUATION`（如 15A -> 20A）并复测 RMSE。

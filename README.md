# SOH Battery Dataset

完整数据集未上传到 GitHub（数据过大）。

请将数据放在：

- `data/`
- `data1/`

仓库中的 `samples/` 仅用于展示数据格式。

## 论文复现：终极路线（推荐主线）

新增单一脚本：`soh_final_pipeline.py`，按你的终极方法一次跑通：

1. **伪标签生成**：前20%样本的85分位容量基准 + 一阶稳健拟合 + `SOH<=100%` 物理约束；
2. **PI-UAE 训练/评估**：交错抽样验证 + 严格训练集 scaler；
3. **双指标导出**：Raw 与 Filtered（时序平滑）同时给出；
4. **SOC接口导出**：`SOH_Predictions_For_SOC.csv`。

运行：

```bash
python soh_final_pipeline.py --epochs 120 --smooth-window 15 --output outputs_final
```

主要输出：

- `outputs_final/soh_metrics_vehicle.csv`（按车辆）
- `outputs_final/soh_metrics_summary.csv`（均值/标准差）
- `outputs_final/soh_predictions_points.csv`（每个测试点的 True/Raw/Filtered）
- `outputs_final/SOH_Predictions_For_SOC.csv`（供第四章SOC模型输入）
- `outputs_final/*_pseudo_labels.csv`（伪标签轨迹，可直接做论文图）

## 消融实验模板（含自动表格导出）

新增模板：`ablation_template.py`。

默认会跑 `smooth_window x seed` 的组合，并自动汇总：

```bash
python ablation_template.py --epochs 80 --output outputs_ablation
```

如果希望消融结束后自动出图：

```bash
python ablation_template.py --epochs 80 --output outputs_ablation --plot
```

输出：

- `outputs_ablation/ablation_results.csv`（完整汇总）
- `outputs_ablation/ablation_results_paper.csv`（论文友好精简表）

## 第三章结果图绘制（无需再次训练）

如果你已经有以下文件：
- `outputs_final/SOH_Predictions_For_SOC.csv`
- `outputs_final/soh_metrics_vehicle.csv`
- `outputs_final/*_pseudo_labels.csv`

可直接画第三章主图（子图排版，黑点真值 + 红线预测，标题含 `车辆名 + RMSE_filtered + R²_filtered`）：

```bash
python plot_experiment_results.py --soh-output outputs_final --ablation-output outputs_ablation --out-dir figures
```

主要输出：
- `figures/chapter3_soh_subplot.png`
- `figures/soh_metric_compare.png`
- `figures/ablation_compare.png`（若提供了消融结果）

> 其中 `chapter3_soh_subplot.png` 会优先使用 `soh_predictions_points.csv`；
> 若没有该文件，会自动回退到 `SOH_Predictions_For_SOC.csv + *_pseudo_labels.csv` 对齐绘图（无需重训）。

## 数据体检（轻量，不依赖 pandas/torch）

```bash
python analyze_samples.py
```

这个脚本可快速输出每个 CSV 的记录数、电流/电压范围、可用充电片段数，帮助你先判断数据质量。

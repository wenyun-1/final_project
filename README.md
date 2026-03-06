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


## Raw 与 Filtered 的含义（避免误读）

- **Raw**：模型对每个离散样本的直接输出（未做时间平滑）。
- **Filtered**：对 Raw 做滑动窗口平滑后的输出（更接近电池慢变老化，推荐作为第三章展示与第四章SOC输入）。

消融图中的 `swX_seedY` 含义：
- `swX` = 平滑窗口大小（X越大越平滑）
- `seedY` = 随机种子

消融实验**不是在比较不同SOH建模方法**，而是在固定终极路线下比较“平滑窗口+随机种子”的稳定性。
脚本会自动标注并输出推荐设置（按 `RMSE_filtered_mean` 最小）。


## 一键重训并输出Top6（按RMSE_filtered最小）

你可以直接运行：

```bash
python run_best_soh_experiment.py --search-root outputs_search --final-output outputs_final_best --figure figures/chapter3_top6_soh.png
```

该脚本会自动：
1. 用扩展网格做30轮快速筛选（`sw=[9,11,13,15,17]`, `seed=[42,3407,2025]`）
2. 以 `RMSE_filtered_mean` 最小选“搜索空间内最优”参数
3. 用最优参数进行150轮最终重训
4. 按 `RMSE_filtered` 最小选6辆车，输出3行2列子图：
   - 灰色散点：估计散点（raw，体现波动）
   - 红色曲线：SOH下降趋势（filtered）
   - 子图命名：`EV1`~`EV6`（按性能排名）

主要输出：
- `outputs_search/expanded_search_results.csv`
- `outputs_final_best/best_config.txt`
- `outputs_final_best/top6_vehicle_alias.csv`
- `figures/chapter3_top6_soh.png`

## 第四章 SOC 估计：从 SOH 结果到最终 SOC 输出（可直接运行）

你说得非常对，之前版本没有把 SOC 运行链路写清楚。下面是完整三步。

### 0) 先确认你有 SOH 结果文件
推荐使用：
- `outputs_final/SOH_Predictions_For_SOC.csv`

该文件至少要有：
- `Vehicle`（车辆名，如 `TEG6105BEV13_LFP604EV0sample`）
- `Pred_SOH`（SOH 值，0~1）
- 时间定位列之一：`Charge_End_Time` / `DATA_TIME` / `Timestamp` / `Days`

> SOC 数据处理脚本会自动识别上述时间列，并按“最近一次有效 SOH 前向保持（ZOH）”注入放电数据。

### 1) 生成 SOC 训练数据（动态工况 + SOH 因果注入）

```bash
python SOC_DataProcess_Real_Batch.py \
  --data-folder data \
  --soh-mapping outputs_final/SOH_Predictions_For_SOC.csv \
  --output-csv Processed_All_Bus_Data.csv \
  --initial-soh-policy drop_until_first_valid
```

说明：
- `drop_until_first_valid`：首个有效 SOH 出现前的数据会丢弃（更严谨，避免引入默认值偏差）。
- 如果你希望保留全部数据，可改为 `--initial-soh-policy fill_default --default-initial-soh 1.0`。

### 2) 训练 SOC 模型（Attention-GRU + SOH 门控）

```bash
python SOC_Train_Gated.py \
  --file-path Processed_All_Bus_Data.csv \
  --epochs 20 \
  --window-size 30 \
  --sample-stride 10 \
  --model-out Best_SOH_Model.pth
```

### 3) 进行 SOC 对比验证并输出图

```bash
python SOC_Test_Innovation.py \
  --test-file-path data/TEG6105BEV13_LFP604EV8sample.csv \
  --soh-mapping-file outputs_final/SOH_Predictions_For_SOC.csv \
  --model-path Best_SOH_Model.pth \
  --figure-out Macro_Micro_Coestimation_Result.png
```

输出：
- 终端打印 `Ours(有SOH)` vs `Baseline(无SOH)` 的 MAE
- 图像 `Macro_Micro_Coestimation_Result.png`

### 常见报错排查

- **找不到 SOH 文件**：检查 `--soh-mapping` 或 `--soh-mapping-file` 路径。
- **SOH 文件无时间列**：请确保至少有 `Charge_End_Time` / `DATA_TIME` / `Timestamp` / `Days` 之一。
- **测试片段长度不足**：在 `SOC_Test_Innovation.py` 中调整 `--segment-start` 和 `--segment-end`。

## 第四章 SOC 最终版（默认直接使用 outputs_final_best）

你当前的数据结构 `Vehicle,Days,Pred_SOH` 已兼容，且默认读取：
- `outputs_final_best/SOH_Predictions_For_SOC.csv`

同时已兼容车辆名不一致场景（例如 SOH 文件中是 `LFP604EV1`，原始实车文件名是 `TEG6105BEV13_LFP604EV1sample.csv`）。

### 一键执行（推荐）

```bash
python SOC_Run_End2End.py \
  --data-folder data \
  --soh-mapping outputs_final_best/SOH_Predictions_For_SOC.csv \
  --epochs 20
```

执行完成后可直接看到：
- `Processed_All_Bus_Data.csv`（SOC 训练集）
- `Best_SOH_Model.pth`（SOC 最优模型）
- `Macro_Micro_Coestimation_Result.png`（最终对比图，含有/无 SOH 的 SOC 估计效果）

### 分步执行（若你想单独控制每步）

```bash
python SOC_DataProcess_Real_Batch.py \
  --data-folder data \
  --soh-mapping outputs_final_best/SOH_Predictions_For_SOC.csv \
  --output-csv Processed_All_Bus_Data.csv \
  --initial-soh-policy drop_until_first_valid

python SOC_Train_Gated.py \
  --file-path Processed_All_Bus_Data.csv \
  --epochs 20 \
  --window-size 30 \
  --sample-stride 10 \
  --model-out Best_SOH_Model.pth

python SOC_Test_Innovation.py \
  --test-file-path data/TEG6105BEV13_LFP604EV8sample.csv \
  --soh-mapping-file outputs_final_best/SOH_Predictions_For_SOC.csv \
  --model-path Best_SOH_Model.pth \
  --figure-out Macro_Micro_Coestimation_Result.png
```

# SOH / SOC 实车联合估计（当前精简版说明）

> 本仓库不上传完整数据集。`samples/` 仅用于展示字段格式，不用于正式实验。

## 1. 数据目录约定（按当前思路）

请把**完整数据**放在：

- `data/`

典型命名（示例）：

- `data/LFP604EV1.csv`
- `data/LFP604EV12.csv`

当前 `soh_final_pipeline.py` 已做两件关键处理：

1. 默认只扫描 `data`（不再默认读 `samples`）；
2. 统一命名为 `LFP604EV*.csv` 后，车辆匹配和 SOH→SOC 对齐流程会更稳定。

---

## 2. 第三章 SOH 主流程（推荐）

```bash
python soh_final_pipeline.py \
  --data-dirs data \
  --read-chunk-size 200000 \
  --epochs 120 \
  --split-mode cross_vehicle \
  --train-vehicle-count 10 \
  --test-vehicle-count 2 \
  --log-every-epoch 10 \
  --smooth-window 15 \
  --output outputs_final
```

> 默认会把每辆车的“充电片段提取结果”缓存到 `outputs_final/segment_cache/`。
> 下次再次运行会优先读取缓存（日志显示 `[LoadCache]`），避免重复解析大 CSV。

若希望强制重建缓存：

```bash
python soh_final_pipeline.py --data-dirs data --refresh-segment-cache
```

若希望不使用缓存：

```bash
python soh_final_pipeline.py --data-dirs data --no-segment-cache
```

### 训练集不变时的快速运行（跳过训练）

新增脚本：`soh_run_reuse.py`

```bash
python soh_run_reuse.py --data-dirs data --output outputs_final
```

机制说明：
- 若检测到训练车辆集合和样本规模未变化，且 `global_pi_uae.pth` 已存在，则直接复用模型并跳过训练；
- 若训练集发生变化，则自动重新训练并更新模型。

### 说明

- `split-mode=cross_vehicle`：按车辆划分训练/测试，验证可迁移性；
- 默认采用 **严格** `10` 车训练 + `2` 车测试（若车辆不足 12 会直接报错，而不是自动降级）；
- 若内存紧张，可将 `--read-chunk-size` 调小（如 `50000` 或 `20000`）；
- 运行中会打印 `[Load]`、`[Split]`、`[Data]`、`[Train]` 进度信息，便于确认程序未卡住；
- 伪标签趋势拟合已加入数值稳定和退化回退机制，降低 `RankWarning` 风险。

### 主要输出

- `outputs_final/soh_metrics_vehicle.csv`
- `outputs_final/soh_metrics_summary.csv`
- `outputs_final/soh_predictions_points.csv`
- `outputs_final/SOH_Predictions_For_SOC.csv`
- `outputs_final/*_pseudo_labels.csv`
- `outputs_final/hi_features_all_samples.csv`（健康特征样本表）
- `outputs_final/hi_corr_pearson.csv`、`outputs_final/hi_corr_spearman.csv`
- `outputs_final/hi_corr_pearson_heatmap.png`、`outputs_final/hi_corr_spearman_heatmap.png`

---

## 3. 第四章 SOC 流程（使用 SOH 结果）

### 一键跑通

```bash
python SOC_Run_End2End.py \
  --data-folder data \
  --soh-mapping outputs_final/SOH_Predictions_For_SOC.csv \
  --vehicle-split-file outputs_final/vehicle_split.csv \
  --epochs 20
```

### 分步（便于调参）

```bash
python SOC_DataProcess_Real_Batch.py \
  --data-folder data \
  --soh-mapping outputs_final/SOH_Predictions_For_SOC.csv \
  --output-csv Processed_All_Bus_Data.csv \
  --vehicle-split-file outputs_final/vehicle_split.csv \
  --split-role train \
  --read-chunk-size 200000 \
  --initial-soh-policy drop_until_first_valid

python SOC_Train_Gated.py \
  --file-path Processed_All_Bus_Data.csv \
  --epochs 20 \
  --window-size 30 \
  --sample-stride 10 \
  --model-out Best_SOH_Model.pth

python SOC_Test_Innovation.py \
  --test-file-path data/LFP604EV8.csv \
  --soh-mapping-file outputs_final/SOH_Predictions_For_SOC.csv \
  --model-path Best_SOH_Model.pth \
  --figure-out Macro_Micro_Coestimation_Result.png
```

---

## 4. 当前实验口径（强烈建议）

- SOH：跨车辆训练/测试（不做同车随机切分作为主结果）；
- SOH：严格 10 车训练 + 2 车测试，并导出 `vehicle_split.csv`；
- SOC：默认读取同一 `vehicle_split.csv`，训练仅使用 train 车辆，测试车从 test 车辆中自动选择；
- SOC：默认仅放电段训练与评估（避免出现与任务定义不一致的 SOC 上升片段）；
- `samples/`：仅作为格式样例，不参与正式实验统计。

---

## 5. 常见报错排查（本地文件与仓库不一致）

如果出现类似报错：

```text
NameError: name 'shuffled' is not defined
```

且定位在 `class SOHDataset` 附近，通常是本地 `soh_final_pipeline.py` 发生了手工编辑/冲突残留，
导致 `split_vehicles()` 的代码块被错误粘贴到类定义内部。

建议直接用 Git 强制同步当前分支版本：

```bash
git fetch --all
git reset --hard HEAD
python -m py_compile soh_final_pipeline.py
```

若你本地在其他分支工作，请先切到目标分支再执行上述命令。

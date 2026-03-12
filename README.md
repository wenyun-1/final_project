# SOH / SOC 实车联合估计（当前紧急简化版）

> 本仓库不上传完整数据集。`samples/` 仅用于展示字段格式，不用于正式实验。

## 1. 当前 SOH 流程（已按要求简化）

`soh_final_pipeline.py` 现在只做一条主流程：

1. 仅扫描 `data/*.csv`；
2. 每次充电过程只截取 **SOC 75%→90%** 子区间；
3. 基于该子区间电流积分折算容量，计算每个充电片段的 SOH；
4. **每个充电片段的 SOH 都直接作为训练标签**（不再 7 天更新，不再做伪标签方法对比）；
5. 固定测试集为 `LFP604EV1` 与 `LFP604EV8`；
6. 每次运行都**强制重新提取数据片段**（不读旧缓存）。

---

## 2. 数据目录约定

请把完整数据放在：

- `data/`

典型命名示例：

- `data/LFP604EV1.csv`
- `data/LFP604EV8.csv`
- `data/LFP604EV12.csv`

---

## 3. 运行命令

```bash
python soh_final_pipeline.py
```

---

## 4. 主要输出

- `outputs_final/soh_metrics_vehicle.csv`
- `outputs_final/SOH_Predictions_For_SOC.csv`
- `outputs_final/soh_error.png`

---

## 5. 说明

- 当前版本已删除“按 7 天更新真值/注入真值”的流程代码；
- 当前版本已删除“多伪标签策略对比实验”的流程代码；
- 若后续要恢复对比实验，请在本版本基础上另开分支扩展。

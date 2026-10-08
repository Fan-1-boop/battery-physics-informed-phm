# Che et al. (Joule 2025) 复现阶段报告

## 结论状态

- 官方 `diagnostic_free_code.zip` SHA256: `f9f681324ed460d2bee206fb41b22826dd48d80ed4601e7321c0c02f678f48fe`，与 MATR 公开资产记录一致。
- 官方 `ResValData.zip` SHA256: `5898b06bf34550c2b57ea345c488d71fb24db4fbe6030225e260b062e22e840c`，与 MATR 公开资产记录一致。
- Dataset 2 原始文件：94 个 ResVal 文件；作者代码显式跳过 `ResVal_000084_0000ED.072`；按作者 `read_resval_data` 的 1C 20 min 输入筛选逻辑，另有 2 个文件不能形成合格输入，因此最终可用样本为 91。
- 已复现作者模型架构、DVA 物理约束、数据预处理、5-fold 划分逻辑，并成功用作者提供的 `best_model_fine_tune.pth` 在其对应的 C/5 最后一折测试集上进行真实数据 checkpoint 复验。
- 当前尚未完成 Dataset 1 / Dataset 3 / prognosis 的端到端重训练，因为对应原始数据尚未提供；Dataset 2 全 6 rate × 5 fold 的完整 20k-epoch 重训练也尚未完成。

## 官方模型结构审计

输入：200 × 3，分别为归一化电压、归一化容量、EFC（Dataset 2 中 EFC 固定为 1）。

Encoder MLP:

- 600 → 128 → 128 → 64
- 四个输出头：Cp, Cn, x0, y0
- ReLU 保证非负

Decoder:

- 4 → 128 → 256
- 两个 200 维输出：V(Q) 与 Q 序列

DVA constraint:

`SOC_p = y0 - Q/Cp`

`SOC_n = x0 - Q/Cn`

`V_derived = OCP_p(SOC_p) - OCP_n(SOC_n)`

Loss:

`L = L_reg + L_phy + 0.1 L_constraint`

其中 `L_phy` 是 predicted OCV 与 DVA-derived OCV 的 MSE；constraint 包含 Cp/Cn、SOCp/SOCn、预测电压、容量等边界。

Dataset 2 fine-tuning:

- 5-fold KFold, shuffle=True, random_state=0
- 每折训练集内部再随机 80/20 train/validation
- 由 Dataset 1 的 `best_model.pth` 初始化
- AdamW, lr=1e-4
- ReduceLROnPlateau factor=0.95, patience=100
- max_epoch=20000
- early-stopping patience=2000
- gradient clipping max_norm=1.0

## Dataset 2 真实数据 checkpoint 复验

目标：C/5 RPT；测试集为作者 KFold 的第 5 折，18 个电芯。这里使用作者随代码发布的 `best_model_fine_tune.pth`，因此是 checkpoint-level reproduction，而不是重新训练所得模型。

| 指标 | Dataset 1 pretrained（未微调） | Dataset 2 fine-tuned checkpoint |
|---|---:|---:|
| SOH MAE | 9.226 %pt | **1.764 %pt** |
| SOH RMSE | 11.420 %pt | **2.070 %pt** |
| SOH R² | -3.360 | **0.857** |
| Q MAE | 224.90 mAh | **43.63 mAh** |
| Q RMSE | 354.16 mAh | **60.53 mAh** |
| Q R² | 0.9295 | **0.99794** |
| predicted V MAE | 12.69 mV | **10.21 mV** |
| predicted V RMSE | 20.57 mV | **12.41 mV** |
| predicted V R² | 0.99757 | **0.99912** |
| derived V MAE | 25.87 mV | **18.90 mV** |
| derived V RMSE | 64.53 mV | **23.03 mV** |
| derived V R² | 0.97608 | **0.99695** |

论文 Table 1 对 Dataset 2 的六种 RPT rate 汇总值为：SOH MAE 1.86 %pt、Q MAE 43.1 mAh、predicted V MAE 22.4 mV、derived V MAE 43.1 mV。当前复验仅为 C/5 × 第 5 折，因此不能把 V 指标与六倍率聚合值作一一对应；但 SOH MAE 1.764 %pt 与论文聚合 1.86 %pt、Q MAE 43.63 mAh 与论文 43.1 mAh 高度一致，说明数据解析、网络结构、归一化和 checkpoint 加载链路基本闭环。

## 关键观察

1. Dataset 1 checkpoint 直接迁移到 Dataset 2 C/5 时，SOH MAE 高达 9.23 %pt；微调后降至 1.76 %pt。这说明论文中所谓“slight fine-tuning”并非可有可无，域适配对 SOH 标量非常关键。
2. 未微调模型的 voltage MAE 已只有 12.69 mV，但 SOH 很差，说明仅看曲线点级电压误差会高估模型的健康估计能力。
3. 微调后 derived OCV 的 MAE 从 25.87 mV 降到 18.90 mV，物理 latent states 与目标体系的一致性也得到改善。
4. 作者代码存在硬编码 CUDA（`.to('cuda')`）和 Windows 相对路径，原样在 CPU/Linux 环境不能直接运行；本复现实现了数值等价的 CPU 适配版本。
5. 作者代码保存的 `best_model_fine_tune.pth` 会在 rate × fold 循环中反复覆盖，因此发布包只保留最后一次训练的 checkpoint；无法仅凭发布 checkpoint 直接恢复 6 rate × 5 fold 的全部交叉验证预测，需要重新训练。

## 下一步（完整复现所需）

1. Dataset 1：MATR 11 的 processed StructuredData（作者 `func_read_json_files.py` 需要带 `raw_data` 的 `.json.gz` 文件）→ 复现 Figure 2/3、Table 1 Dataset 1、aging modes、prognosis。
2. Dataset 3：Geslin dynamic cycling 全量 CSV + 两个 half-cell BioLogic 文件 → 复现 dynamic charge/discharge 结果。
3. Dataset 2：跑完六种 target rate × 5 folds 的原始 20k/early-stopping 协议，生成 Table 1 聚合值和 Figure 5。
4. 完成 physics/no-physics、boundary/no-boundary 消融并固定 seeds。

## 已生成文件

- `results/dataset2_c5_finalfold_results.json`
- `results/dataset2_c5_finalfold_predictions.npz`
- `results/c5_finalfold_capacity_scatter.png`
- `results/c5_finalfold_ocv_example.png`
- `build_dataset2_cache.py`
- `make_finalfold_results.py`
- `cache_dataset2_meta.json`
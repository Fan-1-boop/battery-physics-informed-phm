# Che et al. (Joule 2025) 复现报告 v3

## 当前结论

Dataset 2 已完成作者协议级的 **6 个目标诊断倍率 × 5-fold = 30 次** fine-tuning / evaluation 复现。该阶段不再是单 checkpoint 验证，而是从 Dataset 1 官方 `best_model.pth` 初始化，按作者代码的 split、seed、optimizer、scheduler、loss、max epoch 和 early stopping 条件重新训练全部 30 个模型。

复现使用 91 个可被作者预处理逻辑实际形成 1C-20min 输入的电芯。原始 ResValData 含 94 个文件；作者代码显式排除 `ResVal_000084_0000ED.072`，另有 `ResVal_000025_000088.013` 和 `ResVal_000043_000091.031` 在 1C 输入筛选时无法形成有效样本。因此论文所述“94 cells”是原始测试集合规模，而当前发布代码实际进入该诊断模型交叉验证的是 91 个样本。

## 与论文 Table 1 的直接比较

| 指标 | 论文 Dataset 2 | 本次 30-run 复现 | 差值 |
|---|---:|---:|---:|
| SOH MAE [%pt] | 1.860 | **1.880** | +0.020 |
| SOH RMSE [%pt] | 2.540 | **2.577** | +0.037 |
| SOH R² | 0.890 | **0.886** | -0.004 |
| Capacity MAE [mAh] | 43.1 | **41.76** | -1.34 |
| Capacity RMSE [mAh] | 68.2 | **66.56** | -1.64 |
| Capacity R² | 0.997 | **0.99754** | +0.00054 |
| Predicted-V MAE [mV] | 22.4 | **21.96** | -0.44 |
| Predicted-V RMSE [mV] | 27.4 | **26.79** | -0.61 |
| Predicted-V R² | 0.996 | **0.99566** | -0.00034 |
| Derived-V MAE [mV] | 43.1 | **42.67** | -0.43 |
| Derived-V RMSE [mV] | 52.4 | **52.01** | -0.39 |
| Derived-V R² | 0.984 | **0.98364** | -0.00036 |

所有核心误差指标与论文差异约在 0–3% 相对范围内；SOH MAE 只相差 0.020 个百分点。Dataset 2 因此可判定为 **PASS_REPRODUCED**。

## 分倍率结果

| 目标倍率 | SOH MAE [%pt] | Q MAE [mAh] | Pred-V MAE [mV] | Derived-V MAE [mV] |
|---|---:|---:|---:|---:|
| C/80 | 2.266 | 45.41 | 30.62 | 59.65 |
| C/40 | 1.626 | 34.77 | 27.26 | 53.23 |
| 0.05 A | 2.580 | 48.47 | 31.64 | 61.27 |
| C/10 | 1.627 | 38.45 | 16.62 | 32.45 |
| C/7 | 1.581 | 40.16 | 13.42 | 26.20 |
| C/5 | 1.598 | 43.32 | 12.19 | 23.24 |

低倍率 C/80 与 0.05 A 的曲线重构明显更难；随着目标 RPT 倍率提高，电压重构误差快速下降。但 SOH MAE 并不严格随倍率单调下降，说明标量健康估计与点级 OCV 重构难度并非完全等价。

## 训练协议

- KFold: 5 folds, `shuffle=True, random_state=0`
- 每折训练集：`torch.manual_seed(123)` 后 shuffle，80/20 train/validation
- 初始化：Dataset 1 官方 `best_model.pth`
- Optimizer: AdamW, lr = 1e-4
- Scheduler: ReduceLROnPlateau, factor = 0.95, patience = 100
- Loss: regression + physics + 0.1 × constraints
- gradient clipping: 1.0
- max epoch: 20,000
- early stop: validation 2,000 epochs no improvement，或 lr < 5e-7

30 个 run 均按上述停止规则完成。不同 fold 的最佳 epoch 跨度很大，约从 1.8k 到 18k，说明若仅统一截断到 2k/4k epoch，会明显改变聚合结果。

## 复现价值

这次完整 30-run 的结果把上一版“某一个官方 checkpoint 能对上”提升为“**从 source checkpoint 重新 fine-tune 后，论文 Dataset 2 的 aggregate Table 1 指标整体可重现**”。这显著降低了后续研究中对数据解析、DVA 实现、归一化、交叉验证和训练流程的复现不确定性。

同时保留此前关键观察：Dataset 1 模型直接 zero-shot 用于 Dataset 2 时，C/5 最后一折 SOH MAE 约 9.23 %pt，而 fine-tuning 后约 1.76 %pt。因此该工作在 Dataset 2 上本质上仍明显依赖目标域监督 fine-tuning；这为“少样本迁移 / 无监督域适配 / physics-guided domain adaptation”留下了明确空间。

## 下一步

1. Dataset 2：追加 formal ablation（physics loss / boundary constraints / target-label fraction），用于确定后续创新基线。
2. Dataset 1：需要作者 Aging Matrix 的 StructuredData / 对应 JSON.GZ，才能完整复现 Figure 2–4、aging-mode 分解与 prognosis。
3. Dataset 3：需要 Geslin dynamic cycling 原始数据和作者代码引用的 half-cell 文件，才能复现动态 charge/discharge 结果。

当前 Dataset 2 状态：**PASS_REPRODUCED**。
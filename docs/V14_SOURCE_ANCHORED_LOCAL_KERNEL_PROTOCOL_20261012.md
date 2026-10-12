# v14 源模型保持的统一局部核残差校准：执行前开发集协议（2026-10-12）

## 状态、假设和方法范围

本轮协议先于v14首次真实数据执行而冻结。**既有v8-v13已反复研究同一47目标电芯：v14是开发集机制检验，绝非独立确认或原始首次预注册。** 不依据v14测试误差修改算法或择优隐瞒失败。保留全部负结果。

**用户确定的方法框架**：不是“预测是否校准”的二元选择器，而是统一容量估计函数 `Qhat=f_RW(X)+g(X)`，其中`g(X)`是训练标签引导的连续局部修正；距离训练支持域很远时可逼近0，统一包含零/弱/强修正，不按工况制定特例。

## 文献边界

来源与前人算法：RBF kernel ridge regression / zero-mean residual GP、transfer residual correction、源模型保持属于现成原理；有关电池迁移可参考：

- 2024 IEEE TTE *State-of-Health Estimation of Li-Ion Batteries Using Semiparametric Adaptive Transfer Learning*, DOI 10.1109/TTE.2023.3266499，AT-GPR采用自适应迁移核与目标任务相似性；
- 2025 Applied Energy *Transfer learning with composite kernel sparse Gaussian process-aided model for probabilistic state of health estimation ...*, DOI 10.1016/j.apenergy.2025.126762，含稀疏高斯过程辅助迁移及概率补偿；
- 2025 Energy *A cross-material ... three-stage domain adaptation*, DOI 10.1016/j.energy.2025.139376，使用MMD、混合域适配及少量目标微调。

v14不是以上作者完整网络或数据设定的等价复现，更不是原创核岭回归。**我们的具体研究改动**是RW8源模型保持、六维统计特征加容量输出导向的核距离、两电芯等权标签约束与固定早期预算的跨工况实际验证。

## Frozen source、target、acquisition

- 与v8–v13**完全相同**XJTU 8 RW源，固定PinN4SOH upstream commit `cc3cc5053caf38f04e0665f7f88cb109144d035e`，同RW源尺度和Ridge(alpha=1)系数，严格核对55个原始CSV Git blob SHA；
- 原47目标5工况：2C8,3C15,R2.5-8,R3-8,Sim_satellite-8，完整充电统计X六特征、放电容量Q目标单位Ah输入/mAh输出，原v8按电芯五折；
- 每折目标训练池2个电芯、前N=10/20/30个早期容量标签，源Ridge、v10残差Ridge(alpha10)、v10偏置offset不变；
- 选择规则原v10 random(5次)与pred_quartiles(1次)；绝不根据任何target test真值选样、缩放、调参；随机5重复不当作独立新电芯；
- 本轮主要比较N30 pred_quartiles，N10/N20和random均必须报告。每目标电芯等权、五工况等权 MAE (mAh)为主要效果。

## 提前唯一固定的候选机制与简明消融

1. 用原 v8 RW源 `design` 得到裁剪后的6维标准化充电统计向量 `z(x)`；RW标准化参数**不允许用目标电芯更新**。
2. 源模型预测 `q_RW(x)=f_RW(x)` (Ah)，源权重中6维系数 `b` 的L2范数 `s=max(||b||_2,1e-8)` 是固定量。
3. 所有方法统一使用RBF高斯核，欧氏距离先除6：
   - 标准特征核 `D_X² = ||z-z'||²/6`；`K_X=exp(-0.5 D_X²)`。
   - **主方法：输出引导核** `D_XQ² = (||z-z'||² + ((q_RW(x)-q_RW(x'))/s)²)/7`；`K_XQ=exp(-0.5 D_XQ²)`。等价于在6D标准化特征上加入沿源Ridge预测方向的第7个任务导向维度。这是线性相关的再加权度量，绝不宣称独立信息增加。
4. 每个候选选择固定标签库`L`由两电芯各N循环组成。残差 `r_i=y_i-f_RW(x_i)`，每观测的权重 `w_i=1/(2N)`（各电芯总权重1/2）。取固定正则强度`rho=0.05` Ah尺度对应残差RKHS二范数惩罚 `rho||g||_H²`。核岭回归的解：
   `alpha=(K_L+rho*diag(1/w_i))^{-1}r`
   `g(x)=K(x,L)alpha`。
   不含独立常数残差项，因此当核与全部标注点趋于0时残差自动趋于0、即零校准属于同一函数极限。
5. **本轮只计算两种提前固定的核空间，不做任何网格/贝叶斯/外层或测试集超参搜索**：`RBF-X`（朴素局部残差核，必要消融）和`RBF-XQ`（用户希望的任务导向统一校准主方法）。不使用v13事后Oracle作为训练数据、决策器或者基线。
6. 预测时不访问待测电芯容量；正常预测可用该循环自身的充电X（无需未来循环容量），特征距离在RW冻结尺度完成。记录最小核支持、最大绝对修正值及更远位置`K(x,L)->0`的数学性质，但不承诺已数据域必出现零修正。

## 预定比较与成功标准

每N×policy×target报`Source-only`, `Ridge`, `offset`, `RBF-X`, `RBF-XQ`，N30 pred_quartiles另报各目标负迁移数量、3C逐电芯负迁移与相对源模型增益、所有输入成本和对原v10/v12历史结果逐个匹配。

研究成功应至少在五域平均MAE及3C负迁移之间有合理、稳定的权衡，最好在random/quartiles及N10/20/30上不明显恶化；若不达标则**如实归档失败**，不能将“远点核趋零”的数学性质误当成实测已减少负迁移。

## 数据审计和研究边界

- v12原始逐测试电芯的source、ridge、offset、选中电芯ID、label数量、每fold heldout顺序为冻结参考，逐MAE重算误差必须<1e-6 mAh；
- 零测试标签泄漏；仅训练pool内2个电芯的真实容量允许拟合`g`；其余训练pool容量在加载的原始CSV中存在但实验程序不应索引；
- 所有原实验及结果文件不覆盖；v14新文件另存`results/v14`；
- N10通过目标训练pool前12无标签循环做选样，技术上需事先储存前10容量记录，否则到cycle12才标注不能倒测前10循环；不得声称真实在线的10循环前瞻选样无需回溯；
- 单工况目标没有独立未使用电芯；所有优势只能作**探索性开发集实证**，后续仍应寻未见的新数据集。

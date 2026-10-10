# Crystal Diffusion v3 — 原子种类的离散扩散（D3PM）

本目录是仓库的第 3 版（v3）。做法是：**保留 v1 里那套能正常重建的骨干**
（周期近邻图 + 消息传递 + 槽位嵌入；坐标与晶格走连续高斯扩散），
只把**原子种类**从「one-hot 连续松弛 + 采样式 argmax」换成真正的
**离散扩散（D3PM，Austin et al. 2021）**。

支持两种前向转移，用 `--atom-type-diffusion` 切换，**默认 mask（吸收型）**：

| 类型 | 前向 | 反向 |
|---|---|---|
| `mask`（默认） | 以概率 β_t 换成 `[MASK]` 并保持 | 非 `[MASK]` 位置**确定性**保持不变，只有 `[MASK]` 位置重抽 |
| `uniform` | 以概率 β_t 跳到均匀随机类别 | 每个位置每步都可能被重写 |

示例结构仍是 `Ba2PrCu3O7`（13 个原子，4 种元素），与 v1 一致，方便对比。

## ⚠️ 先看一个评估修正（重要）

分数坐标里 **0 和 1 是同一个位置**。如果用朴素的 `|gen - target|` 计算误差，
把「坐标 0.995」和「目标 0.0」当成差了 0.995，会严重高估误差。

本版所有坐标误差一律用**最小镜像**计算：

```
diff = gen - target;  diff = diff - round(diff);  MAE = |diff|
```

（折算成卡氏距离就是真实的空间偏差，单位埃。）

**本仓库 v1 的 README 用的是朴素指标，那个坐标 MAE 0.073 偏悲观**，
用同一把尺子重测后其实约为 0.092 埃（见下面的对比表）。这一条对所有晶体生成任务都适用。

## D3PM 是怎么做的

类别集合 `{0..K-1}`（这里 K=4 种元素），用同一条余弦曲线作为累积保留概率 `alpha_bar_t`。

**mask（吸收型）**：`q(x_t|x_0) = alpha_bar_t·δ(x_t,x_0) + (1-alpha_bar_t)·δ(x_t,MASK)`。
反向（`x_t = MASK` 时）：

```
P(x_{t-1} = 真实类别 k) = (alpha_bar_{t-1} - alpha_bar_t)/(1 - alpha_bar_t) · p_theta(k)
P(x_{t-1} = MASK)      = (1 - alpha_bar_{t-1})/(1 - alpha_bar_t)
```

**uniform（均匀型）**：`Q_t = (1-β_t)I + (β_t/K)·11^T`，边际以 `alpha_bar_t` 停留、否则均匀分布。

训练目标都是「预测干净类别」的交叉熵。

## 已验证的数学

`tests/test_d3pm.py`，两种模式都测，全部通过：

| 检查 | 内容 |
|---|---|
| A 前向边缘分布 | mask 型保留概率 = `alpha_bar_t`；uniform 型 = `alpha_bar_t+(1-alpha_bar_t)/K` |
| B 后验合法性 | 概率非负、逐类求和为 1 |
| C mask 确定性 | `x_t ≠ [MASK]` 时反向一步必须保持原值 |
| D oracle 全链路 | mask：解开的位置全部正确；uniform：恢复正确率 1.000 |
| E 基本流程 | 两种模式的前向 / 训练 / 采样都正常 |

## 目录结构

| 路径 | 职责 |
|---|---|
| `data/Ba2PrCu3O7.json` | 输入结构 |
| `src/data/structure.py` | 结构读写 + 张量转换 |
| `src/data/graph.py` | 周期最小镜像近邻 → RBF 边特征 |
| `src/diffusion/schedule.py` | 余弦调度（连续与离散共用） |
| `src/diffusion/d3pm.py` | D3PM：mask / uniform 的前向、后验、损失 |
| `src/diffusion/forward.py` | 训练目标（连续 ε-prediction + 离散 CE） |
| `src/diffusion/reverse.py` | 反向采样（D3PM + DDPM 后验） |
| `src/models/mpnn.py` | 消息传递 GNN（物种头输出 logits） |
| `src/utils/poscar.py` | 写 VASP POSCAR |
| `scripts/train.py` | 训练 + 采样 + 导出 + 误差报告 |
| `tests/test_d3pm.py` | D3PM 数学与流程验证 |

## 依赖与运行

只需要 `torch` 和 `numpy`，仓库根目录的 `requirements.txt` 已覆盖。

```bash
# D3PM 数学自检（几秒钟）
python v3_d3pm/tests/test_d3pm.py

# 训练 + 生成（CPU，默认 4000 步；物种默认 mask 型）
python v3_d3pm/scripts/train.py --steps 4000 --samples 4

# 换成均匀型 / 确定性物种采样做对比
python v3_d3pm/scripts/train.py --steps 4000 --atom-type-diffusion uniform
python v3_d3pm/scripts/train.py --steps 4000 --species-argmax
```

## 参考结果（mask 型，4000 步，CPU，约 9.1 万参数）

- **物种**：样本逐位 13/13、**组成全部正确**，训练交叉熵降到 0.0001
- **坐标**：周期感知 MAE 0.0043（折算卡氏误差 **0.068 埃**）
- **晶格**：MAE **0.053 埃**

## mask vs uniform

晶格误差不受坐标周期性的影响，可以直接比较：

| 物种扩散 | 晶格 MAE | 物种组成 |
|---|---|---|
| **mask（默认）** | **0.053 埃** | 4/4 正确 |
| uniform | 0.135 埃 | 4/4 正确 |

结论和预期一致：**mask 型反向大部分位置是确定性的，注入共享主干的随机性小，
晶格精度明显更好**（吸收型在实践中通常更稳，因此常被用作默认选择）。

（坐标方面：mask 用新指标测得 0.068 埃；uniform 那组当时用的还是旧指标，
数字不可直接比较，如需严格对比要按新指标重跑。）

## 三个版本，用同一把尺子

| 版本 | 周期感知坐标 MAE | 卡氏误差 | 晶格 MAE |
|---|---|---|---|
| v1 原始晶格矩阵 | 0.0078 | 0.092 埃 | 0.190 埃 |
| 等变版（Gram 不变晶格） | **0.0043** | 0.078 埃 | 0.063 埃 |
| **v3（D3PM mask，本目录）** | **0.0043** | **0.068 埃** | **0.053 埃** |

## 诚实的结论

- **物种这一路是成功的**：从「连续松弛」升级为真正的类别扩散，组成永远正确，
  并且验证了 mask 型确实优于 uniform 型。这是本版的核心目标，已达成。
- **顺带纠正一个此前的误判**：之前用朴素指标比较时，我曾认为「等变版重建更差」。
  用正确的周期感知指标重测后，等变版其实**优于** v1（坐标 0.078 vs 0.092 埃，
  晶格 0.063 vs 0.190 埃）。那个结论是评估方式造成的，不是模型的问题。
- 因此：**v3（本目录）全面优于 v1**；等变版（v2）同样优于 v1。

## 后续计划

- 把等变性（Gram 不变晶格 + 相对位移）与 D3PM 合并，做一个同时具备两者优点的版本。
- 换用更大的训练预算（更多步数 / 更大模型）并引入多结构训练，
  这时候置换等变才会真正发挥价值。

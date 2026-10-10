# Crystal Diffusion v3 — 原子种类的离散扩散（D3PM）

本仓库的第 3 版。做法是：**保留能正常重建的连续骨干**（周期近邻图 + 消息传递 + 槽位嵌入；
坐标与晶格走连续高斯扩散），只把**原子种类**从「one-hot 连续松弛 + argmax」
换成真正的**离散扩散（D3PM，Austin et al. 2021）**。

示例结构仍是 `Ba2PrCu3O7`（13 个原子，4 种元素），与 v1/v2 一致，便于对比。

## 本版改了什么（相对 v1 / v2）

| | v1 / v2 | v3 |
|---|---|---|
| 原子种类 | one-hot 当连续量加高斯噪声，采样时 argmax | **类别变量 + D3PM 离散扩散** |
| 物种输出头 | 预测噪声（回归） | 预测**干净类别的 logits**（分类） |
| 物种损失 | MSE | **交叉熵** |
| 物种采样 | 高斯反向 + argmax | **D3PM 后验的祖先采样** |
| 坐标 / 晶格 | 连续扩散 | 不变 |

## 两种前向转移（用 `--atom-type-diffusion` 切换）

| 类型 | 前向 | 反向 |
|---|---|---|
| **`mask`（默认，吸收型）** | 以概率 β_t 换成 `[MASK]` 并保持 | 非 `[MASK]` 位置**确定性**保持不变，只有 `[MASK]` 位置按后验重抽 |
| `uniform`（均匀型） | 以概率 β_t 跳到均匀随机类别 | 每个位置、每一步都可能被重写 |

累积保留概率 `alpha_bar_t` 复用同一条余弦曲线。

**mask 型的后验**（`x_t = [MASK]` 时）：

```
P(x_{t-1} = 真实类别 k) = (alpha_bar_{t-1} - alpha_bar_t)/(1 - alpha_bar_t) · p_theta(k)
P(x_{t-1} = [MASK])    = (1 - alpha_bar_{t-1})/(1 - alpha_bar_t)
```

**uniform 型**：`Q_t = (1-β_t)·I + (β_t/K)·11^T`；边际以 `alpha_bar_t` 停留、否则均匀分布。

## 已验证的数学

`tests/test_d3pm.py`，两种模式都测，五项全部通过：

| 检查 | 内容 |
|---|---|
| A 前向边缘分布 | mask 型保留概率 = `alpha_bar_t`；uniform 型 = `alpha_bar_t+(1-alpha_bar_t)/K` |
| B 后验合法性 | 概率非负、逐类求和为 1 |
| C mask 确定性 | `x_t ≠ [MASK]` 时反向一步必须保持原值 |
| D oracle 全链路 | mask：解开的位置全部正确；uniform：恢复正确率 1.000 |
| E 基本流程 | 两种模式的前向 / 一步训练 / 采样都正常 |

## 目录结构

| 路径 | 职责 |
|---|---|
| `data/Ba2PrCu3O7.json` | 输入结构 |
| `src/data/{structure,graph}.py` | 结构转换 / 周期最小镜像边特征 |
| `src/diffusion/schedule.py` | 余弦调度（连续与离散共用） |
| `src/diffusion/d3pm.py` | D3PM：mask / uniform 的前向、后验、损失 |
| `src/diffusion/forward.py` | 训练目标（连续 ε-prediction + 离散交叉熵） |
| `src/diffusion/reverse.py` | 反向采样（D3PM + DDPM 后验） |
| `src/models/mpnn.py` | 消息传递 GNN（物种头输出 logits） |
| `src/utils/poscar.py` | 写 VASP POSCAR |
| `scripts/train.py` | 训练 + 采样 + 导出 + 误差报告 |
| `tests/test_d3pm.py` | D3PM 数学与流程验证 |

## 依赖与运行

只需要 `torch` 和 `numpy`。

```bash
python v3_d3pm/tests/test_d3pm.py
python v3_d3pm/scripts/train.py --steps 4000 --samples 4           # 默认 mask 型
python v3_d3pm/scripts/train.py --steps 4000 --atom-type-diffusion uniform
python v3_d3pm/scripts/train.py --steps 4000 --species-argmax      # 物种用确定性 argmax
```

## 参考结果

统一协议：8 个样本、固定随机种子、坐标误差用**最小镜像**、CPU 训练 4000 步。

- **物种**：逐位 13/13，**组成正确 8/8**，训练交叉熵降到 0.0001
- **坐标**：卡氏误差 **0.088 埃**
- **晶格**：**0.048 埃**（三个版本里最好）

## mask vs uniform

晶格误差不受坐标周期性的影响，可以直接比较：

| 物种扩散 | 晶格 MAE | 物种组成 |
|---|---|---|
| **mask（默认，8 样本）** | **0.048 埃** | 8/8 正确 |
| uniform（4 样本） | 0.135 埃 | 4/4 正确 |

结论与预期一致：**mask 型反向时大部分位置是确定性的，注入共享主干的随机性小**，
因此晶格精度明显更好（样本数不同，但差距超过一倍）。

## 三个版本对比（同一把尺子）

| 版本 | 核心改动 | 坐标（卡氏） | 晶格 | 物种组成 |
|---|---|---|---|---|
| v1 | 基线：周期图 + 消息传递 GNN | 0.112 埃 | 0.197 埃 | 8/8 正确 |
| v2 | 旋转不变 Gram 晶格 + 相对位移 | **0.075 埃** | 0.065 埃 | 8/8 正确 |
| **v3（本目录）** | 原子种类改用 D3PM（mask） | 0.088 埃 | **0.048 埃** | 8/8 正确 |

三者物种都正确；v2 的坐标最准，v3 的晶格最准，都明显优于 v1。

## 局限

- 物种的离散扩散只作用于类别；坐标与晶格仍是各向同性的高斯扩散。
- 仍是单结构过拟合，不是学分布；没有条件生成。
- 训练预算很小（4000 步、约 9 万参数、T=200），换更大预算应能进一步提升。

## 后续计划

- 把 v2 的等变性（Gram 不变晶格 + 相对位移）与 v3 的 D3PM 合并。
- 多结构训练：到那时置换等变才会真正发挥价值。

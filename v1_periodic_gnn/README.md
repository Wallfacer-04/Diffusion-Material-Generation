# Crystal Diffusion v1 — 周期图 + 消息传递 GNN

一个从零实现的晶体结构扩散生成模型：在**单个晶体结构**上训练，再从纯噪声生成它自己。
本目录是仓库的第 1 版（v1），后续改进会以 `v2_*`、`v3_*` 的形式并列添加，互不影响。

示例结构：`Ba2PrCu3O7`（13 个原子，正交晶胞）。

## 目标

给定一个晶体结构，训练一个扩散模型，使它能够从纯噪声出发、生成回该结构。
这是「无中生有」生成一个具体晶体的最小完整流程。

## 结构表示

| 部分 | 表示 | 形状 |
|---|---|---|
| 原子种类 | 固定元素表上的 one-hot 类别向量 | (N, K) |
| 坐标 | 分数坐标（周期性） | (N, 3) |
| 晶格 | 3×3 矩阵，每行一个晶格向量（Å） | (3, 3) |

晶格整体除以 `LATTICE_SCALE=10` 压到 O(1)，让三部分可以用同一套噪声调度。

## 模型与扩散

- **骨干**：消息传递 GNN
  - 节点特征 = one-hot 物种 + 坐标 + 晶格（全局广播）+ 时间嵌入 + 槽位嵌入
  - 边特征 = 最小镜像距离的高斯 RBF（周期性）
  - 消息 `m_ij = MLP([h_i, h_j, e_ij])`，用 `sum` 聚合，再更新 `h_i`
- **噪声调度**：余弦调度（默认 T=200）
- **训练目标**：epsilon-prediction，物种 / 坐标 / 晶格三部分联合 MSE
- **采样**：DDPM 反向，用解析后验均值与方差

## 目录结构

| 路径 | 职责 |
|---|---|
| `data/Ba2PrCu3O7.json` | 输入结构（晶格 + 元素 + 分数坐标） |
| `src/data/structure.py` | 结构读写 + 张量转换 |
| `src/data/graph.py` | 最小镜像近邻 → 高斯 RBF 边特征 |
| `src/diffusion/schedule.py` | 余弦噪声调度 |
| `src/diffusion/forward.py` | 前向加噪 + 训练损失 |
| `src/diffusion/reverse.py` | DDPM 反向采样 |
| `src/models/mpnn.py` | 消息传递 GNN 去噪器 |
| `src/utils/poscar.py` | 写 VASP POSCAR（纯文本） |
| `scripts/train.py` | 训练 + 采样 + 导出 + 误差报告 |
| `tests/test_smoke.py` | 快速自检 |

## 依赖

只需要 `torch` 和 `numpy`（其余为标准库），仓库根目录的 `requirements.txt` 已覆盖。

## 运行

```bash
# 自检（几秒钟）
python v1_periodic_gnn/tests/test_smoke.py

# 训练 + 生成（CPU，默认 4000 步，约几分钟）
python v1_periodic_gnn/scripts/train.py --steps 4000 --samples 4
```

可调参数：`--steps`（训练步数）、`--num-steps`（扩散步数 T）、`--hidden`、`--layers`、`--samples`、`--seed`。

## 产出

- `outputs/crystal_gnn.pt`：模型权重
- `outputs/sample_*.vasp`：生成的 POSCAR（可用 VESTA 打开）
- 控制台打印每个样本的「物种匹配数 / 坐标 MAE / 晶格 MAE」

## 参考结果

CPU 上训练 4000 步（模型约 9.1 万参数）：

- 4 个样本的物种全部 13/13 匹配
- 坐标 MAE 0.009–0.115
- 晶格 MAE 0.12–0.27 Å

坐标基本复原；晶格是相对最难的部分。

## 适用范围与局限

- **物种**用 one-hot 连续松弛（加高斯噪声、采样时取 argmax），不是离散扩散。
- **对称性**：边特征是平移/旋转不变的，但节点特征吃的是裸坐标与裸晶格，且每个槽位有独立嵌入，
  因此模型不是严格置换不变、也没有 E(3) 等变。
- **单结构**：只过拟合一个结构，不是学分布；换结构需要重新训练。
- 没有条件生成。

## 后续版本计划

- **v2**：换等变骨干（相对位移更新 + 不变晶格表示），去掉槽位嵌入。
- **v3**：物种改用离散扩散（D3PM）。
- **v4**：条件生成与多结构训练。

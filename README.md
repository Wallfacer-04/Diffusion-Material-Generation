# Diffusion-Material-Generation

用扩散模型做**晶体结构生成**的实验仓库。三个版本都用同一个示例结构
`Ba2PrCu3O7`（13 个原子、4 种元素），都能在 **CPU** 上从头训练（约几分钟 / 4000 步、约 9 万参数）。

| 版本 | 目录 | 核心改动 | 坐标误差 | 晶格误差 | 物种组成 |
|---|---|---|---|---|---|
| v1 | [`v1_periodic_gnn`](v1_periodic_gnn) | 基线：周期近邻图 + 消息传递 GNN + 连续扩散 | 0.112 埃 | 0.197 埃 | 正确 |
| v2 | [`v2_equivariant_gnn`](v2_equivariant_gnn) | 旋转不变 **Gram 晶格表示** + 相对位移消息 | **0.075 埃** | 0.065 埃 | 正确 |
| v3 | [`v3_d3pm`](v3_d3pm) | 原子种类改用 **D3PM 离散扩散**（mask 型） | 0.088 埃 | **0.048 埃** | 正确 |

> 统一协议：每个版本取 8 个生成样本、固定随机种子、坐标误差用**最小镜像**计算、CPU 训练 4000 步。

## 每版在做什么

- **v1**：把整条链路跑通——前向加噪、噪声预测训练、DDPM 反向采样，从纯噪声生成一个晶体。
  坐标与晶格都用连续高斯扩散，原子种类用 one-hot 连续松弛。
- **v2**：解决对称性。晶格改用 `G = L·L^T`（度量张量）表示，对旋转天然不变；
  消息传递改用最小镜像的**相对位移**；并把「置换对称」做成开关（含数值验证与负收益实验）。
- **v3**：把原子种类从「连续松弛 + argmax」升级为真正的**类别离散扩散（D3PM）**，
  支持吸收型（mask）与均匀型（uniform）两种前向，后者用于对比。

## 一个通用的评估提醒（重要）

晶体的分数坐标是**周期性**的：**0 和 1 是同一个位置**。
若用朴素的 `|gen - target|`，把「生成 0.995」与「目标 0.0」当成差了 0.995，
会严重高估坐标误差，甚至得出与实际相反的结论。

本仓库所有坐标误差一律用**最小镜像**计算：

```
diff = gen - target;  diff = diff - round(diff);  MAE = |diff|
```

折算成卡氏距离即为真实空间偏差（单位埃）。这是这个任务里最容易踩的坑。

## 运行方式

每个版本都是独立的自包含目录（`data/` + `src/` + `scripts/` + `tests/`），
只需要 `torch` 和 `numpy`：

```bash
python v1_periodic_gnn/tests/test_smoke.py
python v2_equivariant_gnn/tests/test_symmetry.py
python v3_d3pm/tests/test_d3pm.py

python v3_d3pm/scripts/train.py --steps 4000 --samples 4
```

各版本的实现细节、验证内容与局限见其各自的 `README.md`。

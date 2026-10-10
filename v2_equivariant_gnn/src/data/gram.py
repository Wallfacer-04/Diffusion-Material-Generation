"""晶格的旋转不变表示：Gram 矩阵（度量张量）G = L · L^T。

对晶格做旋转，等价于右乘一个正交矩阵：L → L·R。
此时

    G = L·L^T  →  L·R·R^T·L^T = L·L^T = G

保持不变。也就是说，G 丢掉了"晶体朝向"这一自由度，只保留形状与尺寸。
用 G 的 6 个独立分量来表示晶格，模型就天然对旋转不变——不需要任何等变层。
"""

from __future__ import annotations

import numpy as np


def gram6_from_lattice(lattice: np.ndarray) -> np.ndarray:
    """(3, 3) 晶格矩阵（每一行是一个晶格向量）→ (6,) [Gxx, Gyy, Gzz, Gxy, Gxz, Gyz]。"""
    g = np.asarray(lattice, dtype=np.float64) @ np.asarray(lattice, dtype=np.float64).T
    return np.array([g[0, 0], g[1, 1], g[2, 2], g[0, 1], g[0, 2], g[1, 2]], dtype=np.float64)


def gram6_to_matrix(g: np.ndarray) -> np.ndarray:
    """(6,) → (3, 3) 对称矩阵。"""
    g = np.asarray(g, dtype=np.float64)
    matrix = np.zeros((3, 3), dtype=np.float64)
    matrix[0, 0], matrix[1, 1], matrix[2, 2] = g[0], g[1], g[2]
    matrix[0, 1] = matrix[1, 0] = g[3]
    matrix[0, 2] = matrix[2, 0] = g[4]
    matrix[1, 2] = matrix[2, 1] = g[5]
    return matrix


def project_positive_definite(matrix: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """把对称矩阵投影到正定：对称化后把特征值抬到 eps 以上。"""
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, eps, None)
    return (vectors * values) @ vectors.T


def lattice_from_gram6(g: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """(6,) → (3, 3) 晶格矩阵。做法：投影到正定，再 Cholesky 分解。

    Cholesky 给出下三角矩阵 L，满足 L·L^T = G。它和真实晶格只差一个旋转，
    物理上是同一个晶体，因此这是一个规范的（无旋转歧义的）取法。
    """
    matrix = project_positive_definite(gram6_to_matrix(g), eps)
    return np.linalg.cholesky(matrix)

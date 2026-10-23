"""SeaDiff 12 通道输入拼接。

固定顺序为 ``[x_t, U, H, J]``，每项 3 通道。E3 在调用前把 H 槽位置零，
并用 ``J_t = U + w(t)(J - U)`` 替换 J。
"""

import torch
from torch import Tensor

INPUT_ORDER = "paper_u_h_j"


def pack_paper_order(
    noisy: Tensor,
    underwater: Tensor,
    histogram: Tensor,
    physical: Tensor,
) -> Tensor:
    """把四个 ``[B,3,H,W]`` 条件拼成 ``[B,12,H,W]``。"""
    for name, value in (
        ("noisy", noisy),
        ("underwater", underwater),
        ("histogram", histogram),
        ("physical", physical),
    ):
        if value.ndim != 4 or value.shape[1] != 3:
            raise ValueError(f"{name} 必须是 [B,3,H,W]。")
    if not (
        noisy.shape == underwater.shape == histogram.shape == physical.shape
    ):
        raise ValueError("pack_paper_order 四个输入形状必须一致。")
    return torch.cat((noisy, underwater, histogram, physical), dim=1)

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
    # 通道 0:3 是可变状态 x_t，3:6 是 U，6:9 是 H，9:12 是 J。
    # 函数只负责排列；E3 调用方应先传入 H=0 和当前 t 对应的 J_t。
    # 12 通道形状正确并不能发现 H/J 交换，因此训练、蒸馏、推理须共用此入口。
    return torch.cat((noisy, underwater, histogram, physical), dim=1)

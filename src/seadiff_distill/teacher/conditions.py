"""E3 动态物理条件。

网络输入顺序固定为 ``[x_t, U, H, J]``，每项 3 通道。E3 只改变 H/J 两个
槽位，不改变网络结构、扩散公式或损失：

- H 槽位全零；
- J 槽位替换为 ``J_t = U + w(t) * (J - U)``；
- ``w(t) = 0.1 + 0.9 * (1 - alpha_bar_t)^2``。

``w(t)`` 是物理先验 J 的输入插值强度，不是网络内部的注意力权重。
训练与推理共用本模块，alpha_bar 固定为 CPU FP32 上的 canonical 日程，
避免不同设备计算 linspace/cumprod 的舍入差异。
"""

from typing import Dict, Tuple

import torch
from torch import Tensor

TRAIN_TIMESTEPS = 1000
BETA_START = 1e-6
BETA_END = 0.02
W_LOW = 0.1
W_SCALE = 0.9


def canonical_alpha_bar() -> Tensor:
    """线性 beta 日程的 alpha_bar（CPU FP32）。"""
    # 此处 beta_t 是扩散噪声日程；不是 Beta_UNet(U) 估计的 RGB 水体衰减系数。
    # alpha_bar_t 表示从干净图到第 t 步仍保留的累计信号比例。
    betas = torch.linspace(BETA_START, BETA_END, TRAIN_TIMESTEPS)
    return torch.cumprod(1.0 - betas, dim=0)


def dynamic_weight(alpha_bar_t: Tensor) -> Tensor:
    """``w(t) = 0.1 + 0.9 * (1 - alpha_bar_t)^2``。"""
    # t 越大噪声越强，w 越接近 1；反向 999→0 时 w 由约 1 降到 0.1。
    # 这是预先规定的时间函数，无可学习参数，也不读取当前 x_t 的内容。
    return W_LOW + W_SCALE * (1.0 - alpha_bar_t) ** 2


def _check_image(name: str, value: Tensor) -> None:
    if value.ndim != 4 or value.shape[1] != 3:
        raise ValueError(f"{name} 必须是 [B,3,H,W]。")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} 包含 NaN 或 Inf。")


def dynamic_physical_slots(
    underwater: Tensor,
    histogram: Tensor,
    physical: Tensor,
    *,
    timesteps: Tensor,
    alpha_bar: Tensor,
) -> Tuple[Tensor, Tensor]:
    """返回 ``(H_slot, J_t)``；H_slot 为全零，J_t 按样本时间步插值。"""
    _check_image("underwater", underwater)
    _check_image("histogram", histogram)
    _check_image("physical", physical)
    if not (
        underwater.shape == histogram.shape == physical.shape
        and underwater.dtype == histogram.dtype == physical.dtype
    ):
        raise ValueError(
            "underwater/histogram/physical 的形状与 dtype 必须一致。"
        )

    batch = underwater.shape[0]
    if timesteps.ndim != 1 or timesteps.shape[0] != batch:
        raise ValueError(f"timesteps 必须是 [{batch}]。")
    if timesteps.min().item() < 0 or timesteps.max().item() >= TRAIN_TIMESTEPS:
        raise ValueError("timesteps 越界。")

    # 每幅样本可以有不同的 t；[B] 查表后转为 [B,1,1,1]，广播到 RGB 全图。
    alpha_bar_t = alpha_bar.to(
        device=underwater.device, dtype=underwater.dtype
    )[timesteps]
    weight = dynamic_weight(alpha_bar_t).reshape(batch, 1, 1, 1)
    # physical 是由 U 生成的静态 J；每步只重算插值，不从 x_t 重新估计 J。
    j_t = underwater + weight * (physical - underwater)
    if not torch.isfinite(j_t).all().item():
        raise ValueError("J_t 包含 NaN 或 Inf。")
    # 保留 H 的三通道槽位以兼容原骨干，但 E3 不使用其中的颜色直方图内容。
    return torch.zeros_like(histogram), j_t


def describe() -> Dict[str, object]:
    return {
        "mode": "dynamic_physical",
        "histogram_slot": "zeros",
        "physical_slot": "U + w(t)(J - U)",
        "w_formula": "w(t) = 0.1 + 0.9 * (1 - alpha_bar_t)^2",
        "beta_used": True,
        "note": "w(t) 是 J 的输入插值强度，不是网络内部注意力权重。",
    }

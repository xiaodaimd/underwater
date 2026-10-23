"""物理先验图 J 的计算。

J 由 Beta 网络、深度图和环境光共同决定：

    T = exp(-beta * depth)        # 透射率
    S = 1 - T                     # 散射
    J = clamp((U - S * A) / T, 0, 1)

其中 U 为水下图，A 为上游 ``get_A`` 估计的环境光。教师训练时 Beta 参与
反传；学生蒸馏与推理时 Beta 冻结，J 只作为条件。
"""

from typing import Tuple

import torch
from torch import Tensor

from ..models.seadiff_unet import Beta_UNet
from ..models.seadiff_utils import get_A


def ambient_light_from_input(underwater: Tensor) -> Tensor:
    """逐图调用上游 ``get_A``，返回与输入同形状的环境光。"""
    return torch.stack(
        [get_A(image) for image in underwater],
        dim=0,
    ).to(device=underwater.device, dtype=torch.float32)


def compute_physical_image(
    *,
    underwater: Tensor,
    depth: Tensor,
    beta_predictor: Beta_UNet,
    return_preclamp: bool = False,
) -> Tuple[Tensor, Tensor] | Tuple[Tensor, Tensor, Tensor]:
    """计算可微的物理恢复图。

    underwater: ``[B,3,H,W]``，[0,1]
    depth:      ``[B,1,H,W]`` 或 ``[B,3,H,W]``，逐图已归一化到 [0,1]
    返回 ``(beta[B,3,1,1], J[B,3,H,W])``；``return_preclamp=True`` 时额外
    返回 clamp 前的 J，仅用于训练诊断，不改变前向数值。
    """
    if underwater.ndim != 4 or underwater.shape[1] != 3:
        raise ValueError("underwater 必须为 [B,3,H,W]。")
    if depth.ndim != 4 or depth.shape[1] not in (1, 3):
        raise ValueError("depth 必须为 [B,1,H,W] 或 [B,3,H,W]。")
    if (
        depth.shape[0] != underwater.shape[0]
        or depth.shape[2:] != underwater.shape[2:]
    ):
        raise ValueError("depth 与 underwater 的 batch/空间尺寸不一致。")
    if (
        not torch.isfinite(underwater).all().item()
        or not torch.isfinite(depth).all().item()
    ):
        raise ValueError("underwater/depth 包含 NaN 或 Inf。")
    if (depth < 0.0).any().item() or (depth > 1.0).any().item():
        raise ValueError("depth 必须在 [0,1] 范围内。")

    batch = underwater.shape[0]
    predicted_beta = beta_predictor(underwater)

    # 官方 Beta_UNet 在 batch=1 时会 squeeze 掉 batch 维。
    if batch == 1 and predicted_beta.shape == (3, 1, 1):
        predicted_beta = predicted_beta.unsqueeze(0)

    if predicted_beta.shape != (batch, 3, 1, 1):
        raise RuntimeError(
            f"Beta 输出形状异常：{tuple(predicted_beta.shape)}"
        )
    if predicted_beta.dtype != torch.float32:
        raise RuntimeError("Beta 前向必须使用 FP32。")
    if predicted_beta.device != underwater.device:
        raise RuntimeError("Beta 输出与输入设备不一致。")
    if not torch.isfinite(predicted_beta).all().item():
        raise RuntimeError("Beta 输出包含 NaN 或 Inf。")
    if (
        (predicted_beta < 0.0) | (predicted_beta > 1.0)
    ).any().item():
        raise RuntimeError("Beta 输出必须在 [0,1] 范围内。")

    # 单通道深度按灰度复制到 RGB；已归一化，不再二次归一化。
    depth_rgb = (
        depth.expand(-1, 3, -1, -1) if depth.shape[1] == 1 else depth
    )
    ambient_light = ambient_light_from_input(underwater)

    attenuation = torch.exp(-predicted_beta * depth_rgb)
    transmission = attenuation.clamp(0.0, 1.0)
    scattering = (1.0 - attenuation).clamp(0.0, 1.0)

    # beta, depth 均在 [0,1]，transmission >= exp(-1)，分母不会为零。
    preclamp = (
        underwater - scattering * ambient_light
    ) / transmission
    physical_image = preclamp.clamp(0.0, 1.0)

    if not torch.isfinite(physical_image).all().item():
        raise RuntimeError("物理恢复图包含 NaN 或 Inf。")

    if return_preclamp:
        return predicted_beta, physical_image, preclamp
    return predicted_beta, physical_image

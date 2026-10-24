"""SeaDiff 训练用扩散日程（线性 beta，T=1000，x0 预测）。"""

from typing import Optional

import torch
from torch import Tensor, nn

TRAIN_TIMESTEPS = 1000
BETA_START = 1e-6
BETA_END = 0.02


class TrainingDiffusion(nn.Module):
    """只实现官方训练所用的 noisy_image 与系数。"""

    def __init__(self, timesteps: int = TRAIN_TIMESTEPS):
        super().__init__()
        if type(timesteps) is not int or timesteps < 2:
            raise ValueError("timesteps 必须是 >=2 的整数。")
        scale = 1000 / timesteps
        # beta_t 为每个扩散步的噪声方差；alpha_bar_t 为累计信号保留量。
        # sampling_steps 减少只会跳过推理时间点，不应据此重建训练日程。
        betas = torch.linspace(
            BETA_START * scale, BETA_END * scale, timesteps
        ).float()
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.timesteps = timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer(
            "sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar)
        )

    def noisy_image(
        self,
        timesteps: Tensor,
        clean: Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ):
        """官方 GaussianDiffusion.noisy_image 的等价实现。"""
        if timesteps.ndim != 1 or timesteps.shape[0] != clean.shape[0]:
            raise ValueError("timesteps 必须是 [B]。")
        if timesteps.min().item() < 0 or timesteps.max().item() >= self.timesteps:
            raise ValueError("timesteps 越界。")

        noise = torch.randn(
            clean.shape,
            generator=generator,
            device=clean.device,
            dtype=clean.dtype,
        )
        # timesteps 为 [B]；重塑后广播到每幅图的通道和空间维。
        sqrt_alpha = self.sqrt_alpha_bar[timesteps].reshape(
            -1, *((1,) * (clean.ndim - 1))
        )
        sqrt_one_minus = self.sqrt_one_minus_alpha_bar[timesteps].reshape(
            -1, *((1,) * (clean.ndim - 1))
        )
        # q(x_t|Y)：x_t = sqrt(alpha_bar_t)Y + sqrt(1-alpha_bar_t)epsilon。
        # clean 是训练参考图 Y；本项目去噪器监督目标仍是 Y（x0），不是 epsilon。
        return sqrt_alpha * clean + sqrt_one_minus * noise, noise

"""SeaDiff 教师采样器：确定性 DDIM 与随机 DDPM。

两者都消费同一份 12 通道输入：
``[初始噪声 x_T, U, H, J]``，并按 x0 预测（PRE_ORI=True）更新状态。

- DDIM：eta=0，NFE 等于 ``sampling_steps``；
- DDPM：复现上游 1000 步随机采样，NFE=1000。
"""

from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn

from ..models.seadiff_unet import Denoise_UNet

TRAIN_TIMESTEPS = 1000
BETA_START = 1e-6
BETA_END = 0.02


def _linear_betas(device=None) -> Tensor:
    return torch.linspace(
        BETA_START,
        BETA_END,
        TRAIN_TIMESTEPS,
        dtype=torch.float32,
        device=device,
    )


def _validate_packed(packed_input: Tensor, expected_device: torch.device) -> None:
    if not isinstance(packed_input, Tensor):
        raise TypeError("packed_input 必须是 Tensor。")
    if packed_input.ndim != 4:
        raise ValueError("输入必须为 [B,12,H,W]。")
    batch, channels, height, width = packed_input.shape
    if batch < 1 or channels != 12:
        raise ValueError("输入必须包含非空 batch 和 12 个通道。")
    if height < 8 or width < 8 or height % 8 != 0 or width % 8 != 0:
        raise ValueError("输入高宽必须是正的 8 的倍数。")
    if packed_input.dtype != torch.float32:
        raise TypeError("教师采样只支持 FP32。")
    if packed_input.device != expected_device:
        raise ValueError("输入和采样器必须位于同一设备。")
    if not torch.isfinite(packed_input).all().item():
        raise ValueError("教师输入包含 NaN 或 Inf。")


class DDIMTeacherSampler(nn.Module):
    """1000 步训练日程上的确定性 DDIM（eta=0）。"""

    def __init__(self, denoiser: Denoise_UNet, *, sampling_steps: int):
        super().__init__()
        if not isinstance(denoiser, Denoise_UNet):
            raise TypeError("denoiser 必须是 Denoise_UNet。")
        if not denoiser.is_noise:
            raise ValueError("教师必须保留时间步条件。")
        if type(sampling_steps) is not int or not 2 <= sampling_steps <= 1000:
            raise ValueError("DDIM 步数必须是 2 到 1000 的整数。")
        if any(p.dtype != torch.float32 for p in denoiser.parameters()):
            raise ValueError("教师采样只支持 FP32 权重。")

        self.denoiser = denoiser
        self.sampling_steps = sampling_steps
        # 在原 1000 步训练日程上取递减子网格，端点仍为 999 与 0。
        # sampling_steps 是网络前向次数，不是重新训练一个更短的噪声日程。
        timesteps = tuple(
            round(index * 999 / (sampling_steps - 1))
            for index in reversed(range(sampling_steps))
        )
        if len(set(timesteps)) != len(timesteps) or list(timesteps) != sorted(
            timesteps, reverse=True
        ):
            raise RuntimeError("DDIM 时间网格必须唯一且严格递减。")
        self.register_buffer(
            "timesteps_tensor", torch.tensor(timesteps, dtype=torch.long)
        )

        betas = _linear_betas()
        alpha_bar = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer(
            "sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar)
        )
        self.eval()

    def train(self, mode: bool = True):
        """采样器始终 eval，冻结骨干的 Dropout。"""
        super().train(mode)
        self.denoiser.eval()
        return self

    def get_extra_state(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "sampler": "ddim",
            "eta": 0.0,
            "prediction_type": "x0",
            "training_timesteps": TRAIN_TIMESTEPS,
            "schedule": "linear",
            "beta_start": BETA_START,
            "beta_end": BETA_END,
            "sampling_timesteps": [
                int(v) for v in self.timesteps_tensor.tolist()
            ],
            "clip_x0": False,
        }

    def set_extra_state(self, state) -> None:
        if state != self.get_extra_state():
            raise ValueError("加载的采样协议与当前采样器不一致。")

    @torch.no_grad()
    def forward(self, packed_input: Tensor) -> Tensor:
        _validate_packed(packed_input, self.sqrt_alpha_bar.device)
        self.denoiser.eval()

        # 仅前三通道是可变 x_t，后九通道 [U,H,J] 在本静态采样器中保持不变。
        # E3 的动态 J_t 必须使用 condition_samplers.py 中的条件感知子类。
        state = packed_input[:, :3]
        conditions = packed_input[:, 3:]
        batch = packed_input.shape[0]
        timesteps = [int(v) for v in self.timesteps_tensor.tolist()]

        for index, timestep in enumerate(timesteps):
            time_batch = torch.full(
                (batch,), timestep, dtype=torch.long, device=packed_input.device
            )
            predicted_x0 = self.denoiser(
                torch.cat((state, conditions), dim=1), time_batch
            )
            self._check_prediction(predicted_x0, state)

            # eta=0：给定相同初始噪声和条件后轨迹确定，末次直接返回 x0_hat。
            if index == len(timesteps) - 1:
                return predicted_x0

            next_timestep = timesteps[index + 1]
            # x0 预测参数化：先由 x_t 与 x0_hat 反推出当前 epsilon_hat。
            predicted_noise = (
                state - self.sqrt_alpha_bar[timestep] * predicted_x0
            ) / self.sqrt_one_minus_alpha_bar[timestep]
            state = (
                self.sqrt_alpha_bar[next_timestep] * predicted_x0
                + self.sqrt_one_minus_alpha_bar[next_timestep]
                * predicted_noise
            )

        raise RuntimeError("教师采样未产生最终输出。")

    @staticmethod
    def _check_prediction(predicted_x0: Tensor, state: Tensor) -> None:
        if (
            predicted_x0.shape != state.shape
            or predicted_x0.device != state.device
            or predicted_x0.dtype != torch.float32
        ):
            raise RuntimeError("教师输出形状/设备/精度异常。")


class DDPMTeacherSampler(nn.Module):
    """复现上游 PRE_ORI=True 的 1000 步随机 DDPM。"""

    def __init__(self, denoiser: Denoise_UNet):
        super().__init__()
        if not isinstance(denoiser, Denoise_UNet):
            raise TypeError("denoiser 必须是 Denoise_UNet。")
        if not denoiser.is_noise:
            raise ValueError("教师必须保留时间步条件。")
        if any(p.dtype != torch.float32 for p in denoiser.parameters()):
            raise ValueError("教师采样只支持 FP32 权重。")
        self.denoiser = denoiser

        betas = _linear_betas()
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.nn.functional.pad(
            alpha_bar, (1, 0), value=1.0
        )[:TRAIN_TIMESTEPS]
        # posterior_var 保留为日程信息；下方采样实际遵循上游约定使用 beta_t。
        posterior_var = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("gammas", alpha_bar)
        self.register_buffer("posterior_var", posterior_var)
        self.register_buffer("coeff1", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "coeff2",
            torch.sqrt(1.0 / alphas)
            * (1.0 - alphas)
            / torch.sqrt(1.0 - alpha_bar),
        )
        self.register_buffer("sqrt_gammas", torch.sqrt(alpha_bar))
        self.register_buffer(
            "sqrt_one_minus_gammas", torch.sqrt(1.0 - alpha_bar)
        )
        self.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.denoiser.eval()
        return self

    def get_extra_state(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "sampler": "ddpm",
            "eta": None,
            "prediction_type": "x0",
            "training_timesteps": TRAIN_TIMESTEPS,
            "schedule": "linear",
            "beta_start": BETA_START,
            "beta_end": BETA_END,
            "sampling_timesteps": list(range(TRAIN_TIMESTEPS - 1, -1, -1)),
            "clip_x0": False,
            "stochastic": True,
        }

    def set_extra_state(self, state) -> None:
        if state != self.get_extra_state():
            raise ValueError("加载的采样协议与当前采样器不一致。")

    @torch.no_grad()
    def forward(
        self,
        packed_input: Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        _validate_packed(packed_input, self.gammas.device)
        self.denoiser.eval()

        # 仅前三通道是可变 x_t，后九通道 [U,H,J] 在本静态采样器中保持不变。
        # E3 的动态 J_t 必须使用 condition_samplers.py 中的条件感知子类。
        state = packed_input[:, :3]
        conditions = packed_input[:, 3:]
        batch = packed_input.shape[0]

        for timestep in range(TRAIN_TIMESTEPS - 1, -1, -1):
            time_batch = torch.full(
                (batch,), timestep, dtype=torch.long, device=packed_input.device
            )
            predicted_x0 = self.denoiser(
                torch.cat((state, conditions), dim=1), time_batch
            )
            if (
                predicted_x0.shape != state.shape
                or predicted_x0.device != state.device
                or predicted_x0.dtype != torch.float32
            ):
                raise RuntimeError("教师输出形状/设备/精度异常。")

            # x0 预测参数化：先由 x_t 与 x0_hat 反推出当前 epsilon_hat。
            predicted_noise = (
                state - self.sqrt_gammas[timestep] * predicted_x0
            ) / self.sqrt_one_minus_gammas[timestep]
            previous_mean = (
                self.coeff1[timestep] * state
                - self.coeff2[timestep] * predicted_noise
            )

            # 每个非末步另采高斯噪声；因此固定初始噪声还不足以固定 DDPM 全轨迹，
            # 还需固定此 generator。t=0 只取均值，不再注入噪声。
            if timestep > 0:
                variance = self.betas[timestep]
                noise = torch.randn(
                    state.shape,
                    generator=generator,
                    device=state.device,
                    dtype=state.dtype,
                )
                state = previous_mean + torch.sqrt(variance) * noise
            else:
                state = previous_mean

        if not torch.isfinite(state).all().item():
            raise RuntimeError("教师最终输出包含 NaN 或 Inf。")
        return state

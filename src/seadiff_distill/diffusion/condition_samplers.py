"""条件随 t 变化的教师采样器（E3 推理用）。

DDPM/DDIM 更新公式与静态采样器完全一致；唯一区别是每个时间步调用
``condition_fn(timestep, conditions)`` 重建条件块。E3 用它把 J 槽位
替换为 ``J_t = U + w(t)(J - U)``。
"""

from typing import Any, Callable, Dict, Optional

import torch
from torch import Tensor

from .samplers import (
    TRAIN_TIMESTEPS,
    DDIMTeacherSampler,
    DDPMTeacherSampler,
    _validate_packed,
)

ConditionFn = Callable[[int, Tensor], Tensor]


class ConditionAwareDDPMTeacherSampler(DDPMTeacherSampler):
    """每步重建条件的 1000 步随机 DDPM。"""

    def __init__(
        self,
        denoiser,
        *,
        condition_fn: Optional[ConditionFn] = None,
        condition_desc: str = "none",
    ):
        super().__init__(denoiser)
        self.condition_fn = condition_fn
        self.condition_desc = str(condition_desc)

    def get_extra_state(self) -> Dict[str, Any]:
        state = super().get_extra_state()
        state["condition_aware"] = True
        state["condition_fn"] = self.condition_desc
        return state

    def set_extra_state(self, state) -> None:
        super().set_extra_state(state)

    @torch.no_grad()
    def forward(
        self,
        packed_input: Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        _validate_packed(packed_input, self.gammas.device)
        self.denoiser.eval()

        state = packed_input[:, :3]
        conditions = packed_input[:, 3:]
        batch = packed_input.shape[0]

        for timestep in range(TRAIN_TIMESTEPS - 1, -1, -1):
            current_conditions = conditions
            if self.condition_fn is not None:
                current_conditions = self.condition_fn(
                    timestep, conditions
                )
                if (
                    current_conditions.shape != conditions.shape
                    or current_conditions.device != conditions.device
                    or current_conditions.dtype != conditions.dtype
                ):
                    raise RuntimeError(
                        "condition_fn 输出形状/设备/dtype 与条件块不一致。"
                    )

            time_batch = torch.full(
                (batch,), timestep, dtype=torch.long, device=packed_input.device
            )
            predicted_x0 = self.denoiser(
                torch.cat((state, current_conditions), dim=1), time_batch
            )
            if (
                predicted_x0.shape != state.shape
                or predicted_x0.device != state.device
                or predicted_x0.dtype != torch.float32
            ):
                raise RuntimeError("教师输出形状/设备/精度异常。")

            predicted_noise = (
                state - self.sqrt_gammas[timestep] * predicted_x0
            ) / self.sqrt_one_minus_gammas[timestep]
            previous_mean = (
                self.coeff1[timestep] * state
                - self.coeff2[timestep] * predicted_noise
            )

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


class ConditionAwareDDIMTeacherSampler(DDIMTeacherSampler):
    """每步重建条件的确定性 DDIM（eta=0），用于 E3 推理。"""

    def __init__(
        self,
        denoiser,
        *,
        sampling_steps: int,
        condition_fn: Optional[ConditionFn] = None,
        condition_desc: str = "none",
    ):
        super().__init__(denoiser, sampling_steps=sampling_steps)
        self.condition_fn = condition_fn
        self.condition_desc = str(condition_desc)

    def get_extra_state(self) -> Dict[str, Any]:
        state = super().get_extra_state()
        state["condition_aware"] = True
        state["condition_fn"] = self.condition_desc
        return state

    def set_extra_state(self, state) -> None:
        super().set_extra_state(state)

    @torch.no_grad()
    def forward(self, packed_input: Tensor) -> Tensor:
        _validate_packed(packed_input, self.sqrt_alpha_bar.device)
        self.denoiser.eval()

        state = packed_input[:, :3]
        conditions = packed_input[:, 3:]
        batch = packed_input.shape[0]
        timesteps = [int(v) for v in self.timesteps_tensor.tolist()]

        for index, timestep in enumerate(timesteps):
            current_conditions = conditions
            if self.condition_fn is not None:
                current_conditions = self.condition_fn(
                    timestep, conditions
                )
                if (
                    current_conditions.shape != conditions.shape
                    or current_conditions.device != conditions.device
                    or current_conditions.dtype != conditions.dtype
                ):
                    raise RuntimeError(
                        "condition_fn 输出形状/设备/dtype 与条件块不一致。"
                    )
            time_batch = torch.full(
                (batch,),
                timestep,
                dtype=torch.long,
                device=packed_input.device,
            )
            predicted_x0 = self.denoiser(
                torch.cat((state, current_conditions), dim=1),
                time_batch,
            )
            if (
                predicted_x0.shape != state.shape
                or predicted_x0.device != state.device
                or predicted_x0.dtype != torch.float32
            ):
                raise RuntimeError("教师输出形状/设备/精度异常。")

            if index == len(timesteps) - 1:
                if not torch.isfinite(predicted_x0).all().item():
                    raise RuntimeError("教师最终输出包含 NaN 或 Inf。")
                return predicted_x0

            next_timestep = timesteps[index + 1]
            predicted_noise = (
                state - self.sqrt_alpha_bar[timestep] * predicted_x0
            ) / self.sqrt_one_minus_alpha_bar[timestep]
            state = (
                self.sqrt_alpha_bar[next_timestep] * predicted_x0
                + self.sqrt_one_minus_alpha_bar[next_timestep]
                * predicted_noise
            )

        raise RuntimeError("教师采样未产生最终输出。")

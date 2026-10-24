"""搜索阶段的一次双层更新。

顺序：

1. 同一参数状态下分别计算内层（480 张）与外层（120 张）损失
   ``L_total = L_original + 0.5 * L_distill``；
2. 用 BiLaLoRA 式 (7) 估计门控超梯度，裁剪 1.0 后用 Adam 更新门控；
3. 用新门控重新前向内层，计算 Adapter 梯度，裁剪 1.0 后用 AdamW 更新。

外层每 4 个内层 batch 执行一次；其余 batch 只执行第 3 步。
全程 FP32，出现 NaN/Inf 立即报错。
"""

import math
from typing import Dict, Iterable, Tuple

import torch
from torch import Tensor, nn

from ..losses.distillation_loss import DistillationLoss
from ..models.adapter_student import OneStepSeaDiffStudent
from .hypergradients import compute_bilalora_hypergradient

ADAPTER_LR_DEFAULT = 1e-4
ADAPTER_WEIGHT_DECAY_DEFAULT = 1e-4
GATE_LR_DEFAULT = 3e-4
GATE_WEIGHT_DECAY_DEFAULT = 1e-3
GRAD_CLIP_DEFAULT = 1.0


def _check_optimizer(
    optimizer: torch.optim.Optimizer,
    expected_parameters: Tuple[nn.Parameter, ...],
    name: str,
    expected_class: type,
    expected_options: Dict[str, object],
) -> None:
    """核验优化器类型、参数绑定与关键超参。"""
    if not isinstance(optimizer, expected_class):
        raise TypeError(f"{name} 必须是 {expected_class.__name__}。")
    if len(optimizer.param_groups) != 1:
        raise ValueError(f"{name} 必须恰好包含一个参数组。")

    group = optimizer.param_groups[0]
    if [id(p) for p in group["params"]] != [
        id(p) for p in expected_parameters
    ]:
        raise ValueError(f"{name} 的参数及顺序必须与参数接口一致。")

    learning_rate = group["lr"]
    if (
        type(learning_rate) not in (int, float)
        or not math.isfinite(learning_rate)
        or learning_rate < 0
    ):
        raise ValueError(f"{name} 的学习率必须是有限非负数。")

    for option, expected in expected_options.items():
        if group.get(option) != expected:
            raise ValueError(
                f"{name}.{option} 必须为 {expected}，实际为 {group.get(option)}。"
            )


def _require_finite(
    tensors: Iterable[Tensor],
    description: str,
) -> None:
    flags = torch.stack(
        [torch.isfinite(tensor.detach()).all() for tensor in tensors]
    )
    if not flags.all().item():
        raise FloatingPointError(f"{description} 包含 NaN 或 Inf。")


def _clip_gradients(
    gradients: Tuple[Tensor, ...],
    clip_value: float,
) -> Tuple[Tuple[Tensor, ...], float, float]:
    """按全局范数裁剪梯度，返回 (梯度, 裁剪前范数, 裁剪后范数)。"""
    device = gradients[0].device
    total = torch.zeros((), dtype=torch.float64, device=device)
    for gradient in gradients:
        total += gradient.detach().to(dtype=torch.float64).pow(2).sum()
    norm_before = float(total.sqrt().item())

    if norm_before == 0.0:
        return gradients, 0.0, 0.0

    scale = min(1.0, clip_value / norm_before)
    clipped = tuple(
        (gradient.detach() * scale).to(
            dtype=gradient.dtype, device=gradient.device
        )
        for gradient in gradients
    )
    total = torch.zeros((), dtype=torch.float64, device=device)
    for gradient in clipped:
        total += gradient.to(dtype=torch.float64).pow(2).sum()
    return clipped, norm_before, float(total.sqrt().item())


class BilevelOptimizer:
    """门控超梯度更新 + Adapter 内层更新。"""

    def __init__(
        self,
        student: OneStepSeaDiffStudent,
        adapter_optimizer: torch.optim.AdamW,
        architecture_optimizer: torch.optim.Adam,
        *,
        loss_module: DistillationLoss | None = None,
        epsilon: float = 1e-8,
        grad_clip: float = GRAD_CLIP_DEFAULT,
    ):
        if not isinstance(student, OneStepSeaDiffStudent):
            raise TypeError("student 必须是 OneStepSeaDiffStudent。")

        for name, value in (
            ("epsilon", epsilon),
            ("grad_clip", grad_clip),
        ):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} 必须是有限数值。")
        if epsilon <= 0 or grad_clip <= 0:
            raise ValueError("epsilon 与 grad_clip 必须为正数。")

        self.student = student
        self.adapter_optimizer = adapter_optimizer
        self.architecture_optimizer = architecture_optimizer
        self.loss_module = (
            DistillationLoss() if loss_module is None else loss_module
        )
        self.weights = tuple(student.adapter_parameters())
        self.architecture = tuple(student.architecture_parameters())
        self.epsilon = epsilon
        self.grad_clip = grad_clip
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        if not self.weights or not self.architecture:
            raise ValueError("Adapter 和门控参数组不能为空。")

        parameters = self.weights + self.architecture
        expected_ids = {id(p) for p in parameters}
        trainable_ids = {
            id(p) for p in self.student.parameters() if p.requires_grad
        }
        if (
            len(expected_ids) != len(parameters)
            or expected_ids != trainable_ids
        ):
            raise ValueError(
                "Adapter 与门控应互不重叠，并覆盖全部可训练参数。"
            )
        if any(
            p.requires_grad for p in self.student.backbone.parameters()
        ):
            raise ValueError("搜索阶段的学生骨干必须冻结。")
        if any(p.dtype != torch.float32 for p in parameters):
            raise TypeError("双层更新全程使用 FP32。")

        _check_optimizer(
            self.adapter_optimizer,
            self.weights,
            "adapter_optimizer",
            torch.optim.AdamW,
            {
                "lr": ADAPTER_LR_DEFAULT,
                "weight_decay": ADAPTER_WEIGHT_DECAY_DEFAULT,
            },
        )
        _check_optimizer(
            self.architecture_optimizer,
            self.architecture,
            "architecture_optimizer",
            torch.optim.Adam,
            {
                "lr": GATE_LR_DEFAULT,
                "weight_decay": GATE_WEIGHT_DECAY_DEFAULT,
            },
        )

    @torch.enable_grad()
    def step(
        self,
        *,
        inner_packed_input: Tensor,
        inner_timesteps: Tensor,
        inner_reference: Tensor,
        inner_teacher_target: Tensor,
        update_gates: bool,
        outer_packed_input: Tensor | None = None,
        outer_timesteps: Tensor | None = None,
        outer_reference: Tensor | None = None,
        outer_teacher_target: Tensor | None = None,
    ) -> Dict[str, float]:
        self._validate_configuration()
        self.student.train()
        self.student.zero_grad(set_to_none=True)

        try:
            device = self.weights[0].device
            inner = {
                "inner_packed_input": inner_packed_input,
                "inner_timesteps": inner_timesteps,
                "inner_reference": inner_reference,
                "inner_teacher_target": inner_teacher_target,
            }
            for name, value in inner.items():
                if not isinstance(value, Tensor):
                    raise TypeError(f"{name} 必须是 Tensor。")
                if name.endswith("timesteps"):
                    if value.dtype != torch.long or value.device != device:
                        raise ValueError(
                            f"{name} 必须是与学生同设备的 long 张量。"
                        )
                elif value.dtype != torch.float32 or value.device != device:
                    raise ValueError(
                        f"{name} 必须是与学生同设备的 FP32 张量。"
                    )
                inner[name] = value.detach()

            _require_finite(inner.values(), "内层训练输入")

            # 第一次内层前向用于损失记录和门控超梯度；此时还未更新任何参数。
            inner_before = self.loss_module(
                self.student(
                    inner["inner_packed_input"],
                    inner["inner_timesteps"],
                ),
                inner["inner_reference"],
                inner["inner_teacher_target"],
            )
            _require_finite(inner_before.values(), "内层组合损失")

            logs: Dict[str, float] = {
                "inner_loss_total_before": float(
                    inner_before["loss_total"].detach().item()
                ),
                "inner_loss_original_before": float(
                    inner_before["loss_original"].detach().item()
                ),
                "inner_loss_distill_before": float(
                    inner_before["loss_distill"].detach().item()
                ),
                "inner_loss_mse_before": float(
                    inner_before["loss_mse"].detach().item()
                ),
                "inner_loss_perceptual_before": float(
                    inner_before["loss_perceptual"].detach().item()
                ),
            }

            if update_gates:
                outer = {
                    "outer_packed_input": outer_packed_input,
                    "outer_timesteps": outer_timesteps,
                    "outer_reference": outer_reference,
                    "outer_teacher_target": outer_teacher_target,
                }
                for name, value in outer.items():
                    if value is None:
                        raise TypeError(
                            f"update_gates=True 时 {name} 不能为 None。"
                        )
                    if not isinstance(value, Tensor):
                        raise TypeError(f"{name} 必须是 Tensor。")
                    if name.endswith("timesteps"):
                        if value.dtype != torch.long or value.device != device:
                            raise ValueError(
                                f"{name} 必须是与学生同设备的 long 张量。"
                            )
                    elif value.dtype != torch.float32 or value.device != device:
                        raise ValueError(
                            f"{name} 必须是与学生同设备的 FP32 张量。"
                        )
                    outer[name] = value.detach()

                _require_finite(outer.values(), "外层训练输入")

                # 外层数据来自dev划分，但同样含GT原始损失和KD项；不是仅算GT L1。
                # 与inner_before保持同一参数点，才能比较两份梯度的投影。
                outer_before = self.loss_module(
                    self.student(
                        outer["outer_packed_input"],
                        outer["outer_timesteps"],
                    ),
                    outer["outer_reference"],
                    outer["outer_teacher_target"],
                )
                _require_finite(outer_before.values(), "外层组合损失")
                logs["outer_loss_total_before"] = float(
                    outer_before["loss_total"].detach().item()
                )

                # 调用式(7)秩一近似；没有展开可微AdamW虚拟更新。
                gate_gradients, diagnostics = (
                    compute_bilalora_hypergradient(
                        inner_before["loss_total"],
                        outer_before["loss_total"],
                        adapter_parameters=self.weights,
                        architecture_parameters=self.architecture,
                        epsilon=self.epsilon,
                    )
                )
                gate_gradients, pre_norm, post_norm = _clip_gradients(
                    gate_gradients, self.grad_clip
                )
                logs["gate_grad_norm_before_clip"] = pre_norm
                logs["gate_grad_norm_after_clip"] = post_norm

                for parameter, gradient in zip(
                    self.architecture, gate_gradients
                ):
                    parameter.grad = gradient.detach()
                self.architecture_optimizer.step()
                _require_finite(self.architecture, "更新后的门控")
                logs.update(diagnostics)
                self.student.zero_grad(set_to_none=True)

            # 门控可能刚被更新，需重新构图再求Adapter梯度；目标与噪声仍复用本批常量。
            # 当前实现即使update_gates=False也重算内层，因此搜索每批有两次内层前向，
            # 门控更新批还多一次外层前向；“部署单步”不等于训练每批仅一次前向。
            inner_after_gate = self.loss_module(
                self.student(
                    inner["inner_packed_input"],
                    inner["inner_timesteps"],
                ),
                inner["inner_reference"],
                inner["inner_teacher_target"],
            )
            _require_finite(
                inner_after_gate.values(), "门控更新后的内层损失"
            )

            weight_gradients = torch.autograd.grad(
                inner_after_gate["loss_total"],
                self.weights,
                create_graph=False,
                retain_graph=False,
                allow_unused=False,
            )
            _require_finite(weight_gradients, "Adapter 梯度")
            weight_gradients, pre_norm, post_norm = _clip_gradients(
                weight_gradients, self.grad_clip
            )
            logs["adapter_grad_norm_before_clip"] = pre_norm
            logs["adapter_grad_norm_after_clip"] = post_norm

            for parameter, gradient in zip(self.weights, weight_gradients):
                parameter.grad = gradient.detach()
            self.adapter_optimizer.step()
            _require_finite(self.weights, "更新后的 Adapter")

            # 记录的是更新门控之后、更新Adapter之前的损失，不是完整step后的新损失。
            logs["inner_loss_total_after_gate"] = float(
                inner_after_gate["loss_total"].detach().item()
            )
            logs["adapter_lr"] = float(
                self.adapter_optimizer.param_groups[0]["lr"]
            )
            logs["architecture_lr"] = float(
                self.architecture_optimizer.param_groups[0]["lr"]
            )
            return logs

        finally:
            self.student.zero_grad(set_to_none=True)

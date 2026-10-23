"""BiLaLoRA 式 (7) 的稳定化秩一超梯度近似。"""

import math
from typing import Dict, Iterable, Tuple

import torch
from torch import Tensor, nn


def _flatten_gradients(gradients: Tuple[Tensor, ...]) -> Tensor:
    """拼接梯度并以 FP64 累计投影项。"""
    return torch.cat(
        [
            gradient.detach().reshape(-1)
            for gradient in gradients
        ],
        dim=0,
    ).to(dtype=torch.float64)


def compute_bilalora_hypergradient(
    inner_loss: Tensor,
    outer_loss: Tensor,
    *,
    adapter_parameters: Iterable[nn.Parameter],
    architecture_parameters: Iterable[nn.Parameter],
    epsilon: float = 1e-8,
) -> Tuple[Tuple[Tensor, ...], Dict[str, float]]:
    """
    计算门控 logits 的近似超梯度：

        c = dot(d_w outer, d_w inner)
            / (norm(d_w inner)^2 + epsilon)

        hypergradient = d_a outer - stopgrad(c) * d_a inner

    w 为 Adapter 参数，a 为门控 logits。
    sigmoid 的导数由 autograd 自动处理。

    调用约定：
    1. 两个损失来自同一参数状态下的两次独立前向。
    2. 计算两份损失及本函数执行期间，不更新参数。
    3. 所有搜索候选参数都必须连接到两份损失。
    4. 本函数消费两份计算图，不修改或清空参数的 .grad。
    5. 更新门控后，需重新前向计算 Adapter 更新所用的损失。

    返回梯度顺序与 architecture_parameters 的顺序一致。
    """
    if (
        type(epsilon) not in (int, float)
        or not math.isfinite(epsilon)
        or epsilon <= 0
    ):
        raise ValueError("epsilon 必须是有限正数。")

    weights = tuple(adapter_parameters)
    architecture = tuple(architecture_parameters)

    if not weights or not architecture:
        raise ValueError("Adapter 与门控参数组均不能为空。")

    parameters = weights + architecture

    if any(
        not isinstance(parameter, nn.Parameter)
        for parameter in parameters
    ):
        raise TypeError("参数组必须包含 nn.Parameter。")

    parameter_ids = [id(parameter) for parameter in parameters]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise ValueError("参数组内及两组之间不能包含重复参数。")

    reference = parameters[0]

    if reference.dtype not in (torch.float32, torch.float64):
        raise TypeError("超梯度计算支持 FP32 或 FP64 参数。")

    for index, parameter in enumerate(parameters):
        if not parameter.is_leaf or not parameter.requires_grad:
            raise ValueError(
                f"第 {index} 个参数必须是可训练的叶子参数。"
            )

        if (
            parameter.numel() == 0
            or parameter.layout != torch.strided
        ):
            raise ValueError("参数必须是非空的普通稠密张量。")

        if (
            parameter.device != reference.device
            or parameter.dtype != reference.dtype
        ):
            raise ValueError("两组参数必须具有相同设备和精度。")

    if inner_loss is outer_loss:
        raise ValueError("内外层损失必须分别前向计算。")

    for name, loss in (
        ("inner_loss", inner_loss),
        ("outer_loss", outer_loss),
    ):
        if not isinstance(loss, Tensor):
            raise TypeError(f"{name} 必须是 Tensor。")

        if loss.ndim != 0 or not loss.requires_grad:
            raise ValueError(f"{name} 必须是保留计算图的标量。")

        if (
            loss.device != reference.device
            or loss.dtype != reference.dtype
        ):
            raise ValueError(
                f"{name} 必须与参数具有相同设备和精度。"
            )

        if not torch.isfinite(loss.detach()).item():
            raise FloatingPointError(f"{name} 包含 NaN 或 Inf。")

    # 严格要求计算图连通。
    # 初始零梯度合法；未连接的参数会直接报错。
    inner_gradients = torch.autograd.grad(
        inner_loss,
        parameters,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )
    outer_gradients = torch.autograd.grad(
        outer_loss,
        parameters,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )

    weight_count = len(weights)

    inner_weight_vector = _flatten_gradients(
        inner_gradients[:weight_count]
    )
    outer_weight_vector = _flatten_gradients(
        outer_gradients[:weight_count]
    )
    inner_arch_vector = _flatten_gradients(
        inner_gradients[weight_count:]
    )
    outer_arch_vector = _flatten_gradients(
        outer_gradients[weight_count:]
    )

    vectors = (
        inner_weight_vector,
        outer_weight_vector,
        inner_arch_vector,
        outer_arch_vector,
    )

    finite_flags = torch.stack(
        [torch.isfinite(vector).all() for vector in vectors]
    )
    if not finite_flags.all().item():
        raise FloatingPointError("内层或外层梯度包含 NaN 或 Inf。")

    # 投影只使用 Adapter 梯度，不把门控梯度混入 w。
    norm_squared = torch.dot(
        inner_weight_vector,
        inner_weight_vector,
    )
    gradient_dot = torch.dot(
        outer_weight_vector,
        inner_weight_vector,
    )

    if not torch.isfinite(
        torch.stack((norm_squared, gradient_dot))
    ).all().item():
        raise FloatingPointError("梯度投影计算出现溢出。")

    # 内层梯度为零时，分子也为零，c 自然等于零。
    coefficient = (
        gradient_dot / (norm_squared + epsilon)
    ).detach()

    # 使用论文的减号，不在这里额外乘学习率。
    architecture_gradients = tuple(
        (
            outer_gradient.detach().to(torch.float64)
            - coefficient
            * inner_gradient.detach().to(torch.float64)
        ).to(dtype=parameter.dtype).detach()
        for parameter, inner_gradient, outer_gradient in zip(
            architecture,
            inner_gradients[weight_count:],
            outer_gradients[weight_count:],
        )
    )

    hypergradient_vector = _flatten_gradients(
        architecture_gradients
    )

    if not torch.isfinite(hypergradient_vector).all().item():
        raise FloatingPointError("门控超梯度包含 NaN 或 Inf。")

    diagnostic_values = torch.stack(
        (
            norm_squared.sqrt(),
            outer_weight_vector.norm(),
            gradient_dot,
            coefficient,
            outer_arch_vector.norm(),
            hypergradient_vector.norm(),
        )
    ).detach()

    if not torch.isfinite(diagnostic_values).all().item():
        raise FloatingPointError("超梯度诊断值出现非有限数值。")

    diagnostic_names = (
        "inner_adapter_grad_norm",
        "outer_adapter_grad_norm",
        "adapter_gradient_dot",
        "correction_coefficient",
        "direct_arch_grad_norm",
        "hypergradient_norm",
    )

    diagnostics = dict(
        zip(
            diagnostic_names,
            diagnostic_values.cpu().tolist(),
        )
    )

    return architecture_gradients, diagnostics
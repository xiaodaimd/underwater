"""从 Student 训练 checkpoint 重建推理模型。

训练 checkpoint 保留完整九分支结构（便于恢复与复现），推理时只启用
``selected_sites`` 中的 Top-3 分支。骨干必须由调用方提供，通常从 E3
教师的 denoiser 权重构造。
"""

from pathlib import Path
from typing import Any, Dict

import torch

from ..models.adapter_student import OneStepSeaDiffStudent
from ..models.seadiff_unet import Denoise_UNet

CHECKPOINT_KIND = "seadiff_distill_checkpoint"
SCHEMA_VERSION = 1


def load_student_training_checkpoint(
    path: str | Path,
    backbone: Denoise_UNet,
    *,
    device: str = "cpu",
) -> tuple[OneStepSeaDiffStudent, Dict[str, Any]]:
    """加载 ``finetune_epoch100.pt`` 一类的训练 checkpoint。

    返回已设 eval 的学生模型和精简元数据。只接受本项目的 checkpoint
    格式，不执行任何训练逻辑。
    """
    source = Path(path).expanduser().resolve(strict=True)
    payload = torch.load(
        source, map_location="cpu", weights_only=False
    )
    if (
        not isinstance(payload, dict)
        or payload.get("kind") != CHECKPOINT_KIND
        or payload.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError(f"不是 SeaDiff Student checkpoint：{source}")

    saved = payload.get("student")
    if not isinstance(saved, dict) or "state_dict" not in saved:
        raise ValueError("checkpoint 缺少 student.state_dict。")

    student = OneStepSeaDiffStudent(
        backbone,
        adapter_reduction=4,
        fixed_timestep=999,
    )
    # 先恢复完整训练结构，再恢复非state_dict的硬选择标记；否则会误执行九个分支。
    # 此函数仅重建学生，不负责Beta物理条件、初始噪声或部署输入的打包。
    student.load_state_dict(saved["state_dict"], strict=True)
    extra = saved.get("gates_extra_state")
    if extra is not None:
        student.gates.set_extra_state(extra)

    selected = saved.get("selected_sites")
    values = saved.get("fixed_gate_values")
    if saved.get("hard_selected"):
        if not selected or values is None:
            raise ValueError("硬选择 checkpoint 缺少位置或门控值。")
        names = student.site_names()
        fixed = {
            name: float(values[names.index(name)])
            for name in selected
        }
        student.configure_hard_selection(selected, fixed)
        student.freeze_unselected_adapters()

    # 默认前向时间步为999；单次推理仍需调用方提供[x_t,U,0,J_t]输入。
    student.to(device=device)
    student.eval()
    metadata = {
        "path": str(source),
        "checkpoint_name": payload.get("checkpoint_name"),
        "epoch": payload.get("epoch"),
        "phase": payload.get("phase"),
        "stage": payload.get("stage"),
        "selected_sites": list(student.active_site_names()),
        "hard_selected": student.is_hard_selected(),
        "fixed_gate_values": {
            name: float(value)
            for name, value in zip(
                student.site_names(),
                student.fixed_gate_values.detach().cpu().tolist(),
            )
        },
    }
    return student, metadata

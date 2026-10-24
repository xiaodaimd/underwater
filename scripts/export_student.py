"""把固定 Top-3 训练 checkpoint 导出为只含选中 Adapter 的部署权重。"""

import argparse
import json
from pathlib import Path

import torch

from common import (
    add_common_arguments,
    resolve_config,
)

from seadiff_distill.models.adapter_student import (
    load_deployment_student,
)
from seadiff_distill.models.teacher_loader import load_seadiff_teacher
from seadiff_distill.student.checkpoint import (
    load_student_training_checkpoint,
)
from seadiff_distill.training.checkpoint import (
    checkpoint_sha256,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, default_config="configs/student.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--deployment", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    config = resolve_config(args)
    output_dir = Path(config["output_dir"])
    checkpoint = Path(
        args.checkpoint
        or output_dir / "checkpoints" / "finetune_epoch100.pt"
    ).expanduser().resolve(strict=True)
    deployment = Path(
        args.deployment
        or output_dir / "deployment" / "top3_student.pt"
    ).expanduser().resolve()
    report_path = Path(
        args.report or output_dir / "deployment" / "export_report.json"
    ).expanduser().resolve()

    teacher_cfg = config["teacher"]
    teacher, _ = load_seadiff_teacher(
        denoiser_path=teacher_cfg["denoiser_path"],
        beta_predictor_path=teacher_cfg["beta_predictor_path"],
        denoiser_sha256=teacher_cfg["denoiser_sha256"],
        beta_predictor_sha256=teacher_cfg["beta_predictor_sha256"],
        beta_use_norm=bool(teacher_cfg.get("beta_use_norm", True)),
    )
    student, metadata = load_student_training_checkpoint(
        checkpoint, teacher.denoiser, device="cpu"
    )
    if not metadata["hard_selected"]:
        raise RuntimeError("checkpoint 尚未完成 Top-3 固定选择。")

    student.requires_grad_(False)
    student.eval()
    # 保留显式 gate 乘法，避免将 gate 乘入升维权重后产生浮点运算顺序差异。
    # 此文件不打包 E3 骨干/Beta/深度模型，推理端还需相应外部权重和条件。
    student.save_deployment(
        deployment,
        absorb_gates=False,
        metadata={
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_sha256": checkpoint_sha256(checkpoint),
            "epoch": metadata["epoch"],
            "selected_sites": metadata["selected_sites"],
            "fixed_gate_values": metadata["fixed_gate_values"],
        },
    )

    deployed = load_deployment_student(
        deployment, teacher.denoiser, device="cpu"
    )
    generator = torch.Generator(device="cpu").manual_seed(3407)
    # 合成 12 通道输入只验证结构导出等价性，不代表图像质量评估或完整数据测试。
    probe = torch.randn(2, 12, 64, 64, generator=generator)
    timesteps = torch.full((2,), 999, dtype=torch.long)
    with torch.no_grad():
        expected = student(probe, timesteps)
        actual = deployed(probe)
    max_diff = float((expected - actual).abs().max())

    total_params = sum(p.numel() for p in deployed.parameters())
    adapter_params = sum(
        p.numel() for p in deployed.adapter_parameters()
    )
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "deployment": str(deployment),
        "deployment_sha256": checkpoint_sha256(deployment),
        "selected_sites": metadata["selected_sites"],
        "fixed_gate_values": metadata["fixed_gate_values"],
        "gates_absorbed": False,
        "equivalence_max_abs_diff": max_diff,
        "equivalence_ok": max_diff <= 1e-6,
        "total_params": total_params,
        "adapter_params": adapter_params,
    }
    if not report["equivalence_ok"]:
        raise AssertionError(
            f"部署模型与 checkpoint 学生不一致：max|diff|={max_diff}"
        )
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

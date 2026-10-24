"""单步 Student 推理：一次 U-Net 前向输出增强图。

支持两种权重来源：
- 导出的 Top-3 部署权重（推荐，只含选中 Adapter）；
- 完整训练 checkpoint（包含九分支结构与门控）。
骨干始终从 E3 教师 denoiser 加载；输入为 ``[x_T, U, 0, J_999]``。
"""

import argparse
import json
from pathlib import Path

import torch

from common import (
    add_common_arguments,
    resolve_config,
    resolve_device,
)

from seadiff_distill.data.dataset import InferenceDataset
from seadiff_distill.data.packing import pack_paper_order
from seadiff_distill.data.physical import compute_physical_image
from seadiff_distill.evaluation.inference import (
    derive_initial_noise,
    save_prediction,
)
from seadiff_distill.evaluation.metrics import evaluate_batch, summarize
from seadiff_distill.models.adapter_student import (
    load_deployment_student,
)
from seadiff_distill.models.teacher_loader import load_seadiff_teacher
from seadiff_distill.student.checkpoint import (
    load_student_training_checkpoint,
)
from seadiff_distill.teacher.conditions import (
    canonical_alpha_bar,
    dynamic_physical_slots,
)


def safe_name(sample_id: str) -> str:
    return "".join(
        char if char.isalnum() or char in "-_." else "_"
        for char in sample_id
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, default_config="configs/student.yaml")
    parser.add_argument("--manifest", default="configs/data/uieb_test130.csv")
    parser.add_argument("--output-dir", default="outputs/infer_student")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--deployment", default=None)
    source.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--metrics-json",
        default=None,
        help="可选：在有 reference 的清单上输出 PSNR/L1 指标",
    )
    parser.add_argument(
        "--ssim",
        action="store_true",
        help="指标中额外计算 SSIM（需要 scikit-image）",
    )
    args = parser.parse_args()

    config = resolve_config(args)
    teacher_cfg = config["teacher"]
    data_cfg = config["data"]
    device = resolve_device(args.device)

    teacher, metadata = load_seadiff_teacher(
        denoiser_path=teacher_cfg["denoiser_path"],
        beta_predictor_path=teacher_cfg["beta_predictor_path"],
        denoiser_sha256=teacher_cfg["denoiser_sha256"],
        beta_predictor_sha256=teacher_cfg["beta_predictor_sha256"],
        beta_use_norm=bool(teacher_cfg.get("beta_use_norm", True)),
    )

    # 部署包只含选中的 Adapter/固定门控；两种加载方式都依赖外部 E3 骨干。
    if args.deployment:
        student = load_deployment_student(
            args.deployment, teacher.denoiser, device="cpu"
        )
        source_path = Path(args.deployment).expanduser().resolve(strict=True)
    else:
        student, student_metadata = load_student_training_checkpoint(
            args.checkpoint, teacher.denoiser, device="cpu"
        )
        source_path = Path(args.checkpoint).expanduser().resolve(strict=True)
        print(
            "checkpoint:",
            json.dumps(student_metadata, ensure_ascii=False),
            flush=True,
        )

    teacher = teacher.to(device).eval()
    student = student.to(device).eval()

    dataset = InferenceDataset(
        manifest_path=args.manifest,
        data_root=data_cfg["data_root"],
        image_size=tuple(data_cfg["image_size"]),
    )
    alpha_bar = canonical_alpha_bar().to(device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    count = min(len(dataset), args.limit) if args.limit else len(dataset)
    metric_rows = []

    for index in range(count):
        sample = dataset[index]
        sample_id = sample["sample_id"]
        underwater = sample["underwater"].unsqueeze(0).to(device)
        depth = sample["depth"].unsqueeze(0).to(device)
        histogram = sample["histogram"].unsqueeze(0).to(device)

        # 推理条件只由 U 及其缓存生成；reference 仅在后面的可选指标分支读取。
        with torch.no_grad():
            _, physical = compute_physical_image(
                underwater=underwater,
                depth=depth,
                beta_predictor=teacher.beta_predictor,
            )
            # 部署从纯噪声出发；区别于蒸馏训练中由 GT 加噪构造的 x_t。
            initial = derive_initial_noise(
                sample_id,
                global_seed=args.seed,
                shape=underwater.shape[1:],
                device=device,
            ).unsqueeze(0)
            timesteps = torch.full(
                (1,), 999, dtype=torch.long, device=device
            )
            histogram_slot, physical_slot = dynamic_physical_slots(
                underwater,
                histogram,
                physical,
                timesteps=timesteps,
                alpha_bar=alpha_bar,
            )
            packed = pack_paper_order(
                initial,
                underwater,
                histogram_slot,
                physical_slot,
            )
            # 未传 timesteps 时模型使用固定 t=999；没有教师链或阶段循环。
            # “一次前向”计数针对 U-Net，前面仍要执行 Beta 和物理条件计算。
            prediction = student(packed)

        save_prediction(
            prediction[0], output_dir / f"{safe_name(sample_id)}.png"
        )
        if args.metrics_json:
            if "reference" not in sample:
                raise ValueError(
                    "计算指标需要清单中的 reference 列："
                    f"{sample_id}"
                )
            reference = sample["reference"].unsqueeze(0).to(device)
            values = evaluate_batch(
                prediction, reference, compute_ssim=args.ssim
            )[0]
            metric_rows.append({"sample_id": sample_id, **values})

        if (index + 1) % 10 == 0 or index + 1 == count:
            print(
                f"[infer_student] {index + 1}/{count} "
                f"last={sample_id}",
                flush=True,
            )

    metrics_summary = None
    if args.metrics_json:
        metric_keys = ["psnr", "l1"]
        if args.ssim:
            metric_keys.append("ssim")
        metrics_summary = {
            key: summarize(row[key] for row in metric_rows)
            for key in metric_keys
        }
        metrics_path = Path(args.metrics_json).expanduser().resolve()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(
                {
                    "samples": count,
                    "rows": metric_rows,
                    "summary": metrics_summary,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "source": str(source_path),
                "samples": count,
                "output_dir": str(output_dir),
                "unet_forward_per_sample": 1,
                "denoiser_sha256": metadata["denoiser"]["sha256"],
                "metrics": metrics_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

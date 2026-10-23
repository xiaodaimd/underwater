"""教师推理：400k 静态条件或 E3 动态物理条件。

E3 每个时间步都重建 ``[x_t, U, 0, J_t]``；400k 使用固定
``[x_t, U, H, J]``。初始噪声由 sample_id 与 seed 确定性生成。
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
from seadiff_distill.diffusion.condition_samplers import (
    ConditionAwareDDIMTeacherSampler,
    ConditionAwareDDPMTeacherSampler,
)
from seadiff_distill.diffusion.samplers import (
    DDIMTeacherSampler,
    DDPMTeacherSampler,
)
from seadiff_distill.evaluation.inference import (
    derive_initial_noise,
    derive_noise_seed,
    save_prediction,
)
from seadiff_distill.models.teacher_loader import load_seadiff_teacher
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
    add_common_arguments(
        parser, default_config="configs/teacher_e3.yaml"
    )
    parser.add_argument("--manifest", default="configs/data/uieb_test130.csv")
    parser.add_argument("--output-dir", default="outputs/infer_teacher")
    parser.add_argument(
        "--mode",
        choices=("paper", "dynamic_physical"),
        default=None,
        help="默认读取配置 inference.condition_mode",
    )
    parser.add_argument("--sampler", choices=("ddim", "ddpm"), default="ddim")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    config = resolve_config(args)
    inference = dict(config.get("inference") or {})
    mode = args.mode or inference.get("condition_mode")
    if mode not in ("paper", "dynamic_physical"):
        raise ValueError("配置缺少 inference.condition_mode。")
    if not inference.get("denoiser") or not inference.get(
        "beta_predictor"
    ):
        raise ValueError("配置缺少 inference 教师权重路径。")

    device = resolve_device(args.device)
    teacher, metadata = load_seadiff_teacher(
        denoiser_path=inference["denoiser"],
        beta_predictor_path=inference["beta_predictor"],
        denoiser_sha256=inference["denoiser_sha256"],
        beta_predictor_sha256=inference["beta_predictor_sha256"],
        beta_use_norm=bool(config["model"].get("beta_norm", True)),
    )
    teacher = teacher.to(device).eval()

    data_cfg = config["data"]
    dataset = InferenceDataset(
        manifest_path=args.manifest,
        data_root=data_cfg["data_root"],
        image_size=tuple(data_cfg.get("image_size", [336, 336])),
    )
    alpha_bar = canonical_alpha_bar().to(device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    count = min(len(dataset), args.limit) if args.limit else len(dataset)
    for index in range(count):
        sample = dataset[index]
        sample_id = sample["sample_id"]
        underwater = sample["underwater"].unsqueeze(0).to(device)
        depth = sample["depth"].unsqueeze(0).to(device)
        histogram = sample["histogram"].unsqueeze(0).to(device)

        with torch.no_grad():
            _, physical = compute_physical_image(
                underwater=underwater,
                depth=depth,
                beta_predictor=teacher.beta_predictor,
            )
            initial = derive_initial_noise(
                sample_id,
                global_seed=args.seed,
                shape=underwater.shape[1:],
                device=device,
            ).unsqueeze(0)
            timesteps = torch.full(
                (1,), 999, dtype=torch.long, device=device
            )

            if mode == "dynamic_physical":
                histogram_slot, physical_slot = dynamic_physical_slots(
                    underwater,
                    histogram,
                    physical,
                    timesteps=timesteps,
                    alpha_bar=alpha_bar,
                )

                def condition_fn(timestep, _conditions):
                    step = torch.full(
                        (1,), timestep, dtype=torch.long, device=device
                    )
                    h_slot, j_slot = dynamic_physical_slots(
                        underwater,
                        histogram,
                        physical,
                        timesteps=step,
                        alpha_bar=alpha_bar,
                    )
                    return torch.cat(
                        (underwater, h_slot, j_slot), dim=1
                    )

                if args.sampler == "ddim":
                    sampler = ConditionAwareDDIMTeacherSampler(
                        teacher.denoiser,
                        sampling_steps=args.steps,
                        condition_fn=condition_fn,
                        condition_desc=(
                            "E3 dynamic_physical per-step J_t"
                        ),
                    ).to(device)
                    generator = None
                else:
                    sampler = ConditionAwareDDPMTeacherSampler(
                        teacher.denoiser,
                        condition_fn=condition_fn,
                        condition_desc=(
                            "E3 dynamic_physical per-step J_t"
                        ),
                    ).to(device)
                    generator = torch.Generator(device=device)
                    generator.manual_seed(
                        derive_noise_seed(sample_id, args.seed)
                    )
            else:
                histogram_slot = histogram
                physical_slot = physical
                if args.sampler == "ddim":
                    sampler = DDIMTeacherSampler(
                        teacher.denoiser, sampling_steps=args.steps
                    ).to(device)
                    generator = None
                else:
                    sampler = DDPMTeacherSampler(
                        teacher.denoiser
                    ).to(device)
                    generator = torch.Generator(device=device)
                    generator.manual_seed(
                        derive_noise_seed(sample_id, args.seed)
                    )

            packed = pack_paper_order(
                initial,
                underwater,
                histogram_slot,
                physical_slot,
            )
            if generator is None:
                prediction = sampler(packed)
            else:
                prediction = sampler(packed, generator=generator)

        save_prediction(
            prediction[0], output_dir / f"{safe_name(sample_id)}.png"
        )
        if (index + 1) % 10 == 0 or index + 1 == count:
            print(
                f"[infer_teacher] {index + 1}/{count} "
                f"last={sample_id}",
                flush=True,
            )

    print(
        json.dumps(
            {
                "mode": mode,
                "sampler": args.sampler,
                "steps": args.steps if args.sampler == "ddim" else 1000,
                "samples": count,
                "output_dir": str(output_dir),
                "weights": {
                    "denoiser_sha256": metadata["denoiser"]["sha256"],
                    "beta_sha256": metadata["beta_predictor"]["sha256"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

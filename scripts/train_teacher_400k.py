"""400k SeaDiff 教师训练入口（UIEB 600 对）。"""

import argparse
import json

from common import (
    add_common_arguments,
    resolve_config,
    resolve_device,
)

from seadiff_distill.config import save_config_snapshot
from seadiff_distill.determinism import configure_determinism
from seadiff_distill.teacher.trainer import TeacherTrainer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(
        parser, default_config="configs/teacher_400k.yaml"
    )
    parser.add_argument("--resume", default=None, help="teacher_last.pt")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="仅用于调试；正式训练使用配置中的 400000",
    )
    args = parser.parse_args()

    config = resolve_config(args)
    device = resolve_device(args.device)
    seed = int(config.get("seed", 3407))
    determinism = configure_determinism(seed, strict=True)

    trainer = TeacherTrainer(config, device)
    save_config_snapshot(config, trainer.output_dir / "config.json")
    summary = trainer.train(
        resume=args.resume, max_iterations=args.max_iterations
    )
    summary["determinism"] = determinism
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

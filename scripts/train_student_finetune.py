"""固定 Top-3 Adapter 的 40 轮蒸馏微调，需要先完成搜索阶段。"""

import argparse
import json
from pathlib import Path

from common import (
    add_common_arguments,
    resolve_config,
    resolve_device,
    training_seed,
)

from seadiff_distill.config import save_config_snapshot
from seadiff_distill.determinism import configure_determinism
from seadiff_distill.student.trainer import StudentDistillationTrainer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, default_config="configs/student.yaml")
    parser.add_argument(
        "--resume",
        default=None,
        help="默认 checkpoints/selected_top3.pt；也可指定微调 rolling.pt",
    )
    args = parser.parse_args()

    config = resolve_config(args)
    checkpoint = (
        Path(args.resume).expanduser()
        if args.resume
        else Path(config["output_dir"])
        / "checkpoints"
        / "selected_top3.pt"
    )
    if not checkpoint.is_file():
        parser.error(
            "找不到已选择 Top-3 的断点，请先运行 "
            f"scripts/train_student_search.py（期望 {checkpoint}）"
        )

    device = resolve_device(args.device)
    determinism = configure_determinism(training_seed(config), strict=True)

    trainer = StudentDistillationTrainer(config, device)
    save_config_snapshot(config, trainer.output_dir / "config.json")
    start_epoch = trainer.restore(checkpoint)
    if start_epoch < 61:
        raise RuntimeError("搜索阶段尚未结束，不能开始固定选择微调。")

    final_path = trainer.checkpoint_dir / "finetune_epoch100.pt"
    if start_epoch > 60 + trainer.finetune_epochs:
        print(
            json.dumps(
                {
                    "status": "already_completed",
                    "checkpoint": str(final_path),
                },
                ensure_ascii=False,
            )
        )
        return 0

    trainer.run_finetune(start_epoch=start_epoch)
    trainer.close_logging()
    print(
        json.dumps(
            {
                "checkpoint": str(final_path),
                "epoch": 60 + trainer.finetune_epochs,
                "determinism": determinism,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

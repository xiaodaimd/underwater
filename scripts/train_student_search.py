"""Student Adapter 架构搜索：5 轮预热 + 55 轮双层优化 + Top-3 选择。"""

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
        help="warmup/rolling/阶段边界 checkpoint；不填则从头开始",
    )
    args = parser.parse_args()

    config = resolve_config(args)
    device = resolve_device(args.device)
    determinism = configure_determinism(training_seed(config), strict=True)

    trainer = StudentDistillationTrainer(config, device)
    save_config_snapshot(config, trainer.output_dir / "config.json")

    start_epoch = 1
    if args.resume:
        start_epoch = trainer.restore(Path(args.resume))

    # 此入口只完成 epoch1..60；阶段教师快照与 Top-3 选择由 trainer 管理。
    if start_epoch <= trainer.warmup_epochs:
        trainer.run_warmup(start_epoch=start_epoch)
        start_epoch = trainer.warmup_epochs + 1
    if start_epoch <= 60:
        trainer.run_search(start_epoch=start_epoch)
    else:
        raise RuntimeError(
            "该断点已进入固定选择微调阶段，请使用 "
            "scripts/train_student_finetune.py。"
        )

    # selected_top3.pt 是下一入口的交接点，尚未包含 epoch61..100 微调结果。
    trainer.close_logging()
    selection = json.loads(
        (trainer.output_dir / "selection_details.json").read_text(
            encoding="utf-8"
        )
    )
    print(
        json.dumps(
            {
                "checkpoint": str(
                    trainer.checkpoint_dir / "selected_top3.pt"
                ),
                "selected_top3": selection["selected_top3"],
                "selected_gate_values": selection[
                    "selected_gate_values"
                ],
                "determinism": determinism,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""单步 Adapter Student 的渐进蒸馏训练。

总预算固定为 5 预热 + 55 搜索 + 40 微调 = 100 epochs：

- 预热：9 个 Adapter 全部可训，门控固定 0.5，教师 E3@G(1000)，
  学生网格 G(500)；
- 搜索：S1..S10 共 10 个嵌套网格阶段，内层 AdamW 更新 Adapter，
  外层 Adam + BiLaLoRA 式 (7) 超梯度更新门控，外层每 4 个内层 batch
  执行一次；每阶段把当前学生冻结为下一阶段的教师快照；
- epoch 60：按门控值取 Top-3，固定门控并冻结落选分支；
- 微调：600 图，AdamW lr 5e-5，教师为冻结的 2 步快照，学生网格 G(1)。

损失：

    L_total = L_original + 0.5 * L_distill

``L_original`` 是教师 MSE + 0.1 VGG19；``L_distill`` 是学生 x0 与
教师反解 x0 目标的 mean MSE。教师输出全部 detach。
"""

import csv
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..data.dataset import SeaDiffDataset
from ..data.packing import pack_paper_order
from ..data.physical import compute_physical_image
from ..determinism import capture_rng_state, restore_rng_state
from ..losses.distillation_loss import DistillationLoss
from ..losses.perceptual import VGG19PerceptualLoss
from ..losses.teacher_loss import TeacherLoss
from ..models.adapter_student import OneStepSeaDiffStudent
from ..models.teacher_loader import load_seadiff_teacher
from ..teacher.conditions import (
    canonical_alpha_bar,
    dynamic_physical_slots,
)
from ..training.checkpoint import (
    checkpoint_sha256,
    load_checkpoint,
    save_checkpoint,
)
from .bilevel_optimizer import BilevelOptimizer
from .grids import build_nested_grids, verify_nested_grids
from .selection import select_top_k
from .stage_teacher import StageTeacherSnapshot, promote_stage_teacher
from .targets import (
    DistillationTargetProvider,
    TeacherChainRunner,
    build_dynamic_physical_pack_fn,
)

CHECKPOINT_KIND = "seadiff_distill_checkpoint"
SCHEMA_VERSION = 1
WARMUP_GATE_VALUE = 0.5
BATCH_SIZE = 4
OUTER_EVERY_INNER = 4
TOP_K = 3

# (阶段名, 教师步数, 学生步数, 轮数)；搜索总轮数必须为 55。
# “步数”在阶段表中对应网格点数；区间端点由嵌套网格决定，实际t不等于这些数字。
STAGE_PLAN: Tuple[Tuple[str, int, int, int], ...] = (
    ("S1", 1000, 500, 2),
    ("S2", 500, 250, 2),
    ("S3", 250, 125, 3),
    ("S4", 125, 63, 3),
    ("S5", 63, 32, 4),
    ("S6", 32, 16, 5),
    ("S7", 16, 8, 6),
    ("S8", 8, 4, 7),
    ("S9", 4, 2, 9),
    ("S10", 2, 1, 14),
)


def _sha256_json(value: Any) -> str:
    text = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _public_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """去掉仅用于记录配置来源的私有键，保证断点哈希稳定。"""
    return {
        key: value
        for key, value in config.items()
        if not str(key).startswith("_")
    }


@dataclass(frozen=True)
class StageBoundary:
    """一个阶段的轮次范围和网格大小。"""

    name: str
    teacher_steps: int
    student_steps: int
    first_epoch: int
    last_epoch: int
    warmup: bool = False


def build_epoch_plan(
    warmup_epochs: int, search_epochs: int
) -> Tuple[StageBoundary, ...]:
    """把固定阶段表映射为 epoch 计划。"""
    if warmup_epochs != 5:
        raise ValueError("预热必须为 5 epochs。")
    total = sum(item[3] for item in STAGE_PLAN)
    if total != search_epochs or search_epochs != 55:
        raise ValueError("搜索阶段必须为 55 epochs（固定阶段表）。")

    boundaries = [
        StageBoundary(
            name="warmup",
            teacher_steps=1000,
            student_steps=500,
            first_epoch=1,
            last_epoch=warmup_epochs,
            warmup=True,
        )
    ]
    cursor = warmup_epochs + 1
    for name, teacher_steps, student_steps, epochs in STAGE_PLAN:
        boundaries.append(
            StageBoundary(
                name=name,
                teacher_steps=teacher_steps,
                student_steps=student_steps,
                first_epoch=cursor,
                last_epoch=cursor + epochs - 1,
            )
        )
        cursor += epochs
    if cursor != 61:
        raise AssertionError("搜索阶段应结束于 epoch 60。")
    return tuple(boundaries)


class StudentDistillationTrainer:
    """预热、双层搜索、Top-3 与固定选择微调的统一训练器。"""

    def __init__(self, config: Mapping[str, Any], device: torch.device):
        self.config = dict(config)
        self.device = device
        training_cfg = config["training"]

        self.seed = int(training_cfg["seed"])
        self.batch_size = int(training_cfg.get("batch_size", BATCH_SIZE))
        if self.batch_size != BATCH_SIZE:
            raise ValueError("学生训练 batch_size 必须为 4。")
        self.outer_every_inner = int(
            training_cfg.get("outer_every_inner", OUTER_EVERY_INNER)
        )
        if self.outer_every_inner != OUTER_EVERY_INNER:
            raise ValueError("外层更新频率必须为每 4 个内层 batch。")
        self.lambda_kd = float(training_cfg.get("lambda_kd", 0.5))
        if self.lambda_kd != 0.5:
            raise ValueError("lambda_kd 必须为 0.5。")

        self.output_dir = Path(config["output_dir"])
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        for directory in (self.checkpoint_dir, self.log_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.events_path = self.log_dir / "events.jsonl"

        self._emit(
            "engine_init",
            {
                "seed": self.seed,
                "batch_size": self.batch_size,
                "outer_every_inner": self.outer_every_inner,
                "lambda_kd": self.lambda_kd,
                "device": str(device),
            },
        )

        self.grids = build_nested_grids()
        self.grid_audit = verify_nested_grids(self.grids)
        self._emit("nested_grids_verified", {"audit": self.grid_audit})

        self.alpha_bar = canonical_alpha_bar()
        self._load_teacher()
        self._build_student()
        # 学生必须从 CPU 骨干构造，构造完成后再搬设备。
        self.teacher = self.teacher.to(self.device)
        self.teacher.eval()
        self._build_data()
        self._build_loss()
        self._build_plan()

        self.snapshots: Dict[str, StageTeacherSnapshot] = {}
        self.current_boundary: Optional[StageBoundary] = None
        self.current_provider: Optional[DistillationTargetProvider] = None
        self.global_batch_index = 0
        # inner与outer各自用批计数驱动目标随机流；本字段当前未写入checkpoint。
        self.dev_batch_counter = 0
        self._restored_optimizer_states: Dict[str, Any] = {}
        self._csv_handle = None
        self._csv_writer = None
        self._csv_fields: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def _load_teacher(self) -> None:
        teacher_cfg = self.config["teacher"]
        if teacher_cfg.get("condition_mode", "dynamic_physical") != (
            "dynamic_physical"
        ):
            raise ValueError("学生教师条件必须为 dynamic_physical。")
        self.teacher, self.teacher_metadata = load_seadiff_teacher(
            denoiser_path=teacher_cfg["denoiser_path"],
            beta_predictor_path=teacher_cfg["beta_predictor_path"],
            denoiser_sha256=teacher_cfg["denoiser_sha256"],
            beta_predictor_sha256=teacher_cfg["beta_predictor_sha256"],
            beta_use_norm=bool(teacher_cfg.get("beta_use_norm", True)),
        )
        self.teacher.eval()
        self.e3_runner = TeacherChainRunner(
            denoise_fn=self.teacher.denoiser,
            make_packed=build_dynamic_physical_pack_fn(self.alpha_bar),
            alpha_bar=self.alpha_bar,
            identity={
                "kind": "e3_dynamic_physical_teacher",
                "denoiser_sha256": teacher_cfg["denoiser_sha256"],
                "beta_sha256": teacher_cfg["beta_predictor_sha256"],
                "condition_mode": "dynamic_physical",
                "beta_use_norm": bool(
                    teacher_cfg.get("beta_use_norm", True)
                ),
            },
        )
        self._emit("e3_teacher_loaded", dict(self.e3_runner.identity))

    def _build_student(self) -> None:
        model_cfg = self.config["model"]
        reduction = int(model_cfg["adapter_reduction"])
        if reduction != 4:
            raise ValueError("Adapter bottleneck 固定为 C/4。")
        self.student = OneStepSeaDiffStudent(
            self.teacher.denoiser,
            adapter_reduction=reduction,
            fixed_timestep=999,
        ).to(self.device)
        self.student.train()
        self.backbone_identity = {
            "source": "E3_dynamic_physical",
            "denoiser_sha256": self.config["teacher"]["denoiser_sha256"],
            "frozen": True,
            "note": "骨干为 E3 权重深拷贝，全程冻结。",
        }
        self._emit(
            "student_built",
            {
                "adapter_reduction": reduction,
                "sites": list(self.student.site_names()),
            },
        )

    def _build_data(self) -> None:
        data_cfg = self.config["data"]
        common = {
            "data_root": data_cfg["data_root"],
            "image_size": tuple(data_cfg["image_size"]),
        }
        self.inner_dataset = SeaDiffDataset(
            manifest_path=data_cfg["inner_manifest"], **common
        )
        self.dev_dataset = SeaDiffDataset(
            manifest_path=data_cfg["dev_manifest"], **common
        )
        self.full_dataset = SeaDiffDataset(
            manifest_path=data_cfg["train_manifest"], **common
        )
        if len(self.inner_dataset) != 480:
            raise ValueError("内层清单必须为 480 张。")
        if len(self.dev_dataset) != 120:
            raise ValueError("外层清单必须为 120 张。")
        if len(self.full_dataset) != 600:
            raise ValueError("微调清单必须为 600 张。")
        self._emit(
            "datasets_loaded",
            {
                "inner": len(self.inner_dataset),
                "dev": len(self.dev_dataset),
                "full": len(self.full_dataset),
                "inner_manifest": data_cfg["inner_manifest"],
                "dev_manifest": data_cfg["dev_manifest"],
                "train_manifest": data_cfg["train_manifest"],
            },
        )

    def _build_loss(self) -> None:
        vgg19 = VGG19PerceptualLoss(device=self.device)
        base_loss = TeacherLoss(perceptual=vgg19)
        self.loss_module = DistillationLoss(
            base_loss=base_loss, lambda_kd=self.lambda_kd
        )
        self._emit(
            "loss_built",
            {
                "base_loss": "teacher_loss_v1_mse_plus_vgg19_0.1",
                "perceptual": vgg19.describe(),
                "lambda_kd": self.lambda_kd,
                "distill_form": (
                    "mean MSE(student_x0, stopgrad(teacher_x0_target))"
                ),
            },
        )

    def _build_plan(self) -> None:
        training_cfg = self.config["training"]
        self.warmup_epochs = int(training_cfg["warmup_epochs"])
        self.search_epochs = int(training_cfg["search_epochs"])
        self.finetune_epochs = int(training_cfg["finetune_epochs"])
        if self.finetune_epochs != 40:
            raise ValueError("微调必须为 40 epochs。")
        if int(training_cfg.get("top_k", TOP_K)) != TOP_K:
            raise ValueError("Top-K 必须为 3。")
        self.epoch_plan = build_epoch_plan(
            self.warmup_epochs, self.search_epochs
        )
        self._emit(
            "epoch_plan_built",
            {
                "warmup_epochs": self.warmup_epochs,
                "search_epochs": self.search_epochs,
                "finetune_epochs": self.finetune_epochs,
                "stages": [
                    {
                        "name": b.name,
                        "teacher_steps": b.teacher_steps,
                        "student_steps": b.student_steps,
                        "epochs": [b.first_epoch, b.last_epoch],
                    }
                    for b in self.epoch_plan
                ],
            },
        )

    def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        record = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "event": event,
            **payload,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # 条件和目标
    # ------------------------------------------------------------------
    def _batch_conditions(
        self, batch: Mapping[str, Any]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        underwater = batch["underwater"].to(self.device)
        reference = batch["reference"].to(self.device)
        depth = batch["depth"].to(self.device)
        histogram = batch["histogram"].to(self.device)
        with torch.no_grad():
            # 冻结Beta每批算一次物理图J；后续教师各步与学生再按自己的t插值得到J_t。
            _, physical = compute_physical_image(
                underwater=underwater,
                depth=depth,
                beta_predictor=self.teacher.beta_predictor,
            )
        return underwater, reference, histogram, physical

    def _pack_student_input(
        self,
        x_t: Tensor,
        underwater: Tensor,
        histogram: Tensor,
        physical: Tensor,
        timestep_start: Tensor,
    ) -> Tensor:
        histogram_slot, physical_slot = dynamic_physical_slots(
            underwater,
            histogram,
            physical,
            timesteps=timestep_start,
            alpha_bar=self.alpha_bar,
        )
        return pack_paper_order(
            x_t, underwater, histogram_slot, physical_slot
        )

    def _compute_targets(
        self,
        batch: Mapping[str, Any],
        batch_index: int,
    ) -> Tuple[Dict[str, Tensor], Tensor, Tensor, Tensor, Tensor]:
        if self.current_provider is None:
            raise RuntimeError("尚未设置当前阶段教师。")
        underwater, reference, histogram, physical = self._batch_conditions(
            batch
        )
        targets = self.current_provider.sample_batch(
            underwater=underwater,
            reference=reference,
            histogram=histogram,
            physical=physical,
            batch_index=batch_index,
        )
        return targets, underwater, reference, histogram, physical

    # ------------------------------------------------------------------
    # 日志
    # ------------------------------------------------------------------
    _LOG_FIELDS = (
        "epoch",
        "phase",
        "stage",
        "batch_index",
        "loss_total",
        "loss_mse",
        "loss_perceptual",
        "loss_original",
        "loss_distill",
        "teacher_forwards",
        "denominator_min",
        "terminal_count",
        "adapter_lr",
        "gate_lr",
        "gate_enc1",
        "gate_enc2",
        "gate_enc3",
        "gate_enc4",
        "gate_mid",
        "gate_dec4",
        "gate_dec3",
        "gate_dec2",
        "gate_dec1",
        "inner_adapter_grad_norm",
        "outer_adapter_grad_norm",
        "adapter_gradient_dot",
        "correction_coefficient",
        "direct_arch_grad_norm",
        "hypergradient_norm",
        "gate_grad_norm_before_clip",
        "gate_grad_norm_after_clip",
        "adapter_grad_norm_before_clip",
        "adapter_grad_norm_after_clip",
    )

    def _csv_open(self, name: str) -> None:
        self._csv_close()
        self._csv_fields = list(self._LOG_FIELDS)
        path = self.log_dir / name
        new_file = not path.exists()
        self._csv_handle = path.open("a", encoding="utf-8", newline="")
        self._csv_writer = csv.DictWriter(
            self._csv_handle, fieldnames=self._csv_fields
        )
        if new_file:
            self._csv_writer.writeheader()

    def _csv_close(self) -> None:
        if self._csv_handle is not None:
            self._csv_handle.close()
            self._csv_handle = None
            self._csv_writer = None

    @staticmethod
    def _float(value: Any) -> float:
        if isinstance(value, Tensor):
            return float(value.detach().item())
        return float(value)

    def _log_batch(
        self,
        epoch: int,
        phase: str,
        stage: str,
        batch_index: int,
        losses: Mapping[str, Any],
        targets: Mapping[str, Tensor],
        *,
        adapter_lr: float,
        gate_lr: float,
        extra: Optional[Mapping[str, float]] = None,
    ) -> None:
        if self._csv_writer is None:
            raise RuntimeError("CSV 日志未打开。")
        row: Dict[str, Any] = {
            "epoch": epoch,
            "phase": phase,
            "stage": stage,
            "batch_index": batch_index,
            "loss_total": self._float(losses.get("loss_total")),
            "loss_mse": self._float(losses.get("loss_mse")),
            "loss_perceptual": self._float(
                losses.get("loss_perceptual")
            ),
            "loss_original": self._float(losses.get("loss_original")),
            "loss_distill": self._float(losses.get("loss_distill")),
            "teacher_forwards": int(targets["teacher_forwards_total"]),
            "denominator_min": float(
                targets["denominator"].abs().min().item()
            ),
            "terminal_count": int(targets["terminal_mask"].sum().item()),
            "adapter_lr": adapter_lr,
            "gate_lr": gate_lr,
        }
        for name, value in self.student.gates.export_values().items():
            row[f"gate_{name}"] = value
        if extra:
            row.update(
                {
                    key: value
                    for key, value in extra.items()
                    if isinstance(value, (int, float))
                }
            )
        self._csv_writer.writerow(
            {key: row.get(key, "") for key in self._csv_fields}
        )

    @staticmethod
    def _make_loader(
        dataset: SeaDiffDataset, epoch: int, seed: int, batch_size: int
    ) -> DataLoader:
        generator = torch.Generator()
        # 每轮从seed+epoch重建打乱次序；恢复入口按epoch边界继续，不支持批中途恢复。
        generator.manual_seed(seed + epoch)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            generator=generator,
        )

    # ------------------------------------------------------------------
    # 预热
    # ------------------------------------------------------------------
    def run_warmup(self, start_epoch: int = 1) -> None:
        boundary = self.epoch_plan[0]
        self.current_boundary = boundary
        self.current_provider = DistillationTargetProvider(
            runner=self.e3_runner,
            teacher_grid=self.grids[boundary.teacher_steps],
            student_grid=self.grids[boundary.student_steps],
            seed=self.seed,
        )
        self.student.train()
        self.student.set_gate_override(WARMUP_GATE_VALUE)
        self.student.gates.logits.requires_grad_(False)

        warmup_cfg = self.config["training"]["warmup"]
        optimizer = torch.optim.AdamW(
            list(self.student.adapter_parameters()),
            lr=float(warmup_cfg["adapter_lr"]),
            weight_decay=float(warmup_cfg["adapter_weight_decay"]),
        )
        self._load_optimizer_state(optimizer, "adapter")
        self._csv_open("search.csv")
        self._emit(
            "warmup_start",
            {
                "epochs": [boundary.first_epoch, boundary.last_epoch],
                "resume_from_epoch": start_epoch,
                "gate_override": WARMUP_GATE_VALUE,
                "adapter_lr": warmup_cfg["adapter_lr"],
                "teacher_steps": boundary.teacher_steps,
                "student_steps": boundary.student_steps,
            },
        )

        for epoch in range(start_epoch, boundary.last_epoch + 1):
            loader = self._make_loader(
                self.inner_dataset, epoch, self.seed, self.batch_size
            )
            for batch_index, batch in enumerate(loader):
                targets, underwater, reference, histogram, physical = (
                    self._compute_targets(batch, self.global_batch_index)
                )
                packed = self._pack_student_input(
                    targets["x_t"],
                    underwater,
                    histogram,
                    physical,
                    targets["timestep_start"],
                )
                prediction = self.student(
                    packed, targets["timestep_start"]
                )
                losses = self.loss_module(
                    prediction, reference, targets["teacher_target"]
                )
                optimizer.zero_grad(set_to_none=True)
                losses["loss_total"].backward()
                optimizer.step()
                self._log_batch(
                    epoch,
                    "warmup",
                    boundary.name,
                    batch_index,
                    losses,
                    targets,
                    adapter_lr=float(optimizer.param_groups[0]["lr"]),
                    gate_lr=0.0,
                )
                self.global_batch_index += 1

            self._emit(
                "warmup_epoch_done",
                {"epoch": epoch, "total_epochs": boundary.last_epoch},
            )
            self._save_rolling_checkpoint(
                epoch=epoch,
                phase="warmup",
                stage=boundary.name,
                optimizers={"adapter": optimizer},
            )

        self.student.set_gate_override(None)
        self.student.gates.logits.requires_grad_(True)
        # 预热只更新学生；此处不晋升教师，下一阶段S1仍用原始E3的1000点网格。
        self._emit("warmup_finished", {"epoch": boundary.last_epoch})
        self._save_boundary_checkpoint("warmup", optimizer=optimizer)

    # ------------------------------------------------------------------
    # 双层搜索
    # ------------------------------------------------------------------
    def _provider_for_boundary(
        self, boundary: StageBoundary
    ) -> DistillationTargetProvider:
        if boundary.name == "S1":
            runner = self.e3_runner
        else:
            # S2起使用上一阶段完成后的学生快照，保持该阶段目标函数不变。
            snapshot = self.snapshots[f"stage_{boundary.teacher_steps}"]
            snapshot.verify_immutable(snapshot.snapshot_sha256)
            runner = snapshot.runner
        return DistillationTargetProvider(
            runner=runner,
            teacher_grid=self.grids[boundary.teacher_steps],
            student_grid=self.grids[boundary.student_steps],
            seed=self.seed,
        )

    def run_search(self, start_epoch: int = 6) -> None:
        bilevel_cfg = self.config["training"]["bilevel"]
        # 搜索入口新建优化器；仅显式恢复断点时载入moment状态，普通预热结束不沿用。
        adapter_optimizer = torch.optim.AdamW(
            list(self.student.adapter_parameters()),
            lr=float(bilevel_cfg["adapter_lr"]),
            weight_decay=float(bilevel_cfg["adapter_weight_decay"]),
        )
        gate_optimizer = torch.optim.Adam(
            list(self.student.architecture_parameters()),
            lr=float(bilevel_cfg["gate_lr"]),
            weight_decay=float(bilevel_cfg["gate_weight_decay"]),
        )
        self._load_optimizer_state(adapter_optimizer, "adapter")
        self._load_optimizer_state(gate_optimizer, "gate")
        bilevel = BilevelOptimizer(
            self.student,
            adapter_optimizer,
            gate_optimizer,
            loss_module=self.loss_module,
            epsilon=float(bilevel_cfg.get("epsilon", 1e-8)),
            grad_clip=float(bilevel_cfg.get("grad_clip", 1.0)),
        )
        self.student.train()
        if self._csv_handle is None:
            self._csv_open("search.csv")

        dev_iterator = None
        for boundary in self.epoch_plan[1:]:
            # 恢复时跳过已经完成的阶段，避免用后续学生状态覆盖旧快照。
            if boundary.last_epoch < start_epoch:
                continue
            self.current_boundary = boundary
            self.current_provider = self._provider_for_boundary(boundary)
            self._emit(
                "search_stage_start",
                {
                    "stage": boundary.name,
                    "epochs": [
                        boundary.first_epoch,
                        boundary.last_epoch,
                    ],
                    "teacher_steps": boundary.teacher_steps,
                    "student_steps": boundary.student_steps,
                    "teacher": self.current_provider.teacher_identity(),
                },
            )

            for epoch in range(
                max(start_epoch, boundary.first_epoch),
                boundary.last_epoch + 1,
            ):
                loader = self._make_loader(
                    self.inner_dataset, epoch, self.seed, self.batch_size
                )
                for batch_index, batch in enumerate(loader):
                    # batch索引从0开始，所以每轮第1、5、9…批更新门控。
                    update_gates = (
                        batch_index % self.outer_every_inner == 0
                    )
                    (
                        targets,
                        underwater,
                        reference,
                        histogram,
                        physical,
                    ) = self._compute_targets(
                        batch, self.global_batch_index
                    )
                    packed = self._pack_student_input(
                        targets["x_t"],
                        underwater,
                        histogram,
                        physical,
                        targets["timestep_start"],
                    )

                    outer_args = {}
                    if update_gates:
                        if dev_iterator is None:
                            dev_iterator = iter(
                                self._make_loader(
                                    self.dev_dataset,
                                    epoch,
                                    self.seed,
                                    self.batch_size,
                                )
                            )
                        try:
                            dev_batch = next(dev_iterator)
                        except StopIteration:
                            dev_iterator = iter(
                                self._make_loader(
                                    self.dev_dataset,
                                    epoch,
                                    self.seed,
                                    self.batch_size,
                                )
                            )
                            dev_batch = next(dev_iterator)
                        (
                            outer_targets,
                            outer_underwater,
                            outer_reference,
                            outer_histogram,
                            outer_physical,
                        ) = self._compute_targets(
                            # outer用独立计数，不直接共享inner的global_batch_index。
                            dev_batch, self.dev_batch_counter
                        )
                        self.dev_batch_counter += 1
                        outer_packed = self._pack_student_input(
                            outer_targets["x_t"],
                            outer_underwater,
                            outer_histogram,
                            outer_physical,
                            outer_targets["timestep_start"],
                        )
                        outer_args = {
                            "outer_packed_input": outer_packed,
                            "outer_timesteps": outer_targets[
                                "timestep_start"
                            ],
                            "outer_reference": outer_reference,
                            "outer_teacher_target": outer_targets[
                                "teacher_target"
                            ],
                        }

                    logs = bilevel.step(
                        inner_packed_input=packed,
                        inner_timesteps=targets["timestep_start"],
                        inner_reference=reference,
                        inner_teacher_target=targets["teacher_target"],
                        update_gates=update_gates,
                        **outer_args,
                    )
                    self._log_batch(
                        epoch,
                        "search",
                        boundary.name,
                        batch_index,
                        {
                            # total是门控更新后/Adapter更新前；其余分项来自门控更新前。
                            "loss_total": logs[
                                "inner_loss_total_after_gate"
                            ],
                            "loss_mse": logs["inner_loss_mse_before"],
                            "loss_perceptual": logs[
                                "inner_loss_perceptual_before"
                            ],
                            "loss_original": logs[
                                "inner_loss_original_before"
                            ],
                            "loss_distill": logs[
                                "inner_loss_distill_before"
                            ],
                        },
                        targets,
                        adapter_lr=logs["adapter_lr"],
                        gate_lr=logs["architecture_lr"],
                        extra=logs,
                    )
                    self.global_batch_index += 1

                self._emit(
                    "search_epoch_done",
                    {"epoch": epoch, "stage": boundary.name},
                )
                self._save_rolling_checkpoint(
                    epoch=epoch,
                    phase="search",
                    stage=boundary.name,
                    optimizers={
                        "adapter": adapter_optimizer,
                        "gate": gate_optimizer,
                    },
                )

            # 当前学生冻结为下一阶段教师（嵌套网格要求步数匹配）。
            # 只有S1..S9会晋升；S9产出的stage_2同时供S10和最后40轮微调使用。
            # S10结束直接硬选择，不创建stage_1教师。
            index = self.epoch_plan.index(boundary)
            if index + 1 < len(self.epoch_plan):
                next_boundary = self.epoch_plan[index + 1]
                if boundary.student_steps == next_boundary.teacher_steps:
                    snapshot = promote_stage_teacher(
                        source=self.student,
                        stage_name=f"stage_{boundary.student_steps}",
                        teacher_grid=self.grids[
                            boundary.student_steps
                        ],
                        backbone_identity=self.backbone_identity,
                        state_path=self.checkpoint_dir
                        / f"snapshot_stage_{boundary.student_steps}.pt",
                    )
                    self.snapshots[
                        f"stage_{boundary.student_steps}"
                    ] = snapshot
                    self._emit(
                        "stage_teacher_promoted",
                        {
                            **snapshot.identity(),
                            "state_file": str(
                                self.checkpoint_dir
                                / f"snapshot_stage_"
                                f"{boundary.student_steps}.pt"
                            ),
                        },
                    )
                    self._save_boundary_checkpoint(
                        f"stage_{boundary.student_steps}",
                        optimizer=adapter_optimizer,
                        gate_optimizer=gate_optimizer,
                    )

        self._select_top3()

    def _select_top3(self) -> None:
        gates = self.student.gates.export_values()
        ordered = self.student.site_names()
        values = [gates[name] for name in ordered]
        distinctness = {
            "gate_values": gates,
            "max": max(values),
            "min": min(values),
            "spread": max(values) - min(values),
            "near_half_count": sum(
                1 for value in values if abs(value - 0.5) < 0.05
            ),
            "indistinct": (
                max(values) - min(values) < 0.1
                or all(abs(value - 0.5) < 0.05 for value in values)
            ),
        }
        self._emit("epoch60_gates_recorded", distinctness)
        if distinctness["indistinct"]:
            # 辨识度不足只记警告，仍执行预定Top-3规则；没有按测试指标另选位置。
            self._emit(
                "gate_indistinctness_warning",
                {
                    "message": (
                        "门控辨识度不足：spread < 0.1 或全部接近 0.5；"
                        "Top-3 排序可能不稳定。"
                    ),
                    **distinctness,
                },
            )

        selected = select_top_k(ordered, values, TOP_K)
        selected_values = {name: gates[name] for name in selected}
        self._emit(
            "top3_selected",
            {
                "selected": list(selected),
                "gates": selected_values,
                "ranking": sorted(
                    gates.items(), key=lambda item: -item[1]
                ),
            },
        )

        # 先保存九分支软门控模型，再固定门控保存 Top-3 模型。
        # 固定值保留实际sigmoid幅度，而非将三个门控改成1；落选分支停止执行。
        self._save_boundary_checkpoint("search_epoch60_full")
        self.student.configure_hard_selection(
            selected, selected_values
        )
        self.student.freeze_unselected_adapters()
        self._save_boundary_checkpoint("selected_top3")

        selection_path = self.output_dir / "selection_details.json"
        with selection_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema_version": 1,
                    "epoch": 60,
                    "gate_values": gates,
                    "selected_top3": list(selected),
                    "selected_gate_values": selected_values,
                    "distinctness": distinctness,
                    "candidate_order": list(ordered),
                },
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )

    # ------------------------------------------------------------------
    # 固定选择微调
    # ------------------------------------------------------------------
    def run_finetune(self, start_epoch: int = 61) -> None:
        if not self.student.is_hard_selected():
            raise RuntimeError("微调前必须先完成 Top-3 硬选择。")
        finetune_cfg = self.config["training"]["finetune"]
        if int(finetune_cfg.get("teacher_steps", 2)) != 2:
            raise ValueError("微调教师必须为 2 步阶段快照。")
        if float(finetune_cfg.get("gate_lr", 0.0)) != 0.0:
            raise ValueError("微调阶段门控学习率必须为 0。")

        boundary = StageBoundary(
            name="finetune",
            teacher_steps=2,
            student_steps=1,
            first_epoch=61,
            last_epoch=60 + self.finetune_epochs,
        )
        self.current_boundary = boundary
        self.student.train()

        # 继续使用S9结束时的完整九分支2步快照，不把Top-3学生自身作为教师。
        snapshot = self.snapshots["stage_2"]
        snapshot.verify_immutable(snapshot.snapshot_sha256)
        self.current_provider = DistillationTargetProvider(
            runner=snapshot.runner,
            teacher_grid=self.grids[2],
            student_grid=self.grids[1],
            seed=self.seed,
        )

        parameters = [
            parameter
            for parameter in self.student.adapter_parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            raise RuntimeError("没有可训练的 Adapter 参数。")
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(finetune_cfg["adapter_lr"]),
            weight_decay=float(finetune_cfg["adapter_weight_decay"]),
        )
        self._load_optimizer_state(optimizer, "adapter")
        self._csv_open("finetune.csv")
        self._emit(
            "finetune_start",
            {
                "epochs": [boundary.first_epoch, boundary.last_epoch],
                "resume_from_epoch": start_epoch,
                "adapter_lr": finetune_cfg["adapter_lr"],
                "teacher": self.current_provider.teacher_identity(),
                "trainable_adapters": self.student.active_site_names(),
                "report_model": f"epoch-{boundary.last_epoch}",
            },
        )

        for epoch in range(start_epoch, boundary.last_epoch + 1):
            loader = self._make_loader(
                self.full_dataset, epoch, self.seed, self.batch_size
            )
            # 合并600图微调仍用GT加噪和教师目标；部署不执行训练循环或读取reference。
            for batch_index, batch in enumerate(loader):
                targets, underwater, reference, histogram, physical = (
                    self._compute_targets(batch, self.global_batch_index)
                )
                packed = self._pack_student_input(
                    targets["x_t"],
                    underwater,
                    histogram,
                    physical,
                    targets["timestep_start"],
                )
                prediction = self.student(
                    packed, targets["timestep_start"]
                )
                losses = self.loss_module(
                    prediction, reference, targets["teacher_target"]
                )
                optimizer.zero_grad(set_to_none=True)
                losses["loss_total"].backward()
                optimizer.step()
                self._log_batch(
                    epoch,
                    "finetune",
                    boundary.name,
                    batch_index,
                    losses,
                    targets,
                    adapter_lr=float(optimizer.param_groups[0]["lr"]),
                    gate_lr=0.0,
                )
                self.global_batch_index += 1

            self._emit(
                "finetune_epoch_done",
                {"epoch": epoch, "total_epochs": boundary.last_epoch},
            )
            self._save_rolling_checkpoint(
                epoch=epoch,
                phase="finetune",
                stage=boundary.name,
                optimizers={"adapter": optimizer},
            )

        # 报告模型固定为预定第100轮，不依据测试分数选best或提前停止。
        self._save_boundary_checkpoint(
            "finetune_epoch100", optimizer=optimizer, final=True
        )
        self._emit(
            "finetune_finished",
            {
                "report_model_epoch": boundary.last_epoch,
                "checkpoint": str(
                    self.checkpoint_dir / "finetune_epoch100.pt"
                ),
            },
        )

    # ------------------------------------------------------------------
    # checkpoint
    # ------------------------------------------------------------------
    def _student_payload(self) -> Dict[str, Any]:
        return {
            "state_dict": self.student.state_dict(),
            "gates_extra_state": self.student.gates.get_extra_state(),
            "hard_selected": self.student.is_hard_selected(),
            "selected_sites": (
                list(self.student.active_site_names())
                if self.student.is_hard_selected()
                else None
            ),
            "fixed_gate_values": self.student.fixed_gate_values.detach()
            .cpu()
            .clone()
            .tolist(),
            "gate_override": self.student.gate_override,
        }

    def _checkpoint_payload(
        self,
        *,
        name: str,
        epoch: Optional[int],
        phase: str,
        stage: str,
        optimizers: Mapping[str, torch.optim.Optimizer],
        final: bool,
    ) -> Dict[str, Any]:
        return {
            "kind": CHECKPOINT_KIND,
            "schema_version": SCHEMA_VERSION,
            "checkpoint_name": name,
            "student": self._student_payload(),
            "optimizers": {
                key: optimizer.state_dict()
                for key, optimizer in optimizers.items()
            },
            "epoch": epoch,
            "phase": phase,
            "stage": stage,
            "snapshots": {
                key: snapshot.identity()
                for key, snapshot in self.snapshots.items()
            },
            "snapshot_state_files": {
                key: str(self.checkpoint_dir / f"snapshot_{key}.pt")
                for key in self.snapshots
            },
            "teacher_identity": (
                self.current_provider.teacher_identity()
                if self.current_provider is not None
                else None
            ),
            "config_sha256": _sha256_json(
                _public_config(self.config)
            ),
            "seed": self.seed,
            "rng_state": capture_rng_state(),
            # 这里只保存inner批计数；dev_batch_counter及dev迭代器当前位置未保存。
            # 因此epoch恢复具备状态校验，但不能据此声称外层随机流逐批完全复现。
            "global_batch_index": self.global_batch_index,
            "final": final,
        }

    def _save_boundary_checkpoint(
        self,
        name: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        gate_optimizer: Optional[torch.optim.Optimizer] = None,
        final: bool = False,
    ) -> None:
        optimizers = {
            key: value
            for key, value in (
                ("adapter", optimizer),
                ("gate", gate_optimizer),
            )
            if value is not None
        }
        payload = self._checkpoint_payload(
            name=name,
            epoch=(
                self.current_boundary.last_epoch
                if self.current_boundary is not None
                else None
            ),
            phase=(
                self.current_boundary.name
                if self.current_boundary is not None
                else name
            ),
            stage=name,
            optimizers=optimizers,
            final=final,
        )
        path = self.checkpoint_dir / f"{name}.pt"
        save_checkpoint(path, payload)
        self._emit(
            "checkpoint_saved",
            {
                "name": name,
                "path": str(path),
                "sha256": checkpoint_sha256(path),
            },
        )

    def _save_rolling_checkpoint(
        self,
        *,
        epoch: int,
        phase: str,
        stage: str,
        optimizers: Mapping[str, torch.optim.Optimizer],
    ) -> None:
        payload = self._checkpoint_payload(
            name="rolling",
            epoch=epoch,
            phase=phase,
            stage=stage,
            optimizers=optimizers,
            final=False,
        )
        save_checkpoint(self.checkpoint_dir / "rolling.pt", payload)

    def restore(self, checkpoint_path: Path) -> int:
        """恢复断点，返回下一个 epoch。"""
        payload = load_checkpoint(checkpoint_path)
        if payload["config_sha256"] != _sha256_json(
            _public_config(self.config)
        ):
            raise ValueError("断点配置与当前配置不一致，拒绝恢复。")

        saved = payload["student"]
        self.student.load_state_dict(saved["state_dict"], strict=True)
        self.student.gates.set_extra_state(saved["gates_extra_state"])

        if saved.get("hard_selected"):
            sites = saved["selected_sites"]
            values = {
                name: saved["fixed_gate_values"][
                    self.student.site_names().index(name)
                ]
                for name in sites
            }
            self.student.configure_hard_selection(sites, values)
            self.student.freeze_unselected_adapters()

        override = saved.get("gate_override")
        if override is not None:
            self.student.set_gate_override(override)

        state_files = payload.get("snapshot_state_files") or {}
        for name, identity in (payload.get("snapshots") or {}).items():
            state_file = Path(state_files.get(name, ""))
            snapshot = StageTeacherSnapshot.from_state_file(
                state_file,
                backbone_source=self.student,
                alpha_bar=self.alpha_bar,
            )
            expected = identity["snapshot_sha256"]
            if snapshot.snapshot_sha256 != expected:
                raise ValueError(
                    f"恢复的快照 {snapshot.stage_name} 与断点身份不一致。"
                )
            self.snapshots[name] = snapshot

        self.global_batch_index = int(
            payload.get("global_batch_index", 0)
        )
        self._restored_optimizer_states = dict(
            payload.get("optimizers") or {}
        )
        if payload.get("rng_state") is not None:
            restore_rng_state(payload["rng_state"])
        epoch = int(payload["epoch"])
        self._emit(
            "checkpoint_restored",
            {
                "path": str(checkpoint_path),
                "epoch": epoch,
                "stage": payload.get("stage"),
                "phase": payload.get("phase"),
                "snapshots_restored": list(self.snapshots),
            },
        )
        return epoch + 1

    def _load_optimizer_state(
        self, optimizer: torch.optim.Optimizer, key: str
    ) -> None:
        saved = self._restored_optimizer_states.get(key)
        if saved is not None:
            optimizer.load_state_dict(saved)
            self._emit("optimizer_state_restored", {"key": key})

    # ------------------------------------------------------------------
    # 编排
    # ------------------------------------------------------------------
    def close_logging(self) -> None:
        """关闭当前 CSV；搜索与微调分脚本运行时由入口调用。"""
        self._csv_close()

    def run(self, resume_from: Optional[str] = None) -> None:
        """完整 100-epoch 流程；输入断点则从下一 epoch 继续。"""
        start_epoch = 1
        if resume_from is not None:
            start_epoch = self.restore(Path(resume_from))

        if start_epoch <= self.warmup_epochs:
            self.run_warmup(start_epoch=start_epoch)
            start_epoch = self.warmup_epochs + 1
        if start_epoch <= 60:
            self.run_search(start_epoch=start_epoch)
        self.run_finetune(start_epoch=max(61, start_epoch))

        self._csv_close()
        self._emit(
            "run_completed",
            {
                "total_epochs": 60 + self.finetune_epochs,
                "report_model": "epoch-100 微调 checkpoint",
            },
        )

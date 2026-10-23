"""阶段教师快照。

搜索阶段每次降网格后，把当前学生整体冻结为下一阶段的教师：

- 深拷贝骨干、9 个 Adapter 与门控；
- 门控以当前 sigmoid 值固化为常数；
- 快照 eval + requires_grad_(False)，作为 TeacherChainRunner 的 denoise_fn；
- 状态单独原子保存，训练 checkpoint 只存引用和 SHA256。
"""

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch import Tensor

from ..models.adapter_student import OneStepSeaDiffStudent
from .targets import (
    TeacherChainRunner,
    build_dynamic_physical_pack_fn,
)


def _state_sha256(module: torch.nn.Module) -> str:
    """state_dict 的确定性 SHA256。"""
    digest = hashlib.sha256()
    state = module.state_dict()
    for name in sorted(state.keys()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        value = state[name]
        if isinstance(value, torch.Tensor):
            digest.update(str(tuple(value.shape)).encode("utf-8"))
            digest.update(b"\x00")
            digest.update(value.detach().cpu().numpy().tobytes())
        else:
            digest.update(
                json.dumps(
                    value,
                    sort_keys=True,
                    ensure_ascii=False,
                    default=str,
                ).encode("utf-8")
            )
        digest.update(b"\x00")
    return digest.hexdigest()


def _bitwise_equal(
    a: torch.nn.Module, b: torch.nn.Module, names: Sequence[str]
) -> List[str]:
    differences = []
    state_a = a.state_dict()
    state_b = b.state_dict()
    for name in names:
        if name not in state_a or name not in state_b:
            differences.append(f"{name}: 缺失")
            continue
        value_a = state_a[name]
        value_b = state_b[name]
        if isinstance(value_a, torch.Tensor) or isinstance(
            value_b, torch.Tensor
        ):
            if not (
                isinstance(value_a, torch.Tensor)
                and isinstance(value_b, torch.Tensor)
                and torch.equal(value_a, value_b)
            ):
                differences.append(f"{name}: 数值不一致")
        elif value_a != value_b:
            differences.append(f"{name}: 数值不一致")
    return differences


class StageTeacherSnapshot:
    """冻结学生快照，作为下一阶段的不可变链教师。"""

    def __init__(
        self,
        *,
        snapshot: OneStepSeaDiffStudent,
        gate_values: Dict[str, float],
        stage_name: str,
        teacher_grid: Sequence[int],
        backbone_identity: Dict[str, object],
        alpha_bar: Tensor | None = None,
    ):
        if not isinstance(snapshot, OneStepSeaDiffStudent):
            raise TypeError("snapshot 必须是 OneStepSeaDiffStudent。")

        self.stage_name = str(stage_name)
        self.teacher_grid = tuple(int(t) for t in teacher_grid)
        self.backbone_identity = dict(backbone_identity)
        self.gate_values = {
            name: float(value) for name, value in gate_values.items()
        }

        self.snapshot = snapshot
        self.snapshot_sha256 = _state_sha256(self.snapshot)

        self.runner = TeacherChainRunner(
            denoise_fn=self._denoise,
            make_packed=build_dynamic_physical_pack_fn(alpha_bar),
            identity={
                "kind": "stage_student_snapshot",
                "stage": self.stage_name,
                "snapshot_sha256": self.snapshot_sha256,
                "teacher_grid_size": len(self.teacher_grid),
                "gate_values": dict(self.gate_values),
            },
        )

    def _denoise(self, packed: Tensor, timesteps: Tensor) -> Tensor:
        return self.snapshot(packed, timesteps)

    def identity(self) -> Dict[str, object]:
        return {
            "kind": "stage_student_snapshot",
            "stage": self.stage_name,
            "snapshot_sha256": self.snapshot_sha256,
            "backbone_identity": dict(self.backbone_identity),
            "teacher_grid": list(self.teacher_grid),
            "gate_values": dict(self.gate_values),
        }

    def save_state(self, path: str | Path) -> str:
        """原子保存快照状态文件，返回文件 SHA256。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": "seadiff_stage_teacher_snapshot",
            "stage": self.stage_name,
            "teacher_grid": list(self.teacher_grid),
            "gate_values": dict(self.gate_values),
            "state_dict": {
                name: (
                    value.detach().cpu().clone()
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for name, value in self.snapshot.state_dict().items()
            },
            "snapshot_sha256": self.snapshot_sha256,
            "backbone_identity": dict(self.backbone_identity),
        }
        descriptor, temporary = tempfile.mkstemp(
            prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
        )
        os.close(descriptor)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

        digest = hashlib.sha256()
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def from_state_file(
        cls,
        path: str | Path,
        *,
        backbone_source: OneStepSeaDiffStudent,
        alpha_bar: Tensor | None = None,
    ) -> "StageTeacherSnapshot":
        """从状态文件重建快照。"""
        source = Path(path).expanduser().resolve(strict=True)
        payload = torch.load(source, map_location="cpu", weights_only=True)
        if (
            not isinstance(payload, dict)
            or payload.get("kind") != "seadiff_stage_teacher_snapshot"
        ):
            raise ValueError(f"不是阶段教师快照文件：{source}")

        if not isinstance(backbone_source, OneStepSeaDiffStudent):
            raise TypeError("backbone_source 必须是 OneStepSeaDiffStudent。")

        source_backbone = backbone_source.backbone
        device = next(backbone_source.parameters()).device
        if next(source_backbone.parameters()).device.type == "cpu":
            cpu_backbone = source_backbone
        else:
            cpu_backbone = copy.deepcopy(source_backbone).cpu()

        clone = OneStepSeaDiffStudent(
            cpu_backbone,
            adapter_reduction=backbone_source.adapter_reduction,
            fixed_timestep=999,
        )
        clone.load_state_dict(payload["state_dict"], strict=True)
        gate_values = dict(payload["gate_values"])
        selected = tuple(
            name for name in clone.site_names() if name in gate_values
        )
        if not selected:
            raise ValueError(f"快照文件没有有效门控：{source}")
        clone.configure_hard_selection(selected, gate_values)
        clone.requires_grad_(False)
        clone.zero_grad(set_to_none=True)
        clone.eval()
        clone = clone.to(device)

        snapshot = cls(
            snapshot=clone,
            gate_values=payload["gate_values"],
            stage_name=payload["stage"],
            teacher_grid=payload["teacher_grid"],
            backbone_identity=payload.get("backbone_identity", {}),
            alpha_bar=alpha_bar,
        )
        expected = payload.get("snapshot_sha256")
        if snapshot.snapshot_sha256 != expected:
            raise ValueError(
                f"恢复的快照 {snapshot.stage_name} 与文件记录 SHA 不一致。"
            )
        return snapshot

    def verify_immutable(self, expected_sha256: str) -> None:
        """确认快照未被修改。"""
        current = _state_sha256(self.snapshot)
        if current != expected_sha256:
            raise RuntimeError(
                f"阶段教师快照 {self.stage_name} 已被修改。"
            )

    def verify_consistency_with_source(
        self,
        source: OneStepSeaDiffStudent,
    ) -> List[str]:
        """快照刚创建时，除 fixed_gate_values 外应与源学生逐位一致。"""
        differences = _bitwise_equal(
            self.snapshot,
            source,
            [
                name
                for name, _ in source.state_dict().items()
                if not name.startswith("backbone.")
                and name != "fixed_gate_values"
            ],
        )
        return differences


def promote_stage_teacher(
    *,
    source: OneStepSeaDiffStudent,
    stage_name: str,
    teacher_grid: Sequence[int],
    backbone_identity: Dict[str, object],
    state_path: str | Path | None = None,
) -> StageTeacherSnapshot:
    """冻结当前学生，核验一致性，并可选落盘快照。"""
    if source.is_hard_selected():
        raise ValueError("阶段快照必须在 Top-K 硬选择之前创建。")

    gate_values = source.gates.export_values()
    clone = copy.deepcopy(source)
    clone.configure_hard_selection(clone.site_names(), gate_values)
    clone.requires_grad_(False)
    clone.zero_grad(set_to_none=True)
    clone.eval()

    snapshot = StageTeacherSnapshot(
        snapshot=clone,
        gate_values=gate_values,
        stage_name=stage_name,
        teacher_grid=teacher_grid,
        backbone_identity=backbone_identity,
    )

    differences = snapshot.verify_consistency_with_source(source)
    if differences:
        raise AssertionError(
            f"阶段教师快照与源学生不一致：{differences}"
        )

    if state_path is not None:
        snapshot.save_state(state_path)
    return snapshot

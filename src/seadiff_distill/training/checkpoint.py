"""完整训练检查点：原子写入、身份校验与恢复。

首版支持 epoch 边界恢复：
- 保存 student（含 SiteGates 的 extra_state 与固定门控 buffer）；
- 两个优化器、可选调度器；
- epoch / global_step / 阶段 / Top-k 选择；
- 配置、教师协议、数据身份；
- Python / NumPy / Torch CPU / CUDA RNG；
- 两个 DataLoader 生成器状态与 meta 迭代位置。
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping

import torch


CHECKPOINT_KIND = "seadiff_distill_checkpoint"
SCHEMA_VERSION = 1


def save_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> None:
    """原子写入检查点，避免中断留下半个文件。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if payload.get("kind") != CHECKPOINT_KIND:
        raise ValueError("checkpoint kind 不正确。")

    descriptor, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path: str | Path, *, map_location="cpu") -> Dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    payload = torch.load(source, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("检查点必须是字典。")
    if payload.get("kind") != CHECKPOINT_KIND:
        raise ValueError(f"不是本项目检查点：{source}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"检查点版本不支持：{payload.get('schema_version')}"
        )
    return payload


def checkpoint_sha256(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_jsonable(value: Any):
    """把配置中的非 JSON 类型转成可读文本，供快照写入。"""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            to_jsonable(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, target)

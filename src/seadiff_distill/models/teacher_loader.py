"""加载与当前 SeaDiff 骨干兼容的教师权重。"""

import hashlib
import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import torch
from torch import Tensor

from .candidate_sites import SEADIFF_UPSTREAM_COMMIT
from .seadiff_unet import DocDiff


def _load_raw_state_dict(
    path: Union[str, Path],
    expected_sha256: str,
) -> Tuple[Mapping, Dict[str, str]]:
    """读取官方格式的原始 state_dict，并核对文件身份。"""
    if not isinstance(expected_sha256, str):
        raise TypeError("expected_sha256 必须是字符串。")

    expected_hash = expected_sha256.strip().lower()
    if (
        len(expected_hash) != 64
        or any(char not in "0123456789abcdef" for char in expected_hash)
    ):
        raise ValueError("expected_sha256 必须是 64 位十六进制字符串。")

    resolved_path = Path(path).expanduser().resolve(strict=True)
    if not resolved_path.is_file():
        raise ValueError(f"权重路径不是文件：{resolved_path}")

    # 从同一份字节计算哈希并加载，确保记录对应实际加载内容。
    payload = resolved_path.read_bytes()
    actual_hash = hashlib.sha256(payload).hexdigest()

    if actual_hash != expected_hash:
        raise ValueError(
            f"权重文件 SHA256 不匹配：{resolved_path}\n"
            f"期望：{expected_hash}\n"
            f"实际：{actual_hash}"
        )

    state = torch.load(
        io.BytesIO(payload),
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(state, Mapping) or not state:
        raise TypeError(
            f"{resolved_path} 必须包含非空的原始 state_dict。"
        )

    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise TypeError(
                "权重必须是参数名到 Tensor 的原始映射；"
                "当前不接受外层 checkpoint 字典。"
            )

        if value.layout != torch.strided:
            raise TypeError(f"参数 {name} 必须是普通稠密张量。")

        # 在 load_state_dict 前检查，避免加载时自动转换精度。
        if value.dtype != torch.float32:
            raise TypeError(
                f"参数 {name} 的精度是 {value.dtype}；"
                "当前加载器只接受 FP32 权重。"
            )

        if not torch.isfinite(value).all().item():
            raise ValueError(f"参数 {name} 包含 NaN 或 Inf。")

    metadata = {
        "path": str(resolved_path),
        "sha256": actual_hash,
    }
    return state, metadata


def load_seadiff_teacher(
    *,
    denoiser_path: Union[str, Path],
    beta_predictor_path: Union[str, Path],
    denoiser_sha256: str,
    beta_predictor_sha256: str,
    beta_use_norm: bool = False,
) -> Tuple[DocDiff, Dict[str, Any]]:
    """
    严格加载教师的两个子网络，返回 CPU FP32 模型及来源记录。

    返回模型已冻结并设为 eval 模式。
    此函数不执行教师采样，也不验证教师增强质量。
    """
    denoiser_state, denoiser_metadata = _load_raw_state_dict(
        denoiser_path,
        denoiser_sha256,
    )
    beta_state, beta_metadata = _load_raw_state_dict(
        beta_predictor_path,
        beta_predictor_sha256,
    )

    # 参数与官方构造调用一致。
    # 当前 DocDiff 内部实际构建 12→3 去噪网络和 3→3 Beta 网络。
    teacher = DocDiff(
        input_channels=6,
        output_channels=3,
        n_channels=32,
        ch_mults=(1, 2, 3, 4),
        n_blocks=1,
        beta_use_norm=beta_use_norm,
    ).cpu().float()

    teacher.denoiser.load_state_dict(denoiser_state, strict=True)
    teacher.beta_predictor.load_state_dict(beta_state, strict=True)

    # 这是“作为教师使用”的冻结加载入口；400k 联合训练/E3 微调训练器
    # 会按各自训练范围设置参数梯度，不能由此推断所有教师训练均冻结。
    teacher.requires_grad_(False)
    teacher.zero_grad(set_to_none=True)
    teacher.eval()

    metadata = {
        "upstream_commit": SEADIFF_UPSTREAM_COMMIT,
        "architecture": {
            "model_channels": 32,
            "channel_mult": [1, 2, 3, 4],
            "num_resblocks": 1,
            "denoiser_input_channels": 12,
            "denoiser_output_channels": 3,
            "beta_use_norm": bool(beta_use_norm),
        },
        "denoiser": denoiser_metadata,
        "beta_predictor": beta_metadata,
    }
    return teacher, metadata
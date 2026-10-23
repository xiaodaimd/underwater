"""训练与恢复所需的随机状态管理。

与旧项目 N0 的确定性协议保持一致：
- Python / NumPy / Torch CPU / CUDA 全部播种；
- cuDNN benchmark 关闭、deterministic 打开；
- 可选 torch.use_deterministic_algorithms(True)。
"""

import os
import random
from typing import Any, Dict

import numpy as np
import torch


def configure_determinism(seed: int, *, strict: bool = True) -> Dict[str, Any]:
    """设置确定性协议，返回实际启用的状态。"""
    if type(seed) is not int:
        raise TypeError("seed 必须是整数。")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    status: Dict[str, Any] = {
        "seed": seed,
        "strict_requested": bool(strict),
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "strict_enabled": False,
        "strict_error": None,
        "unsupported_op": None,
    }

    if strict:
        try:
            torch.use_deterministic_algorithms(True)
            status["strict_enabled"] = True
        except Exception as exc:  # noqa: BLE001 - 记录具体原因，不静默关闭
            status["strict_error"] = f"{type(exc).__name__}: {exc}"
            message = str(exc)
            marker = "does not have a deterministic implementation"
            if marker in message:
                status["unsupported_op"] = message.split(marker)[0].strip()

    return status


def capture_rng_state() -> Dict[str, Any]:
    """捕获四类 RNG 状态，供 epoch 边界恢复。"""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
        "torch_cuda_all": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state()
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any], *, map_location="cpu") -> None:
    """恢复 checkpoint 中的 RNG 状态。"""
    required = ("python", "numpy", "torch_cpu")
    for key in required:
        if key not in state:
            raise ValueError(f"RNG 状态缺少字段：{key}")

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())

    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        cuda_state = state["torch_cuda"]
        if not isinstance(cuda_state, torch.Tensor):
            cuda_state = torch.as_tensor(cuda_state, dtype=torch.uint8)
        torch.cuda.set_rng_state(cuda_state.cpu(), device="cuda")

    if torch.cuda.is_available() and state.get("torch_cuda_all") is not None:
        states = [
            item if isinstance(item, torch.Tensor)
            else torch.as_tensor(item, dtype=torch.uint8)
            for item in state["torch_cuda_all"]
        ]
        torch.cuda.set_rng_state_all([item.cpu() for item in states])

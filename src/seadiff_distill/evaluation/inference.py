"""确定性推理辅助：初始噪声与图像输出。

训练和评估使用同一套初始噪声规则：以 ``global_seed`` 与 ``sample_id``
生成 BLAKE2b 摘要，再作为 CPU ``torch.Generator`` 的种子。这样同一张图
在教师与学生的推理中始终从相同的 ``x_T`` 出发，便于复现与对比。
"""

import hashlib
from pathlib import Path
from typing import Sequence, Union

import torch
from torch import Tensor
from torchvision.utils import save_image


def derive_noise_seed(sample_id: str, global_seed: int) -> int:
    """由样本 ID 与全局种子得到 64 位确定性种子。"""
    digest = hashlib.blake2b(
        f"{int(global_seed)}:{sample_id}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little")


def derive_initial_noise(
    sample_id: str,
    *,
    global_seed: int,
    shape: Sequence[int],
    device: Union[str, torch.device],
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """生成单样本初始噪声，形状通常为 ``[3,H,W]``。

    先固定 CPU 生成器再搬到目标设备，避免 CUDA 随机流差异。
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(derive_noise_seed(sample_id, global_seed))
    noise = torch.randn(tuple(shape), generator=generator, dtype=dtype)
    return noise.to(device=device)


def save_prediction(image: Tensor, path: Union[str, Path]) -> None:
    """保存 ``[3,H,W]`` 或 ``[1,3,H,W]`` 的增强结果（clamp 到 [0,1]）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tensor = image.detach().cpu()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or tensor.shape[1] != 3:
        raise ValueError("输出图像必须是 [3,H,W] 或 [1,3,H,W]。")
    save_image(tensor.clamp(0.0, 1.0), target)

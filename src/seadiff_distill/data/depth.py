"""Depth Anything V1 Small 深度条件生成。

SeaDiff 论文使用第一代 Depth Anything Small（ViT-S）。本模块只接受官方
`depth_anything_vits14.pth`，并复用 ``third_party/SeaDiff/depth_anything``
中的 V1 ``DPT_DINOv2`` 实现；V2 权重与实现不得混用。

预处理复现官方 ``infer_image``：保持长宽比把短边缩放到 518、尺寸取 14 的
倍数、ImageNet 归一化、输出插值回原尺寸。深度只由水下图生成。
"""

import contextlib
import hashlib
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

V1_SMALL_SHA256 = (
    "59afe57cfd9f4284deaf5b753d954723d7136ae842fba3e068ba03537ca1e60e"
)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _constrain_to_multiple_of(x: float, multiple: int, minimum: int) -> int:
    """官方 Resize 的 multiple_of 约束：先取近邻，过小则向上取整。"""
    value = int(np.round(x / multiple) * multiple)
    if value < minimum:
        value = int(np.ceil(x / multiple) * multiple)
    return value


def official_resize_size(
    width: int, height: int, input_size: int = 518
) -> Tuple[int, int]:
    """返回官方 ``Resize(518, 518, lower_bound, multiple_of=14)`` 的宽高。"""
    scale_height = input_size / height
    scale_width = input_size / width
    if scale_width > scale_height:
        scale_height = scale_width
    else:
        scale_width = scale_height
    new_height = _constrain_to_multiple_of(
        scale_height * height, 14, input_size
    )
    new_width = _constrain_to_multiple_of(
        scale_width * width, 14, input_size
    )
    return new_width, new_height


@contextlib.contextmanager
def _sea_diff_import_context():
    """临时把 third_party/SeaDiff 加入路径，供 DPT 本地加载 DINOv2。"""
    import seadiff_distill

    project_root = Path(seadiff_distill.__file__).resolve().parents[2]
    sea_diff_root = project_root / "third_party" / "SeaDiff"
    if not (sea_diff_root / "depth_anything" / "dpt.py").is_file():
        raise FileNotFoundError(f"缺少 bundled depth_anything：{sea_diff_root}")

    previous = os.getcwd()
    sys.path.insert(0, str(sea_diff_root))
    try:
        os.chdir(sea_diff_root)
        yield sea_diff_root
    finally:
        os.chdir(previous)
        if str(sea_diff_root) in sys.path:
            sys.path.remove(str(sea_diff_root))


class DepthAnythingV1Small:
    """冻结的 Depth Anything V1 Small 深度估计器。"""

    version = "depth_anything_v1_small"
    input_size = 518

    def __init__(self, weights_path: str, *, device: str = "cuda"):
        import cv2

        self.cv2 = cv2
        self.device = torch.device(device)
        weights = Path(weights_path).expanduser().resolve(strict=True)
        digest = _sha256(weights)
        if digest != V1_SMALL_SHA256:
            raise ValueError(
                "Depth Anything V1 Small 权重 SHA256 不匹配："
                f"{digest} != {V1_SMALL_SHA256}"
            )
        self.weights_path = str(weights)
        self.weights_sha256 = digest

        with _sea_diff_import_context():
            from depth_anything.dpt import DPT_DINOv2

            model = DPT_DINOv2(
                encoder="vits",
                features=64,
                out_channels=[48, 96, 192, 384],
                localhub=True,
            )
        state = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        model.requires_grad_(False)
        self.model = model.to(self.device)

    @torch.no_grad()
    def infer(self, image_rgb: np.ndarray) -> np.ndarray:
        """输入 HWC uint8 RGB，返回原尺寸 FP32 深度（未归一化）。"""
        if (
            image_rgb.dtype != np.uint8
            or image_rgb.ndim != 3
            or image_rgb.shape[2] != 3
        ):
            raise ValueError("image_rgb 必须是 HWC uint8 RGB。")

        height, width = image_rgb.shape[:2]
        new_width, new_height = official_resize_size(
            width, height, self.input_size
        )
        resized = self.cv2.resize(
            image_rgb,
            (new_width, new_height),
            interpolation=self.cv2.INTER_CUBIC,
        ).astype(np.float32) / 255.0
        normalized = (resized - IMAGENET_MEAN) / IMAGENET_STD
        tensor = (
            torch.from_numpy(
                np.ascontiguousarray(np.transpose(normalized, (2, 0, 1)))
            )
            .unsqueeze(0)
            .to(self.device)
        )

        depth = self.model(tensor)
        if depth.ndim == 4:
            if depth.shape[1] != 1:
                raise RuntimeError(f"深度输出通道异常：{tuple(depth.shape)}")
            depth = depth[:, 0]
        elif depth.ndim != 3:
            raise RuntimeError(f"深度输出形状异常：{tuple(depth.shape)}")
        depth = F.interpolate(
            depth[:, None],
            (height, width),
            mode="bilinear",
            align_corners=True,
        )[0, 0]
        return depth.detach().cpu().float().numpy()

    @staticmethod
    def minmax_float01(depth: np.ndarray) -> np.ndarray:
        """逐图 min-max 到 [0,1]；常量图明确报错。"""
        minimum = float(depth.min())
        maximum = float(depth.max())
        span = maximum - minimum
        if not np.isfinite(span) or span <= 0:
            raise ValueError("深度图是常量或含非有限值，无法逐图 min-max。")
        return ((depth - minimum) / span).astype(np.float32)

    def describe(self) -> dict:
        return {
            "version": self.version,
            "implementation": (
                "bundled third_party/SeaDiff/depth_anything (V1 DPT_DINOv2)"
            ),
            "weights_path": self.weights_path,
            "weights_sha256": self.weights_sha256,
            "input_size": self.input_size,
            "resize": "lower_bound, multiple_of=14, cv2 INTER_CUBIC",
            "normalization": "ImageNet mean/std",
            "output": "original size, bilinear align_corners=True",
            "depth_normalization": "per-image min-max to [0,1] FP32",
        }

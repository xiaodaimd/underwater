"""SeaDiff 配对数据读取。

训练清单固定为五列：``sample_id,underwater,reference,depth,histogram``。
所有路径都是相对于 ``data_root`` 的 POSIX 路径。

- underwater/reference：RGB 图像，bilinear resize 到 ``image_size``；
- depth：``.npy``，FP32，``[H,W]`` 或 ``[1,H,W]``，已逐图归一化到 [0,1]；
- histogram：``.npy``，FP32，``[3,H,W]``，总和为 1。

``SeaDiffDataset`` 用于训练/评测，整份数据读入 CPU 内存以减少磁盘抖动；
``InferenceDataset`` 用于单图或批量推理，按需读取且 reference 可选。
"""

import csv
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, Resize, ToTensor

FIELDS = ("sample_id", "underwater", "reference", "depth", "histogram")
INFERENCE_FIELDS = ("sample_id", "underwater", "depth", "histogram")


@dataclass(frozen=True)
class SeaDiffRecord:
    """清单中的一条样本；reference 在纯推理清单中可以为空。"""

    sample_id: str
    underwater: Path
    reference: Optional[Path]
    depth: Path
    histogram: Path


def _resolve(root: Path, value: str) -> Path:
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "\\" in value
        or ":" in value
    ):
        raise ValueError(f"路径必须是 data_root 下的相对 POSIX 路径：{value}")
    resolved = root.joinpath(*relative.parts).resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"不是文件：{resolved}")
    return resolved


def read_manifest(
    manifest_path: Union[str, Path],
    data_root: Union[str, Path],
    *,
    require_reference: bool = True,
) -> Tuple[SeaDiffRecord, ...]:
    """读取并校验清单；训练清单必须含 reference，推理清单可省略。"""
    manifest = Path(manifest_path).expanduser().resolve(strict=True)
    root = Path(data_root).expanduser().resolve(strict=True)
    required = FIELDS if require_reference else INFERENCE_FIELDS
    records = []
    seen_ids = set()
    seen_inputs = set()
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        if not set(required) <= set(fields):
            missing = set(required) - set(fields)
            raise ValueError(f"清单缺少列：{sorted(missing)}")
        for row in reader:
            values = {key: (row.get(key) or "").strip() for key in fields}
            if any(not values[key] for key in required):
                raise ValueError(f"清单第 {reader.line_num} 行有空字段。")
            sample_id = values["sample_id"]
            if sample_id in seen_ids:
                raise ValueError(f"重复 sample_id：{sample_id}")
            paths = {
                key: _resolve(root, values[key])
                for key in ("underwater", "depth", "histogram")
            }
            reference = (
                _resolve(root, values["reference"])
                if values.get("reference")
                else None
            )
            if require_reference and reference is None:
                raise ValueError(f"训练样本缺少 reference：{sample_id}")
            if paths["underwater"] in seen_inputs:
                raise ValueError(f"重复输入文件：{paths['underwater']}")
            seen_ids.add(sample_id)
            seen_inputs.add(paths["underwater"])
            records.append(
                SeaDiffRecord(
                    sample_id=sample_id,
                    reference=reference,
                    **paths,
                )
            )
    if not records:
        raise ValueError("清单为空。")
    return tuple(records)


def _read_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        if image.mode in ("I", "F") or image.mode.startswith("I;16"):
            raise ValueError(f"{path} 使用高精度模式 {image.mode}")
        return image.convert("RGB")


def _load_condition_arrays(
    record: SeaDiffRecord, image_size: Sequence[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """读取深度与直方图，并检查 dtype、形状、范围和 FP32 约束。"""
    # 深度/直方图来自预处理缓存；这里仅校验并读取，不重新估计或缩放条件。
    # 缓存空间尺寸必须与 resize 后的 RGB 一致，避免物理图中的像素错位。
    depth_array = np.load(record.depth, allow_pickle=False)
    if depth_array.ndim == 3 and depth_array.shape[0] == 1:
        depth_array = depth_array[0]
    if depth_array.shape != tuple(image_size):
        raise ValueError(
            f"深度形状 {depth_array.shape} != {tuple(image_size)}：{record.depth}"
        )
    if depth_array.dtype != np.float32:
        raise ValueError(f"深度不是 FP32：{record.depth}")
    if (
        not np.isfinite(depth_array).all()
        or float(depth_array.min()) < -1e-6
        or float(depth_array.max()) > 1.0 + 1e-6
    ):
        raise ValueError(f"深度不在 [0,1] 或含非有限值：{record.depth}")

    hist_array = np.load(record.histogram, allow_pickle=False)
    if hist_array.ndim == 4 and hist_array.shape[0] == 1:
        hist_array = hist_array[0]
    if hist_array.shape != (3, *image_size):
        raise ValueError(
            f"直方图形状 {hist_array.shape} != {(3, *image_size)}："
            f"{record.histogram}"
        )
    if hist_array.dtype != np.float32:
        raise ValueError(f"直方图不是 FP32：{record.histogram}")
    # sum=1 针对整份三通道直方图张量，不是每个像素的 RGB 和。
    if abs(float(hist_array.sum()) - 1.0) > 1e-4:
        raise ValueError(f"直方图 sum != 1：{record.histogram}")
    return torch.from_numpy(depth_array)[None], torch.from_numpy(hist_array)


class SeaDiffDataset(Dataset):
    """训练/评测数据集；所有样本读入 CPU FP32 张量后按切片返回。"""

    def __init__(
        self,
        *,
        manifest_path: Union[str, Path],
        data_root: Union[str, Path],
        image_size: Sequence[int] = (336, 336),
    ):
        if (
            not isinstance(image_size, (tuple, list))
            or len(image_size) != 2
            or any(type(v) is not int or v < 8 for v in image_size)
        ):
            raise ValueError("image_size 必须是两个 >=8 的整数。")
        self.image_size = tuple(image_size)
        self.manifest_path = Path(manifest_path).expanduser().resolve(strict=True)
        self.data_root = Path(data_root).expanduser().resolve(strict=True)
        self.records = read_manifest(
            self.manifest_path, self.data_root, require_reference=True
        )
        resize = Resize(
            self.image_size, interpolation=InterpolationMode.BILINEAR
        )
        to_tensor = ToTensor()

        underwater, reference, depth, histogram = [], [], [], []
        for record in self.records:
            underwater.append(
                to_tensor(resize(_read_rgb(record.underwater)))
            )
            reference.append(
                to_tensor(resize(_read_rgb(record.reference)))
            )
            depth_item, hist_item = _load_condition_arrays(
                record, self.image_size
            )
            depth.append(depth_item)
            histogram.append(hist_item)

        # 内存保存 [N,3,H,W] 的 U/Y/H 与 [N,1,H,W] 的 D；取样时移除 N 维，
        # 训练器再 stack 成 batch 并搬到设备。数据集本身不构造 J 或 x_t。
        self.underwater = torch.stack(underwater)
        self.reference = torch.stack(reference)
        self.depth = torch.stack(depth)
        self.histogram = torch.stack(histogram)
        self.sample_ids = tuple(record.sample_id for record in self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_ids[index],
            "underwater": self.underwater[index],
            "reference": self.reference[index],
            "depth": self.depth[index],
            "histogram": self.histogram[index],
        }

    def describe(self) -> dict:
        return {
            "manifest": str(self.manifest_path),
            "data_root": str(self.data_root),
            "samples": len(self.records),
            "image_size": list(self.image_size),
            "underwater_shape": list(self.underwater.shape),
            "reference_shape": list(self.reference.shape),
            "depth_shape": list(self.depth.shape),
            "histogram_shape": list(self.histogram.shape),
            "dtype": "float32",
        }


class InferenceDataset(Dataset):
    """推理数据集；reference 可选，按需加载单条样本。"""

    def __init__(
        self,
        *,
        manifest_path: Union[str, Path],
        data_root: Union[str, Path],
        image_size: Sequence[int] = (336, 336),
    ):
        if (
            not isinstance(image_size, (tuple, list))
            or len(image_size) != 2
            or any(type(v) is not int or v < 8 for v in image_size)
        ):
            raise ValueError("image_size 必须是两个 >=8 的整数。")
        self.image_size = tuple(image_size)
        self.records = read_manifest(
            manifest_path, data_root, require_reference=False
        )
        self.resize = Resize(
            self.image_size, interpolation=InterpolationMode.BILINEAR
        )
        self.to_tensor = ToTensor()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        underwater = self.to_tensor(
            self.resize(_read_rgb(record.underwater))
        )
        depth, histogram = _load_condition_arrays(record, self.image_size)
        result: Dict[str, Any] = {
            "sample_id": record.sample_id,
            "underwater": underwater,
            "depth": depth,
            "histogram": histogram,
        }
        # 推理结果的生成只依赖 U/D/H；reference 若存在，供调用方评测使用。
        if record.reference is not None:
            result["reference"] = self.to_tensor(
                self.resize(_read_rgb(record.reference))
            )
        return result

"""从水下图生成深度与 RGB-uv 直方图条件。

输入清单需包含 ``sample_id, underwater, reference, depth, histogram`` 五列；
``depth`` / ``histogram`` 是本脚本要写入的目标路径，均相对于 ``data_root``。
深度采用官方 V1 Small 推理后逐图 min-max 再双线性缩放到训练尺寸；直方图
使用固定参数的 RGBuvHistBlock，保存为 FP32、总和为 1 的 npy。
"""

import argparse
import csv
import os
import sys
from pathlib import Path, PurePosixPath

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from seadiff_distill.data.depth import DepthAnythingV1Small  # noqa: E402
from seadiff_distill.data.histogram import (  # noqa: E402
    RGBuvHistogramBuilder,
)


def relative_file(root: Path, value: str, *, must_exist: bool) -> Path:
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "\\" in value
        or ":" in value
    ):
        raise ValueError(f"路径必须是 data_root 下的相对 POSIX 路径：{value}")
    target = root.joinpath(*relative.parts)
    if must_exist and not target.is_file():
        raise FileNotFoundError(target)
    return target


def atomic_save_npy(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--depth-weights",
        default="weights/depth_anything_v1_small.pth",
    )
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = Path(args.manifest).expanduser().resolve(strict=True)
    data_root = Path(args.data_root).expanduser().resolve(strict=True)
    depth_weights = Path(args.depth_weights).expanduser()
    if not depth_weights.is_absolute():
        depth_weights = ROOT / depth_weights
    depth_weights = depth_weights.resolve(strict=True)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        required = {
            "sample_id",
            "underwater",
            "reference",
            "depth",
            "histogram",
        }
        if not required <= fields:
            raise ValueError(
                f"清单缺少列：{sorted(required - fields)}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError("清单为空。")
    if args.limit:
        rows = rows[: args.limit]

    depth_model = DepthAnythingV1Small(
        str(depth_weights), device=device
    )
    histogram_builder = RGBuvHistogramBuilder(
        device=device, image_size=args.image_size
    )

    written = skipped = 0
    for index, row in enumerate(rows, start=1):
        sample_id = (row.get("sample_id") or "").strip()
        if not sample_id:
            raise ValueError(f"第 {index} 行 sample_id 为空。")
        input_path = relative_file(
            data_root, row["underwater"].strip(), must_exist=True
        )
        depth_path = relative_file(
            data_root, row["depth"].strip(), must_exist=False
        )
        histogram_path = relative_file(
            data_root, row["histogram"].strip(), must_exist=False
        )

        if (
            not args.overwrite
            and depth_path.is_file()
            and histogram_path.is_file()
        ):
            skipped += 1
            continue

        with Image.open(input_path) as image:
            if image.mode in ("I", "F") or image.mode.startswith("I;16"):
                raise ValueError(
                    f"{input_path} 使用高精度模式 {image.mode}"
                )
            rgb = image.convert("RGB")

        # 深度和直方图都只读取水下图 U，清晰 reference 不参与条件生成。
        # 先在原深度分辨率逐图归一化，再缩放，训练读取时不重复 min-max。
        raw_depth = depth_model.infer(np.array(rgb))
        depth01 = depth_model.minmax_float01(raw_depth)
        depth = F.interpolate(
            torch.from_numpy(depth01)[None, None],
            size=(args.image_size, args.image_size),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy().astype(np.float32)
        histogram = (
            histogram_builder.compute(rgb)
            .squeeze(0)
            .numpy()
            .astype(np.float32)
        )

        if depth.shape != (args.image_size, args.image_size):
            raise RuntimeError(f"深度形状异常：{depth.shape}")
        if not np.isfinite(depth).all():
            raise RuntimeError(f"深度含非有限值：{input_path}")
        if histogram.shape != (
            3,
            args.image_size,
            args.image_size,
        ):
            raise RuntimeError(f"直方图形状异常：{histogram.shape}")
        if abs(float(histogram.sum()) - 1.0) > 1e-4:
            raise RuntimeError(f"直方图总和不为 1：{input_path}")

        # 缓存形状分别为 [H,W] 和 [3,H,W]；Dataset 再给深度补单通道维。
        atomic_save_npy(depth, depth_path)
        atomic_save_npy(histogram, histogram_path)
        written += 1
        if index % 50 == 0:
            print(
                f"[conditions] {index}/{len(rows)} "
                f"written={written} skipped={skipped}",
                flush=True,
            )

    print(
        f"完成：written={written} skipped={skipped} total={len(rows)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""RGB-uv 直方图条件图。

使用上游 SeaDiff 的 ``RGBuvHistBlock`` 与固定参数：
``insz=336, h=336, resizing='sampling', method='inverse-quadratic',
sigma=0.02``。输出为 FP32、逐图总和为 1 的 ``[B,3,H,W]`` 张量，不做
uint8 量化，直接作为网络输入的 H 槽位。
"""

import importlib.util
from pathlib import Path
from typing import Dict

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode, Resize, ToTensor

HIST_SIZE = 336
HIST_PARAMS = {
    "insz": HIST_SIZE,
    "h": HIST_SIZE,
    "resizing": "sampling",
    "method": "inverse-quadratic",
    "sigma": 0.02,
    "intensity_scale": True,
    "green_only": False,
}


def _load_block_class():
    """按文件路径加载上游实现，避免把 third_party 做成可安装包。"""
    import seadiff_distill

    project_root = Path(seadiff_distill.__file__).resolve().parents[2]
    module_path = (
        project_root / "third_party" / "SeaDiff" / "utils" / "RGBuvHistBlock.py"
    )
    if not module_path.is_file():
        raise FileNotFoundError(module_path)

    spec = importlib.util.spec_from_file_location(
        "seadiff_upstream_rgbuv_histblock", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RGBuvHistBlock


class RGBuvHistogramBuilder:
    """逐图计算 RGB-uv 直方图条件图。"""

    def __init__(self, *, device: str = "cuda", image_size: int = HIST_SIZE):
        self.device = torch.device(device)
        block_class = _load_block_class()
        self.block = block_class(
            insz=image_size,
            h=image_size,
            resizing=HIST_PARAMS["resizing"],
            method=HIST_PARAMS["method"],
            sigma=HIST_PARAMS["sigma"],
            intensity_scale=HIST_PARAMS["intensity_scale"],
            green_only=HIST_PARAMS["green_only"],
            device=self.device,
        ).to(self.device)
        self.block.eval()
        self.image_size = int(image_size)
        self.resize = Resize(
            (self.image_size, self.image_size),
            interpolation=InterpolationMode.BILINEAR,
        )

    @torch.no_grad()
    def compute(self, image: Image.Image) -> torch.Tensor:
        """输入 PIL RGB 图，返回 CPU FP32 ``[1,3,H,H]``，总和为 1。"""
        tensor = ToTensor()(self.resize(image.convert("RGB"))).unsqueeze(0)
        tensor = tensor.to(self.device, dtype=torch.float32)
        histogram = self.block(tensor)
        if histogram.ndim != 4 or histogram.shape[1] != 3:
            raise RuntimeError(f"直方图输出形状异常：{tuple(histogram.shape)}")
        if not torch.isfinite(histogram).all().item():
            raise RuntimeError("直方图包含 NaN 或 Inf。")
        if abs(float(histogram.sum()) - 1.0) > 1e-4:
            raise RuntimeError("直方图总和不为 1。")
        return histogram.detach().cpu().float()

    def describe(self) -> Dict[str, object]:
        params = dict(HIST_PARAMS)
        params["insz"] = self.image_size
        params["h"] = self.image_size
        return {
            "module": "third_party/SeaDiff/utils/RGBuvHistBlock.py",
            "params": params,
            "image_size": self.image_size,
            "quantization": "none; FP32 normalized sum=1",
        }

"""配对图像质量指标：PSNR / SSIM / L1。

统一约定：
- 输入为 [0,1] 的 FP32 张量或 HWC numpy；
- 指标在 clamp(0,1) 后的图像上计算；
- 逐图计算后再汇总，避免批量平均掩盖分布。
"""

import math
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch import Tensor


def clamp01(image: Tensor) -> Tensor:
    if not isinstance(image, Tensor):
        raise TypeError("image 必须是 Tensor。")
    return image.clamp(0.0, 1.0)


def psnr(prediction: Tensor, reference: Tensor, *, data_range: float = 1.0) -> float:
    """单图 PSNR，输入需形状一致。"""
    if prediction.shape != reference.shape:
        raise ValueError(
            f"PSNR 输入形状不一致：{tuple(prediction.shape)} vs {tuple(reference.shape)}"
        )
    # 网络前向为 FP32；此处仅把指标中的平方误差累计提高到 FP64 精度。
    prediction = clamp01(prediction).double()
    reference = clamp01(reference).double()
    mse = torch.mean((prediction - reference) ** 2).item()
    if mse <= 0.0:
        return float("inf")
    return 10.0 * math.log10((data_range**2) / mse)


def l1(prediction: Tensor, reference: Tensor) -> float:
    if prediction.shape != reference.shape:
        raise ValueError(
            f"L1 输入形状不一致：{tuple(prediction.shape)} vs {tuple(reference.shape)}"
        )
    return torch.mean(torch.abs(clamp01(prediction) - clamp01(reference))).item()


def ssim(prediction: Tensor, reference: Tensor, *, data_range: float = 1.0) -> float:
    """单图 SSIM（skimage 实现，HWC numpy）。"""
    from skimage.metrics import structural_similarity

    if prediction.shape != reference.shape:
        raise ValueError(
            f"SSIM 输入形状不一致：{tuple(prediction.shape)} vs {tuple(reference.shape)}"
        )

    def to_hwc(image: Tensor) -> np.ndarray:
        array = clamp01(image).detach().cpu().double().numpy()
        if array.ndim == 3:
            return np.transpose(array, (1, 2, 0))
        if array.ndim == 2:
            return array
        raise ValueError("SSIM 只接受 [C,H,W] 或 [H,W]。")

    pred = to_hwc(prediction)
    ref = to_hwc(reference)
    channel_axis = 2 if pred.ndim == 3 else None
    return float(
        structural_similarity(
            ref,
            pred,
            data_range=data_range,
            channel_axis=channel_axis,
        )
    )


def summarize(values: Iterable[float]) -> Dict[str, float]:
    """汇总逐图指标；NaN/Inf 单独计数，不静默丢弃。"""
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("指标列表为空。")
    finite = np.isfinite(array)
    return {
        "count": int(array.size),
        "mean": float(array[finite].mean()) if finite.any() else float("nan"),
        "std": float(array[finite].std(ddof=1)) if finite.sum() > 1 else 0.0,
        "min": float(array[finite].min()) if finite.any() else float("nan"),
        "max": float(array[finite].max()) if finite.any() else float("nan"),
        "nonfinite": int((~finite).sum()),
    }


def evaluate_batch(
    prediction: Tensor,
    reference: Tensor,
    *,
    compute_ssim: bool = True,
) -> List[Dict[str, float]]:
    """逐图计算指标；返回与 batch 对齐的列表。"""
    if prediction.ndim != 4 or prediction.shape != reference.shape:
        raise ValueError("prediction/reference 必须是同形状 [B,3,H,W]。")
    rows = []
    for index in range(prediction.shape[0]):
        pred = prediction[index]
        ref = reference[index]
        row = {
            "psnr": psnr(pred, ref),
            "l1": l1(pred, ref),
        }
        if compute_ssim:
            row["ssim"] = ssim(pred, ref)
        rows.append(row)
    return rows

"""上游 ``get_A`` 的最小依赖实现。

``get_A`` 用整图尺度的高斯模糊估计环境光 A；它只依赖 PIL 与 numpy，
必须在 CPU 上逐图执行。这里只保留物理先验实际使用的函数，避免把上游
训练/可视化工具一起打包。
"""

import numpy as np
from PIL import Image, ImageFilter
from torchvision.transforms import ToTensor


def np_to_pil(img_np):
    """把 ``[C,H,W]`` 的 [0,1] 数组转成 PIL 图像。"""
    array = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    if img_np.shape[0] == 1:
        array = array[0]
    else:
        if img_np.shape[0] != 3:
            raise ValueError("只支持 1 或 3 通道图像。")
        array = array.transpose(1, 2, 0)
    return Image.fromarray(array)


def torch_to_np(img_var):
    """把 ``[1,C,H,W]`` 张量转成 ``[C,H,W]`` numpy 数组。"""
    return img_var.detach().cpu().numpy()


def get_A(x):
    """逐图估计环境光：PIL 高斯模糊，窗口为图像宽高均值。"""
    x_np = np.clip(torch_to_np(x), 0, 1)
    x_pil = np_to_pil(x_np)
    h, w = x_pil.size
    windows = (h + w) / 2
    ambient = x_pil.filter(ImageFilter.GaussianBlur(windows))
    return ToTensor()(ambient)

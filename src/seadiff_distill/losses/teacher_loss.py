"""教师重建损失。

    L_teacher = mean_MSE(x0_pred, reference)
                + 0.1 * VGG19_perceptual(x0_pred, reference)

感知网络固定为 torchvision VGG19 IMAGENET1K_V1 的
relu1_2 / relu2_2 / relu3_3，每层 mean MSE 后等权平均。
"""

from typing import Dict, Tuple

from torch import Tensor, nn
from torch.nn import functional as F

from .perceptual import VGG19PerceptualLoss

LOSS_VERSION = "teacher_loss_v1_mse_plus_vgg19_0.1"


class TeacherLoss(nn.Module):
    """MSE + 0.1 * VGG19 感知损失。"""

    def __init__(
        self,
        *,
        perceptual_weight: float = 0.1,
        perceptual: nn.Module | None = None,
    ):
        super().__init__()
        if float(perceptual_weight) != 0.1:
            raise ValueError("教师损失要求 perceptual_weight=0.1。")
        self.perceptual_weight = 0.1
        self.perceptual = (
            perceptual
            if perceptual is not None
            else VGG19PerceptualLoss(device="cpu")
        )

    def forward(
        self, prediction: Tensor, target: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if prediction.shape != target.shape:
            raise ValueError(
                "prediction/target 形状不一致："
                f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
            )
        # mean 归约覆盖 batch、通道和空间维；输入是未经 clamp 的 x0 预测。
        # 同一重建损失也供学生的 GT 监督使用，不表示此处会调用教师网络。
        mse = F.mse_loss(prediction, target)
        perceptual = self.perceptual(prediction, target)
        loss = mse + self.perceptual_weight * perceptual
        return loss, {
            "mse": mse,
            "perceptual": perceptual,
            "perceptual_weighted": self.perceptual_weight * perceptual,
        }

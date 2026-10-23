"""学生蒸馏损失。

    L_total = L_original + lambda_KD * L_distill
    L_original = TeacherLoss(student_x0, GT)
    L_distill  = mean_MSE(student_x0, stopgrad(teacher_x0_target))

``lambda_KD`` 固定为 0.5。各分项单独返回，便于训练日志记录。
"""

import math
from typing import Dict, Optional

from torch import Tensor, nn
from torch.nn import functional as F

from .teacher_loss import TeacherLoss

LAMBDA_KD_DEFAULT = 0.5


class DistillationLoss(nn.Module):
    """原教师损失与蒸馏损失的加权和。"""

    def __init__(
        self,
        *,
        base_loss: Optional[TeacherLoss] = None,
        lambda_kd: float = LAMBDA_KD_DEFAULT,
    ):
        super().__init__()
        if not math.isfinite(lambda_kd) or lambda_kd < 0:
            raise ValueError("lambda_kd 必须是非负有限数值。")
        self.lambda_kd = float(lambda_kd)

        if base_loss is None:
            base_loss = TeacherLoss()
        if not isinstance(base_loss, TeacherLoss):
            raise TypeError("base_loss 必须是 TeacherLoss。")
        self.base_loss = base_loss

    def forward(
        self,
        student_pred: Tensor,
        reference: Tensor,
        teacher_target: Tensor,
    ) -> Dict[str, Tensor]:
        shape = student_pred.shape
        if reference.shape != shape or teacher_target.shape != shape:
            raise ValueError(
                "student_pred/reference/teacher_target 形状必须一致。"
            )

        original, parts = self.base_loss(student_pred, reference)
        distill = F.mse_loss(
            student_pred, teacher_target.detach(), reduction="mean"
        )
        return {
            "loss_total": original + self.lambda_kd * distill,
            "loss_mse": parts["mse"],
            "loss_perceptual": parts["perceptual"],
            "loss_original": original,
            "loss_distill": distill,
        }

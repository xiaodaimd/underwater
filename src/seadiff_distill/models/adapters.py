"""用于 SeaDiff 学生网络的卷积 Adapter。"""

from torch import Tensor, nn


class ConvAdapter2d(nn.Module):
    """产生与输入同形状的特征修正量 delta。

    输入与输出形状均为 [B, C, H, W]。
    本模块只返回 delta。
    外部学生网络负责计算：h_new = h + gate * delta。
    """

    def __init__(self, channels: int, hidden_channels: int):
        super().__init__()

        if type(channels) is not int or channels <= 0:
            raise ValueError("channels 必须是正整数。")

        if type(hidden_channels) is not int or hidden_channels <= 0:
            raise ValueError("hidden_channels 必须是正整数。")

        if hidden_channels > channels:
            raise ValueError("瓶颈通道数不能超过输入通道数。")

        self.channels = channels
        self.hidden_channels = hidden_channels

        # 降低通道数，控制新增参数和计算量。
        self.down = nn.Conv2d(
            channels,
            hidden_channels,
            kernel_size=1,
            bias=False,
        )

        # 不使用原地操作，为后续双层梯度计算保留清晰的计算图。
        self.activation = nn.SiLU(inplace=False)

        # 恢复原通道数，输出特征修正量。
        self.up = nn.Conv2d(
            hidden_channels,
            channels,
            kernel_size=1,
            bias=False,
        )

        # 两个 1×1 无偏置投影不改变 H/W；参数量为 2*C*hidden_channels。
        # 初始修正量为零，接入学生时保持原骨干的初始输出。
        # 首次反传时 up 可先得到梯度；down 与门控的作用随 up 离开零值后显现。
        nn.init.zeros_(self.up.weight)

    def forward(self, features: Tensor) -> Tensor:
        """返回修正量，不截断梯度，也不在这里执行残差相加。"""
        if features.ndim != 4:
            raise ValueError("输入必须为 [B, C, H, W] 四维张量。")

        if features.shape[1] != self.channels:
            raise ValueError(
                f"期望 {self.channels} 个通道，"
                f"实际为 {features.shape[1]} 个通道。"
            )

        return self.up(self.activation(self.down(features)))
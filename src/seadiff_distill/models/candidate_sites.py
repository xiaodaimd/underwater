"""SeaDiff 默认去噪 UNet 的 Adapter 候选插入点。

位置路径相对于 Denoise_UNet，而不是整个 DocDiff。
候选点均位于对应模块的输出端。
本文件只定义搜索空间，不执行网络计算。
"""

from dataclasses import dataclass
from typing import Sequence, Tuple


SEADIFF_UPSTREAM_COMMIT = (
    "623d823ee49bab21c5ce3db00aa67d28ffde82df"
)


@dataclass(frozen=True)
class AdapterSite:
    """不可修改的候选位置描述。

    spatial_stride 表示相对输入的名义下采样倍数。
    输入高宽能被 8 整除时，对应整数尺度的特征尺寸。
    expected_type 用于后续装配时核对模块类型。
    """

    name: str
    module_path: str
    channels: int
    spatial_stride: int
    expected_type: str


def get_default_sites(
    *,
    model_channels: int,
    channel_mult: Sequence[int],
    num_resblocks: int,
) -> Tuple[AdapterSite, ...]:
    """返回与官方默认结构对应的九个候选位置。

    修改学生骨干后，需要重新核对映射。
    后续装配还会检查实际模块、类型和通道数。
    """
    actual_config = (
        model_channels,
        tuple(channel_mult),
        num_resblocks,
    )
    expected_config = (32, (1, 2, 3, 4), 1)

    if actual_config != expected_config:
        raise ValueError(
            "当前候选映射仅支持 SeaDiff 默认结构："
            "model_channels=32、channel_mult=(1, 2, 3, 4)、"
            f"num_resblocks=1；实际配置为 {actual_config}。"
        )

    # 这些位置在残差块之后、对应尺度重采样之外；Adapter 保持特征形状，
    # 不新增下采样/上采样，也不直接作用于 RGB 输入或最终输出。
    return (
        # 编码端：各尺度最后一个 DownBlock 的输出。
        AdapterSite("enc1", "down.0", 32, 1, "DownBlock"),
        AdapterSite("enc2", "down.2", 64, 2, "DownBlock"),
        AdapterSite("enc3", "down.4", 96, 4, "DownBlock"),
        AdapterSite("enc4", "down.6", 128, 8, "DownBlock"),

        # 瓶颈输出。
        AdapterSite("mid", "middle", 128, 8, "MiddleBlock"),

        # 解码端：各尺度最后一个 UpBlock 的输出。
        AdapterSite("dec4", "up.1", 128, 8, "UpBlock"),
        AdapterSite("dec3", "up.4", 96, 4, "UpBlock"),
        AdapterSite("dec2", "up.7", 64, 2, "UpBlock"),
        AdapterSite("dec1", "up.10", 32, 1, "UpBlock"),
    )
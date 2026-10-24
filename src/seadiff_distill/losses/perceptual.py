"""论文规定的 VGG-19 感知损失（relu1_2 / relu2_2 / relu3_3）。

论文 PDF6 式16 后正文：预训练 VGG-19。论文未写层索引、reduction 或输入
归一化；本实现采用以下**补充选择**并记录：
- torchvision VGG19 IMAGENET1K_V1 `features[:16]`；
- 层名 relu1_2 (index 3)、relu2_2 (index 8)、relu3_3 (index 15)；
- 每层 mean MSE，等权平均；
- 输入保持 [0,1]，不额外做 ImageNet mean/std（沿用上游 perceptual loss 输入）。
感知项权重 0.1 由训练器施加。
"""

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import VGG19_Weights, vgg19

VGG19_SHA256 = (
    "dcbb9e9dad569fff7a846263a77324fc34978fea2bfb039c012d710e1776ae44"
)


def _weights_sha256() -> str:
    path = Path(torch.hub.get_dir()) / "checkpoints" / Path(
        VGG19_Weights.IMAGENET1K_V1.url
    ).name
    if not path.is_file():
        return "unknown"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class VGG19PerceptualLoss(nn.Module):
    """VGG19 relu1_2 / relu2_2 / relu3_3 的等权 mean MSE。"""

    layer_mapping = {"3": "relu1_2", "8": "relu2_2", "15": "relu3_3"}

    def __init__(self, *, device: torch.device):
        super().__init__()
        model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:16]
        self._validate_structure(model)
        model = model.to(device).eval()
        # 冻结 VGG 参数仍保留对 prediction 的梯度；因此不能把感知前向
        # 整体放进 no_grad，否则学生/去噪器收不到感知项监督。
        for parameter in model.parameters():
            parameter.requires_grad = False
        self.vgg_layers = model
        self.weights_sha256 = _weights_sha256()
        if self.weights_sha256 != VGG19_SHA256:
            raise ValueError(
                "VGG19 权重 SHA256 不匹配："
                f"{self.weights_sha256} != {VGG19_SHA256}"
            )

    @classmethod
    def _validate_structure(cls, features: nn.Sequential) -> None:
        """按层名/结构检查，而不是盲信索引。"""
        if len(features) != 16:
            raise ValueError(f"features[:16] 长度异常：{len(features)}")
        expected_conv_channels = {"2": 64, "7": 128, "14": 256}
        for index, channels in expected_conv_channels.items():
            module = features[int(index)]
            if not isinstance(module, nn.Conv2d) or module.out_channels != channels:
                raise ValueError(
                    f"features[{index}] 不是 {channels} 通道 Conv2d：{module}"
                )
        for index in (3, 8, 15):
            if not isinstance(features[index], nn.ReLU):
                raise ValueError(f"features[{index}] 不是 ReLU：{features[index]}")

    def output_features(self, x: torch.Tensor):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_mapping:
                output[self.layer_mapping[name]] = x
        return list(output.values())

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        if prediction.shape != target.shape:
            raise ValueError(
                f"感知损失输入形状不一致：{tuple(prediction.shape)} vs {tuple(target.shape)}"
            )
        # 无 ImageNet mean/std、无内部 clamp；GT 通常本身不要求梯度。
        # 每个特征层先求自己的 mean MSE，再对三个标量等权平均，
        # 并非把不同空间大小的特征拼接后统一求平均。
        prediction_features = self.output_features(prediction)
        target_features = self.output_features(target)
        losses = [
            F.mse_loss(pred, tgt)
            for pred, tgt in zip(prediction_features, target_features)
        ]
        return sum(losses) / len(losses)

    def describe(self) -> dict:
        return {
            "model": "torchvision VGG19 IMAGENET1K_V1 features[:16]",
            "layers": ["relu1_2", "relu2_2", "relu3_3"],
            "reduction": "per-layer mean MSE, equal average",
            "input_normalization": "none (inputs in [0,1])",
            "perceptual_weight": 0.1,
            "weights_url": VGG19_Weights.IMAGENET1K_V1.url,
            "weights_sha256": self.weights_sha256,
            "note": (
                "论文只写 VGG-19；层名/reduction/输入归一化是复现补充选择，"
                "不是论文明示"
            ),
        }

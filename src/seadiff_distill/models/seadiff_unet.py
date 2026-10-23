import math

import numpy as np
import torch
from torch import nn


class Swish(nn.Module):
    r"""
    ### Swish actiavation function
    $$x \cdot \sigma(x)$$
    """

    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    """
    ### Embeddings for $t$
    """

    def __init__(self, n_channels: int):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels
        # First linear layer
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        # Activation
        self.act = Swish()
        # Second linear layer
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        # Create sinusoidal position embeddings
        # [same as those from the transformer](../../transformers/positional_encoding.html)
        #
        # \begin{align}
        # PE^{(1)}_{t,i} &= sin\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg) \\
        # PE^{(2)}_{t,i} &= cos\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg)
        # \end{align}
        #
        # where $d$ is `half_dim`
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        #
        return emb


def _norm_groups(channels: int) -> int:
    """选择能整除通道数的 GroupNorm 组数（优先 32）。"""
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    """
    ### Residual block
    A residual block has two convolution layers with group normalization.
    Each resolution is processed with two residual blocks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        dropout: float = 0.1,
        is_noise: bool = True,
        use_norm: bool = False,
    ):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of input channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        * `dropout` is the dropout rate
        """
        super().__init__()
        # Group normalization and the first convolution layer
        self.is_noise = is_noise
        self.use_norm = use_norm
        self.act1 = Swish()
        if use_norm:
            self.norm1 = nn.GroupNorm(_norm_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)
        )

        # Group normalization and the second convolution layer
        self.act2 = Swish()
        if use_norm:
            self.norm2 = nn.GroupNorm(_norm_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)
        )

        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        # Linear layer for time embeddings
        if self.is_noise:
            self.time_emb = nn.Linear(time_channels, out_channels)
            self.time_act = Swish()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # First convolution layer
        if self.use_norm:
            x = self.norm1(x)
        h = self.conv1(self.act1(x))
        # Add time embeddings
        if self.is_noise:
            h += self.time_emb(self.time_act(t))[:, :, None, None]
        # Second convolution layer
        if self.use_norm:
            h = self.norm2(h)
        h = self.conv2(self.dropout(self.act2(h)))

        # Add the shortcut connection and return
        return h + self.shortcut(x)


class DownBlock(nn.Module):
    """
    ### Down block
    This combines `ResidualBlock` and `AttentionBlock`. These are used in the first half of U-Net at each resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        is_noise: bool = True,
        use_norm: bool = False,
    ):
        super().__init__()
        self.res = ResidualBlock(
            in_channels,
            out_channels,
            time_channels,
            is_noise=is_noise,
            use_norm=use_norm,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        return x


class UpBlock(nn.Module):
    """
    ### Up block
    This combines `ResidualBlock` and `AttentionBlock`. These are used in the second half of U-Net at each resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        is_noise: bool = True,
    ):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(
            in_channels + out_channels, out_channels, time_channels, is_noise=is_noise
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        return x


def get_pad(in_, ksize, stride, atrous=1):
    """空洞卷积保持尺寸所需的对称 padding。"""
    out_ = np.ceil(float(in_) / stride)
    return int(
        ((out_ - 1) * stride + atrous * (ksize - 1) + 1 - in_) / 2
    )


class MiddleBlock(nn.Module):
    """
    ### Middle block
    It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock`.
    This block is applied at the lowest resolution of the U-Net.
    """

    def __init__(
        self,
        n_channels: int,
        time_channels: int,
        is_noise: bool = True,
        use_norm: bool = False,
    ):
        super().__init__()
        self.use_norm = use_norm
        self.res1 = ResidualBlock(
            n_channels,
            n_channels,
            time_channels,
            is_noise=is_noise,
            use_norm=use_norm,
        )
        self.dia1 = nn.Conv2d(
            n_channels, n_channels, 3, 1, dilation=2, padding=get_pad(16, 3, 1, 2)
        )
        self.dia2 = nn.Conv2d(
            n_channels, n_channels, 3, 1, dilation=4, padding=get_pad(16, 3, 1, 4)
        )
        self.dia3 = nn.Conv2d(
            n_channels, n_channels, 3, 1, dilation=8, padding=get_pad(16, 3, 1, 8)
        )
        self.dia4 = nn.Conv2d(
            n_channels, n_channels, 3, 1, dilation=16, padding=get_pad(16, 3, 1, 16)
        )
        if use_norm:
            groups = _norm_groups(n_channels)
            self.dia_norms = nn.ModuleList(
                [nn.GroupNorm(groups, n_channels) for _ in range(4)]
            )
        self.res2 = ResidualBlock(
            n_channels,
            n_channels,
            time_channels,
            is_noise=is_noise,
            use_norm=use_norm,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.dia1(x)
        if self.use_norm:
            x = self.dia_norms[0](x)
        x = self.dia2(x)
        if self.use_norm:
            x = self.dia_norms[1](x)
        x = self.dia3(x)
        if self.use_norm:
            x = self.dia_norms[2](x)
        x = self.dia4(x)
        if self.use_norm:
            x = self.dia_norms[3](x)
        x = self.res2(x, t)
        return x


class Upsample(nn.Module):
    r"""
    ### Scale up the feature map by $2 \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class Downsample(nn.Module):
    r"""
    ### Scale down the feature map by $\frac{1}{2} \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class Denoise_UNet(nn.Module):
    """
    ## U-Net
    """

    def __init__(
        self, input_channels, output_channels, n_channels, ch_mults, n_blocks, is_noise
    ):
        """
        * `image_channels` is the number of channels in the image. $3$ for RGB.
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        """
        super().__init__()

        # Number of resolutions
        n_resolutions = len(ch_mults)

        # Project image into feature map
        self.image_proj = nn.Conv2d(
            input_channels, n_channels, kernel_size=(3, 3), padding=(1, 1)
        )

        # Time embedding layer. Time embedding has `n_channels * 4` channels
        self.is_noise = is_noise
        if is_noise:
            self.time_emb = TimeEmbedding(n_channels * 4)

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = n_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(
                    DownBlock(
                        in_channels, out_channels, n_channels * 4, is_noise=is_noise
                    )
                )
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, n_channels * 4, is_noise=False)

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = n_channels * ch_mults[i]
            for _ in range(n_blocks):
                up.append(
                    UpBlock(
                        in_channels, out_channels, n_channels * 4, is_noise=is_noise
                    )
                )
            # Final block to reduce the number of channels
            in_channels = n_channels * (ch_mults[i - 1] if i >= 1 else 1)
            up.append(
                UpBlock(in_channels, out_channels, n_channels * 4, is_noise=is_noise)
            )
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        self.act = Swish()
        self.final = nn.Conv2d(
            in_channels, output_channels, kernel_size=(3, 3), padding=(1, 1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size]`
        """

        # Get time-step embeddings
        if self.is_noise:
            t = self.time_emb(t)
        else:
            t = None
        # Get image projection
        x = self.image_proj(x)

        # `h` will store outputs at each resolution for skip connection
        h = [x]
        # First half of U-Net
        for m in self.down:
            x = m(x, t)
            h.append(x)

        # Middle (bottom)
        x = self.middle(x, t)

        # Second half of U-Net
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                # print(x.shape, s.shape)
                x = torch.cat((x, s), dim=1)
                #
                x = m(x, t)

        # Final normalization and convolution
        return self.final(self.act(x))


class Beta_UNet(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        n_channels,
        ch_mults,
        n_blocks,
        use_norm: bool = False,
    ):
        super().__init__()
        is_noise = False
        self.use_norm = use_norm
        # Number of resolutions
        n_resolutions = len(ch_mults)

        # Project image into feature map
        self.image_proj = nn.Conv2d(
            input_channels, n_channels, kernel_size=(3, 3), padding=(1, 1)
        )

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = n_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(
                    DownBlock(
                        in_channels,
                        out_channels,
                        n_channels * 4,
                        is_noise=is_noise,
                        use_norm=use_norm,
                    )
                )
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))
        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(
            out_channels,
            n_channels * 4,
            is_noise=False,
            use_norm=use_norm,
        )
        self.act = Swish()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.transform = nn.Sequential(nn.Linear(128, 3), Swish(), nn.Linear(3, 3))

    def forward(self, x: torch.Tensor):
        t = None
        x = self.image_proj(x)
        h = [x]
        for m in self.down:
            x = m(x, t)
            h.append(x)
        x = self.middle(x, t)
        logits = self.transform(self.pool(x).squeeze())
        if self.use_norm:
            # 数值稳定性补充：前向夹紧 logits 到 ±8（beta 严格在 (0,1) 内），
            # 反向直通，避免 float32 sigmoid 精确饱和后梯度恒为 0。
            clamped = logits.clamp(-8.0, 8.0)
            logits = logits + (clamped - logits).detach()
        x = torch.sigmoid(logits)
        return x.unsqueeze(-1).unsqueeze(-1)


class DocDiff(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        n_channels,
        ch_mults,
        n_blocks,
        beta_use_norm: bool = False,
    ):
        super(DocDiff, self).__init__()
        self.beta_use_norm = bool(beta_use_norm)
        self.beta_predictor = Beta_UNet(
            3,
            3,
            n_channels,
            ch_mults,
            n_blocks,
            use_norm=self.beta_use_norm,
        )
        self.denoiser = Denoise_UNet(
            12, 3, n_channels, ch_mults, n_blocks, is_noise=True
        )





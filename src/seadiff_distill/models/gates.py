"""Adapter 候选位置的全局可学习门控。"""

from typing import Dict, Sequence

import torch
from torch import Tensor, nn


class SiteGates(nn.Module):
    """为每个候选位置提供一个连续门控。

    forward 返回形状 [候选点数量] 的张量。
    门控顺序与 site_names 完全一致。

    本模块只负责门控计算。
    预热、双层更新和最终位置裁剪由其他模块负责。
    """

    def __init__(self, site_names: Sequence[str]):
        super().__init__()

        if isinstance(site_names, (str, bytes)):
            raise ValueError("site_names 应为名称列表或元组。")

        names = tuple(site_names)

        if not names:
            raise ValueError("至少需要一个候选位置。")

        if any(
            not isinstance(name, str) or not name.isidentifier()
            for name in names
        ):
            raise ValueError("候选名称必须是合法的非空标识符。")

        if len(set(names)) != len(names):
            raise ValueError("候选名称不能重复。")

        self.site_names = names

        # sigmoid(0) = 0.5，为各候选点提供相同的初始门控。
        # 此参数后续只交给外层优化器更新。
        self.logits = nn.Parameter(
            torch.zeros(len(names), dtype=torch.float32)
        )

    def forward(self) -> Tensor:
        """返回可微门控，保留外层优化需要的计算图。"""
        # 每个位置仅一个全局标量，不随图像、像素或时间步变化；
        # E3 随 t 变化的物理插值 w(t) 在 teacher/conditions.py 中独立计算。
        return torch.sigmoid(self.logits)

    @torch.no_grad()
    def export_values(self) -> Dict[str, float]:
        """导出日志数值；训练计算必须使用 forward 的张量。"""
        values = self().detach().cpu().tolist()
        return dict(zip(self.site_names, values))

    def get_extra_state(self):
        """保存候选名称与顺序，防止权重载入后错配位置。"""
        return {
            "schema_version": 1,
            "site_names": self.site_names,
        }

    def set_extra_state(self, state):
        """载入时核对版本和候选顺序，不自动重排门控。"""
        if (
            not isinstance(state, dict)
            or state.get("schema_version") != 1
            or state.get("site_names") != self.site_names
        ):
            raise ValueError(
                "门控权重的版本或候选位置顺序与当前模型不一致。"
            )
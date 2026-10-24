"""带可学习位置门控的 SeaDiff 单步学生网络。"""

import copy
import math
from typing import Dict, Iterator, Mapping, Sequence, Tuple

import torch
from torch import Tensor, nn

from .adapters import ConvAdapter2d
from .candidate_sites import get_default_sites
from .gates import SiteGates
from .seadiff_unet import Denoise_UNet, Upsample


class OneStepSeaDiffStudent(nn.Module):
    """复用 SeaDiff 去噪骨干，单次前向预测恢复图。

    输入顺序（E3 真实拼接，pack_paper_order）：[噪声状态 x_t, 水下图 U, 直方图 H, 物理先验 J]。
    输入形状：[B, 12, H, W]。
    输出形状：[B, 3, H, W]，不在网络内部裁剪数值范围。

    时间步支持：
    - forward(packed_input) 使用 fixed_timestep（默认 999，部署单步协议）；
    - forward(packed_input, timesteps=[B]) 使用逐样本时间步
      （渐进蒸馏 KD 分支用学生网格起点 t，L_original 分支独立采样 t）。

    第一版对应官方 PRE_ORI=True 的 x0 预测设置。
    权重加载由后续加载器负责，本类不会自动获取预训练权重。
    """

    # 当前 E3 的真实槽位内容是 [x_t,U,0,J_t]，H/J 是上述接口槽位名。
    # 当前 trainer/bilevel_optimizer 对同一次学生预测同时计算 GT 与 KD 损失，
    # 并未采用上方旧说明中“L_original 分支独立采样 t”的额外前向。
    def __init__(
        self,
        backbone: Denoise_UNet,
        adapter_reduction: int = 4,
        fixed_timestep: int = 999,
    ):
        super().__init__()

        if not isinstance(backbone, Denoise_UNet):
            raise TypeError("backbone 必须是兼容副本中的 Denoise_UNet。")

        if type(adapter_reduction) is not int or adapter_reduction <= 0:
            raise ValueError("adapter_reduction 必须是正整数。")

        if (
            type(fixed_timestep) is not int
            or not 0 <= fixed_timestep < 1000
        ):
            raise ValueError("第一版时间步必须是 0 到 999 之间的整数。")

        if any(
            parameter.device.type != "cpu"
            or parameter.dtype != torch.float32
            for parameter in backbone.parameters()
        ):
            raise ValueError(
                "请先使用 CPU、FP32 骨干构造学生，"
                "再对完整学生调用 to(device)。"
            )

        # 独立复制，避免冻结或修改学生时影响教师对象。
        self.backbone = copy.deepcopy(backbone)

        # 选用默认映射，再据真实模块结构核对。
        self.sites = get_default_sites(
            model_channels=32,
            channel_mult=(1, 2, 3, 4),
            num_resblocks=1,
        )
        self._validate_backbone()

        self.adapter_reduction = adapter_reduction
        self.adapters = nn.ModuleDict(
            {
                site.name: ConvAdapter2d(
                    channels=site.channels,
                    hidden_channels=max(
                        1, site.channels // adapter_reduction
                    ),
                )
                for site in self.sites
            }
        )

        self.gates = SiteGates(
            [site.name for site in self.sites]
        )

        self._site_index = {
            site.module_path: index
            for index, site in enumerate(self.sites)
        }

        # 随模型保存、加载和迁移设备，但不参与优化。
        self.register_buffer(
            "fixed_timestep",
            torch.tensor(fixed_timestep, dtype=torch.long),
        )

        # 硬选择时使用：与 sites 顺序一致的固定门控值。
        # 未启用硬选择时，forward 仍使用 self.gates() 的可微门控。
        self.register_buffer(
            "fixed_gate_values",
            torch.zeros(len(self.sites), dtype=torch.float32),
        )
        self._use_fixed_gates = False
        self._selected_sites: Tuple[str, ...] = tuple(
            site.name for site in self.sites
        )
        self._hard_selected = False

        # 预热阶段使用：门控覆盖为固定常数（0.5），九分支均可训练；
        # 与硬选择的固定门控相互独立。None 表示关闭覆盖。
        self._gate_override: float | None = None

        # 只冻结骨干参数；后续前向不能整体包在 no_grad 中，
        # 后部冻结卷积仍需把损失梯度传回前部 Adapter 和门控。
        self.backbone.requires_grad_(False)
        self.backbone.zero_grad(set_to_none=True)
        self.backbone.eval()

    def _validate_backbone(self) -> None:
        """检查骨干是否满足当前九点映射的结构约定。"""
        backbone = self.backbone

        if not backbone.is_noise:
            raise ValueError("学生骨干必须保留时间步条件。")

        if (
            backbone.image_proj.in_channels != 12
            or backbone.image_proj.out_channels != 32
            or backbone.final.out_channels != 3
            or len(backbone.down) != 7
            or len(backbone.up) != 11
        ):
            raise ValueError("骨干结构与当前默认候选映射不一致。")

        for site in self.sites:
            block = backbone.get_submodule(site.module_path)

            if type(block).__name__ != site.expected_type:
                raise ValueError(
                    f"{site.module_path} 的模块类型与映射不一致。"
                )

            if site.expected_type == "MiddleBlock":
                output_conv = block.res2.conv2
            else:
                output_conv = block.res.conv2

            if output_conv.out_channels != site.channels:
                raise ValueError(
                    f"{site.module_path} 的输出通道与映射不一致。"
                )

    def train(self, mode: bool = True):
        """学生进入训练模式时，冻结骨干仍保持 eval 模式。"""
        super().train(mode)
        self.backbone.eval()
        return self

    # ------------------------------------------------------------------
    # 参数接口
    # ------------------------------------------------------------------
    def adapter_parameters(self) -> Iterator[nn.Parameter]:
        """提供给内层优化器的 Adapter 参数（只含当前激活且仍存在的分支）。

        硬选择后落选分支不参与前向，也不应进入优化器/超梯度计算：
        BilevelSearchStep 用 autograd.grad(allow_unused=False) 求 Adapter
        梯度，未参与前向的落选参数会直接报错。九分支全部激活时
        （主实验搜索期）与旧行为完全一致。
        """
        for name in self._selected_sites:
            if name not in self.adapters:
                continue
            yield from self.adapters[name].parameters()

    def architecture_parameters(self) -> Iterator[nn.Parameter]:
        """提供给外层优化器的位置门控参数。"""
        return self.gates.parameters()

    def site_names(self) -> Tuple[str, ...]:
        return tuple(site.name for site in self.sites)

    def active_site_names(self) -> Tuple[str, ...]:
        return tuple(self._selected_sites)

    def is_hard_selected(self) -> bool:
        return self._hard_selected

    # ------------------------------------------------------------------
    # 硬选择 / Top-k
    # ------------------------------------------------------------------
    def configure_hard_selection(
        self,
        site_names: Sequence[str],
        gate_values: Mapping[str, float],
    ) -> None:
        """固定门控、只执行选中的 Adapter 分支。

        - 未选中的分支在 forward 中**不会被调用**（不是乘 0）；
        - 选中分支使用 gate_values 中的固定常数；
        - 不删除模块，因此 checkpoint 结构保持稳定，便于恢复/继续训练。
        """
        if isinstance(site_names, (str, bytes)):
            raise ValueError("site_names 必须是名称序列。")
        requested = tuple(site_names)
        if not requested:
            raise ValueError("至少选择一个位置。")
        if len(set(requested)) != len(requested):
            raise ValueError("选中位置不能重复。")

        all_names = self.site_names()
        unknown = tuple(name for name in requested if name not in all_names)
        if unknown:
            raise ValueError(f"未知位置：{unknown}")

        missing = tuple(name for name in requested if name not in gate_values)
        if missing:
            raise ValueError(f"缺少门控值：{missing}")

        # 按 canonical 顺序保存，避免集合顺序影响日志。
        selected = tuple(name for name in all_names if name in set(requested))

        values = torch.zeros_like(self.fixed_gate_values)
        for name in selected:
            value = float(gate_values[name])
            if not (value == value) or value in (float("inf"), float("-inf")):
                raise ValueError(f"位置 {name} 的门控值不是有限数。")
            values[all_names.index(name)] = value

        # 固定值沿用搜索得到的幅度，不把入选门控自动变成 1。
        # 是否按 Top-k 选择由调用方决定，本方法只实施给定结构。
        self.fixed_gate_values.copy_(values)
        self._selected_sites = selected
        self._hard_selected = True
        self._use_fixed_gates = True

    def clear_hard_selection(self) -> None:
        """恢复为九分支可微门控。"""
        self._selected_sites = self.site_names()
        self._hard_selected = False
        self._use_fixed_gates = False
        self.fixed_gate_values.zero_()

    def set_gate_override(self, value: float | None) -> None:
        """预热阶段：把全部九分支门控覆盖为固定常数（无梯度）。

        value=None 恢复可微门控。硬选择生效期间调用会抛错，防止冲突。
        """
        if self._hard_selected:
            raise RuntimeError("硬选择已配置，不能再用预热覆盖。")
        if value is not None:
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("门控覆盖必须是有限数值。")
            self._gate_override = value
        else:
            self._gate_override = None

    @property
    def gate_override(self) -> float | None:
        return self._gate_override

    def freeze_unselected_adapters(self, *, freeze_gates: bool = True) -> None:
        """冻结落选分支；默认同时冻结门控参数。

        固定位置消融需要让门控参数保持 requires_grad=True 以通过
        BilevelSearchStep 的参数组一致性校验，但引擎以 update_gates=False
        运行，门控优化器从不 step，logits 实际保持初值（freeze_gates=False）。
        """
        if not self._hard_selected:
            raise RuntimeError("请先调用 configure_hard_selection。")

        selected = set(self._selected_sites)
        for name, adapter in self.adapters.items():
            adapter.requires_grad_(name in selected)
        if freeze_gates:
            self.gates.logits.requires_grad_(False)

    def prune_unselected_adapters(self) -> None:
        """删除落选 Adapter 模块。仅用于导出，不可在训练中调用。"""
        if not self._hard_selected:
            raise RuntimeError("请先调用 configure_hard_selection。")
        for name in list(self.adapters.keys()):
            if name not in set(self._selected_sites):
                del self.adapters[name]

    def absorb_fixed_gates_into_adapters(self) -> None:
        """把固定门控吸收进上投影权重：g * up(x) == (g*up)(x)。

        只对当前存在的 Adapter 生效；吸收后固定门控变为 1。
        该变换在数学上等价，必须由测试验证数值一致性。
        """
        if not self._hard_selected:
            raise RuntimeError("请先调用 configure_hard_selection。")

        all_names = self.site_names()
        for name in self._selected_sites:
            if name not in self.adapters:
                continue
            index = all_names.index(name)
            gate = self.fixed_gate_values[index]
            self.adapters[name].up.weight.data.mul_(gate)
            self.fixed_gate_values[index] = 1.0

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------
    def _apply_adapter(
        self,
        module_path: str,
        features: Tensor,
        gate_values: Tensor,
    ) -> Tensor:
        """在候选模块输出端进行一次门控残差相加。

        硬选择时，落选位置直接返回，Adapter 前向调用次数为零。
        """
        index = self._site_index.get(module_path)

        if index is None:
            return features

        site = self.sites[index]

        if site.name not in self._selected_sites:
            return features

        if site.name not in self.adapters:
            # 导出后模块被裁剪，但 selected 列表仍保留时按恒等处理。
            return features

        # Adapter 只生成 delta；此处保留原特征主路并施加一个位置标量，
        # 同一门控广播到整个 batch、所有通道和所有空间像素。
        delta = self.adapters[site.name](features)
        return features + gate_values[index] * delta

    def forward(
        self,
        packed_input: Tensor,
        timesteps: Tensor | None = None,
    ) -> Tensor:
        """遍历一次去噪网络，保留 Adapter 与门控的计算图。

        timesteps 为 None 时使用 fixed_timestep（部署单步协议）；
        否则必须是 [B] 的 long 张量，逐样本指定时间步（0..999）。
        """
        if not isinstance(packed_input, Tensor):
            raise TypeError("packed_input 必须是 Tensor。")

        if packed_input.ndim != 4:
            raise ValueError("输入必须为 [B, 12, H, W]。")

        batch, channels, height, width = packed_input.shape

        if channels != 12 or batch < 1:
            raise ValueError("输入必须包含非空 batch 和 12 个通道。")

        if (
            height < 8
            or width < 8
            or height % 8 != 0
            or width % 8 != 0
        ):
            raise ValueError("输入高宽必须是正的 8 的倍数。")

        if not packed_input.is_floating_point():
            raise TypeError("输入必须是浮点张量。")

        if packed_input.device != self.fixed_timestep.device:
            raise ValueError("输入和完整学生模型必须位于同一设备。")

        if timesteps is None:
            timesteps = self.fixed_timestep.expand(batch)
        else:
            if not isinstance(timesteps, Tensor):
                raise TypeError("timesteps 必须是 Tensor。")
            if timesteps.ndim != 1 or timesteps.shape[0] != batch:
                raise ValueError(f"timesteps 必须是 [{batch}]。")
            if timesteps.dtype != torch.long:
                raise TypeError("timesteps 必须是 long 张量。")
            if timesteps.device != self.fixed_timestep.device:
                raise ValueError("timesteps 与学生模型必须位于同一设备。")
            if timesteps.min().item() < 0 or timesteps.max().item() >= 1000:
                raise ValueError("timesteps 越界（0..999）。")

        # packed_input 的 H/J 是槽位名称；E3 调用方实际填入 0 和 J_t。
        # 网络只消费已拼好的 12 通道，不在内部重新计算动态物理条件。
        backbone = self.backbone
        time_embedding = backbone.time_emb(timesteps)

        # 整次前向只计算一组门控，保留 Tensor 梯度。
        if self._gate_override is not None:
            gate_values = torch.full(
                (len(self.sites),),
                self._gate_override,
                dtype=torch.float32,
                device=self.fixed_timestep.device,
            )
        elif self._use_fixed_gates:
            gate_values = self.fixed_gate_values
        else:
            gate_values = self.gates()
        features = backbone.image_proj(packed_input)
        skips = [features]

        for index, block in enumerate(backbone.down):
            features = block(features, time_embedding)
            features = self._apply_adapter(
                f"down.{index}", features, gate_values
            )
            # 保存修正后的特征，同时供主路和跳跃连接使用。
            skips.append(features)

        features = backbone.middle(features, time_embedding)
        features = self._apply_adapter(
            "middle", features, gate_values
        )

        for index, block in enumerate(backbone.up):
            if not isinstance(block, Upsample):
                skip = skips.pop()
                features = torch.cat((features, skip), dim=1)

            features = block(features, time_embedding)
            features = self._apply_adapter(
                f"up.{index}", features, gate_values
            )

        if skips:
            raise RuntimeError("跳跃连接未完整消费，请检查骨干结构。")

        # 直接预测 x0（恢复 RGB），不是 epsilon；评估时的 clamp 由外部完成。
        return backbone.final(backbone.act(features))

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------
    def deployment_payload(
        self,
        *,
        absorb_gates: bool = False,
        metadata: Mapping[str, object] | None = None,
    ) -> Dict[str, object]:
        """生成仅含选中分支的部署字典（不修改当前模型）。"""
        if not self._hard_selected:
            raise RuntimeError("导出前必须先配置硬选择。")

        all_names = self.site_names()
        adapters: Dict[str, Dict[str, Tensor]] = {}
        gates: Dict[str, float] = {}

        for name in self._selected_sites:
            if name not in self.adapters:
                raise RuntimeError(f"选中的位置 {name} 没有对应 Adapter。")
            state = {
                key: value.detach().cpu().clone()
                for key, value in self.adapters[name].state_dict().items()
            }
            gate = float(self.fixed_gate_values[all_names.index(name)].item())
            if absorb_gates:
                # g * up(x) == (g*up)(x)：这里在拷贝上吸收，不改原模型。
                state["up.weight"] = state["up.weight"] * gate
                gate = 1.0
            adapters[name] = state
            gates[name] = gate

        # 轻量部署包只含选中 Adapter、门控和结构信息，不含 E3 骨干/Beta。
        # 推理端必须另外加载与训练时相同的骨干及物理条件网络。
        return {
            "schema_version": 1,
            "kind": "seadiff_single_step_deployment",
            "selected_sites": list(self._selected_sites),
            "candidate_order": list(all_names),
            "fixed_gates": gates,
            "gates_absorbed": bool(absorb_gates),
            "adapter_reduction": int(self.adapter_reduction),
            "adapters": adapters,
            "fixed_timestep": int(self.fixed_timestep.item()),
            "metadata": dict(metadata or {}),
        }

    def save_deployment(
        self,
        path,
        *,
        absorb_gates: bool = False,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """原子写入部署字典。模型权重之外不包含优化器状态。"""
        import os
        from pathlib import Path

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.deployment_payload(
            absorb_gates=absorb_gates, metadata=metadata
        )
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, target)


def load_deployment_student(
    path,
    backbone: Denoise_UNet,
    *,
    device: str = "cpu",
) -> OneStepSeaDiffStudent:
    """从部署字典重建学生；只构造选中的 Adapter。"""
    from pathlib import Path

    payload = torch.load(
        Path(path).expanduser().resolve(strict=True),
        map_location="cpu",
        weights_only=True,
    )

    if (
        not isinstance(payload, dict)
        or payload.get("kind") != "seadiff_single_step_deployment"
        or payload.get("schema_version") != 1
    ):
        raise ValueError("不是受支持的部署权重格式。")

    selected = tuple(payload["selected_sites"])
    candidate_order = tuple(payload["candidate_order"])
    gates = dict(payload["fixed_gates"])
    adapters_state = payload["adapters"]

    # 先按完整候选结构装配并严格载入，再删除落选模块；最终仅执行入选分支。
    student = OneStepSeaDiffStudent(
        backbone,
        adapter_reduction=int(payload["adapter_reduction"]),
        fixed_timestep=int(payload["fixed_timestep"]),
    )

    if tuple(student.site_names()) != candidate_order:
        raise ValueError("部署权重的候选顺序与当前骨干不匹配。")

    for name in selected:
        if name not in student.adapters:
            raise ValueError(f"部署权重包含未知位置：{name}")
        student.adapters[name].load_state_dict(adapters_state[name], strict=True)

    student.configure_hard_selection(selected, gates)
    student.prune_unselected_adapters()
    student.freeze_unselected_adapters()
    student.to(device=device)
    student.eval()
    return student

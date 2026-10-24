"""渐进蒸馏的教师目标。

每个 batch 的流程：

1. 逐样本在 Student 网格上取一个跳跃区间 ``[t -> s]``；
2. ``x_t = sqrt(alpha_bar_t) * GT + sqrt(1 - alpha_bar_t) * epsilon``；
3. 教师从同一 ``x_t`` 出发，在教师网格上执行覆盖该区间的 DDIM 链，
   每步按当前 t 重建 ``J_t = U + w(t)(J - U)``；
4. 教师状态 ``x_s^T``：内部段为到达 s 的跨步状态；``s=0`` 时为教师
   t=0 前向输出 ``x0``；
5. 反解 ``x0`` 目标：
   ``x0_T = (x_s^T - (sigma_s/sigma_t) x_t) /
            (alpha_s - (sigma_s/sigma_t) alpha_t)``。

教师输出全部 detach；学生梯度只经过自身前向。
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from ..data.packing import pack_paper_order
from ..teacher.conditions import (
    canonical_alpha_bar,
    dynamic_physical_slots,
)

# 反解分母的相对稳定下界：|denom| >= 1e-4 * (sigma_s / sigma_t)。
MIN_DENOMINATOR_REL = 1e-4

DenoiseFn = Callable[[Tensor, Tensor], Tensor]
PackFn = Callable[[Tensor, Tuple[Tensor, ...], Tensor], Tensor]


def build_dynamic_physical_pack_fn(
    alpha_bar: Optional[Tensor] = None,
) -> PackFn:
    """构造 runner 的动态条件打包回调。"""
    schedule = (
        canonical_alpha_bar() if alpha_bar is None else alpha_bar
    ).detach().cpu().clone()

    def pack(
        state: Tensor,
        conditions: Tuple[Tensor, ...],
        time_batch: Tensor,
    ) -> Tensor:
        underwater, histogram, physical = conditions
        # physical 是本 batch 预先算好的 J；回调按当前链时间步重新构造 J_t。
        # 重建的是条件插值，教师链每一跳的噪声状态则由 DDIM 更新。
        return make_dynamic_physical_packed(
            state,
            time_batch,
            underwater,
            histogram,
            physical,
            schedule,
        )

    return pack


def make_dynamic_physical_packed(
    state: Tensor,
    timesteps: Tensor,
    underwater: Tensor,
    histogram: Tensor,
    physical: Tensor,
    alpha_bar: Tensor,
) -> Tensor:
    """拼接 12 通道输入 ``[x_t, U, 0, J_t]``。"""
    histogram_slot, physical_slot = dynamic_physical_slots(
        underwater,
        histogram,
        physical,
        timesteps=timesteps,
        alpha_bar=alpha_bar,
    )
    return pack_paper_order(
        state, underwater, histogram_slot, physical_slot
    )


@dataclass(frozen=True)
class ChainOutcome:
    """单样本的教师链结果。"""

    start: int
    end: int
    x_t: Tensor
    x_s_teacher: Tensor
    target_x0: Tensor
    denominator: Tensor
    chain_length: int
    teacher_forwards: int
    terminal: bool


class TeacherChainRunner:
    """在教师网格的子链上执行确定性 DDIM（eta=0）。"""

    def __init__(
        self,
        *,
        denoise_fn: DenoiseFn,
        make_packed: PackFn,
        alpha_bar: Optional[Tensor] = None,
        identity: Optional[Dict[str, object]] = None,
    ):
        self.denoise_fn = denoise_fn
        self.make_packed = make_packed
        self.alpha_bar = (
            canonical_alpha_bar()
            if alpha_bar is None
            else alpha_bar.detach().cpu().clone()
        )
        self.identity = dict(identity or {})

    def sqrt_alpha_bar(self, device: torch.device) -> Tensor:
        return self.alpha_bar.to(device=device).sqrt()

    def sqrt_one_minus(self, device: torch.device) -> Tensor:
        return (1.0 - self.alpha_bar).to(device=device).sqrt()

    @torch.no_grad()
    def run_chain(
        self,
        *,
        x_t: Tensor,
        chain: Sequence[int],
        conditions: Tuple[Tensor, ...],
    ) -> Tuple[Tensor, int, bool]:
        """沿 ``chain=[t, m1, ..., s]`` 执行教师链。

        返回 ``(x_s^T, 前向次数, 是否终点段)``。
        """
        if len(chain) < 2:
            raise ValueError("教师链至少需要起点与终点两个点。")

        device = x_t.device
        sqrt_alpha = self.sqrt_alpha_bar(device)
        sqrt_one_minus = self.sqrt_one_minus(device)
        state = x_t
        batch = x_t.shape[0]
        forwards = 0

        for index, timestep in enumerate(chain[:-1]):
            time_batch = torch.full(
                (batch,), timestep, dtype=torch.long, device=device
            )
            packed = self.make_packed(state, conditions, time_batch)
            predicted_x0 = self.denoise_fn(packed, time_batch)
            forwards += 1

            # 模型直接预测 x0，再由当前状态反推出 epsilon；eta=0 不另加随机噪声。
            next_timestep = chain[index + 1]
            predicted_noise = (
                state - sqrt_alpha[timestep] * predicted_x0
            ) / sqrt_one_minus[timestep]
            state = (
                sqrt_alpha[next_timestep] * predicted_x0
                + sqrt_one_minus[next_timestep] * predicted_noise
            )

        end = chain[-1]
        terminal = end == 0
        if terminal:
            # 到达离散 t=0 后还会再前向一次，返回这次 x0 预测，而非刚到达的状态。
            # 因此终端链前向数为 len(chain)，内部链为 len(chain)-1。
            time_batch = torch.full(
                (batch,), 0, dtype=torch.long, device=device
            )
            packed = self.make_packed(state, conditions, time_batch)
            state = self.denoise_fn(packed, time_batch)
            forwards += 1

        if not torch.isfinite(state).all().item():
            raise FloatingPointError("教师链状态包含 NaN 或 Inf。")
        return state, forwards, terminal

    @torch.no_grad()
    def inverse_target(
        self,
        *,
        x_t: Tensor,
        x_s_teacher: Tensor,
        start: int,
        end: int,
    ) -> Tuple[Tensor, Tensor]:
        """反解教师 x0 目标，并检查分母稳定性。"""
        device = x_t.device
        sqrt_alpha = self.sqrt_alpha_bar(device)
        sqrt_one_minus = self.sqrt_one_minus(device)

        alpha_t = sqrt_alpha[start]
        sigma_t = sqrt_one_minus[start]
        # alpha/sigma 指 sqrt(alpha_bar) / sqrt(1-alpha_bar)，不是 beta 调度值。
        # 即使 end=0，也沿用离散调度第0项；此处没有改成理想清洁端点 (1,0)。
        # 这与 run_chain 的终端额外 x0 前向共同定义当前代码的目标边界约定。
        alpha_s = sqrt_alpha[end]
        sigma_s = sqrt_one_minus[end]

        ratio = sigma_s / sigma_t
        denominator = alpha_s - ratio * alpha_t

        # 相近时间步会使反解分母很小；按噪声比设相对下界，避免放大数值误差。
        minimum = denominator.abs()
        floor = MIN_DENOMINATOR_REL * ratio
        if not torch.isfinite(minimum) or not (minimum >= floor).item():
            raise FloatingPointError(
                f"反解分母绝对值 {minimum.item():.3e} 低于相对下界 "
                f"{floor.item():.3e}（start={start}, end={end}）。"
            )

        target = (x_s_teacher - ratio * x_t) / denominator
        if not torch.isfinite(target).all().item():
            raise FloatingPointError(
                f"反解目标包含 NaN 或 Inf（start={start}, end={end}）。"
            )
        return target, denominator


class DistillationTargetProvider:
    """按当前教师/学生网格生成逐样本蒸馏目标。"""

    def __init__(
        self,
        *,
        runner: TeacherChainRunner,
        teacher_grid: Sequence[int],
        student_grid: Sequence[int],
        seed: int,
    ):
        if len(teacher_grid) < 2:
            raise ValueError("教师网格至少需要 2 个点。")
        if not student_grid:
            raise ValueError("学生网格不能为空。")

        self.runner = runner
        self.teacher_grid = tuple(teacher_grid)
        self.student_grid = tuple(student_grid)
        self.seed = int(seed)

        if len(self.student_grid) == 1:
            # G(1)：单点网格，对应部署跳跃 [999 -> 0]。
            self._intervals: Tuple[Tuple[int, int], ...] = (
                (self.student_grid[0], 0),
            )
        else:
            # N点网格有N-1个相邻训练区间；数字N不是本次样本的时间步t。
            self._intervals = tuple(
                zip(self.student_grid[:-1], self.student_grid[1:])
            )

    def intervals(self) -> Tuple[Tuple[int, int], ...]:
        return self._intervals

    def teacher_identity(self) -> Dict[str, object]:
        return dict(self.runner.identity)

    def interval_chain(self, start: int, end: int) -> List[int]:
        """教师网格中覆盖 ``[start -> end]`` 的链。"""
        chain = [
            point for point in self.teacher_grid if end < point <= start
        ] + [end]
        if chain[0] != start or chain[-1] != end:
            raise AssertionError("区间端点不在教师网格上。")
        return chain

    def sample_batch(
        self,
        *,
        underwater: Tensor,
        reference: Tensor,
        histogram: Tensor,
        physical: Tensor,
        batch_index: int,
        interval_indices: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """为一个 batch 生成 KD 目标；教师链与反解均在 no_grad 下执行。"""
        batch = underwater.shape[0]
        device = underwater.device

        if interval_indices is None:
            # 每张图独立选区间，所以同一 batch 可含不同 t；训练并非始终 t=999。
            interval_indices = torch.randint(
                len(self._intervals),
                (batch,),
                generator=self._make_generator(batch_index, device),
                device=device,
            )

        sqrt_alpha = self.runner.sqrt_alpha_bar(device)
        sqrt_one_minus = self.runner.sqrt_one_minus(device)

        outcomes: List[ChainOutcome] = []
        total_forwards = 0
        # 噪声生成器重新以同一批索引初始化；区间抽样未推进这个新生成器的状态。
        generator = self._make_generator(batch_index, device)

        for row in range(batch):
            start, end = self._intervals[int(interval_indices[row])]
            chain = self.interval_chain(start, end)

            sample_ref = reference[row : row + 1]
            noise = torch.randn(
                sample_ref.shape,
                generator=generator,
                device=device,
                dtype=sample_ref.dtype,
            )
            # 训练状态包含 GT 的前向加噪项；部署使用纯初始噪声，不需要 reference。
            x_t = (
                sqrt_alpha[start] * sample_ref
                + sqrt_one_minus[start] * noise
            )

            conditions = (
                underwater[row : row + 1],
                histogram[row : row + 1],
                physical[row : row + 1],
            )
            x_s_teacher, forwards, terminal = self.runner.run_chain(
                x_t=x_t,
                chain=chain,
                conditions=conditions,
            )
            total_forwards += forwards

            target, denominator = self.runner.inverse_target(
                x_t=x_t,
                x_s_teacher=x_s_teacher,
                start=start,
                end=end,
            )
            # 返回的教师状态/反解目标都作为常量；梯度仅经过之后的学生前向。
            outcomes.append(
                ChainOutcome(
                    start=start,
                    end=end,
                    x_t=x_t.detach(),
                    x_s_teacher=x_s_teacher.detach(),
                    target_x0=target.detach(),
                    denominator=denominator.detach(),
                    chain_length=len(chain),
                    teacher_forwards=forwards,
                    terminal=terminal,
                )
            )

        return {
            "timestep_start": torch.tensor(
                [o.start for o in outcomes],
                dtype=torch.long,
                device=device,
            ),
            "timestep_end": torch.tensor(
                [o.end for o in outcomes],
                dtype=torch.long,
                device=device,
            ),
            "x_t": torch.stack([o.x_t for o in outcomes], dim=0).squeeze(1),
            "x_s_teacher": torch.stack(
                [o.x_s_teacher for o in outcomes], dim=0
            ).squeeze(1),
            "teacher_target": torch.stack(
                [o.target_x0 for o in outcomes], dim=0
            ).squeeze(1),
            "denominator": torch.stack(
                [o.denominator for o in outcomes], dim=0
            ).reshape(batch, 1, 1, 1),
            "chain_length": torch.tensor(
                [o.chain_length for o in outcomes],
                dtype=torch.long,
                device=device,
            ),
            "teacher_forwards_total": total_forwards,
            "terminal_mask": torch.tensor(
                [o.terminal for o in outcomes],
                dtype=torch.bool,
                device=device,
            ),
            "interval_index": interval_indices,
            "teacher_identity": self.teacher_identity(),
        }

    def _make_generator(
        self, batch_index: int, device=None
    ) -> torch.Generator:
        generator = (
            torch.Generator(device=device)
            if device is not None
            else torch.Generator()
        )
        # 仅由配置seed与调用方批索引决定，独立于全局RNG；相同索引会重放同一随机流。
        generator.manual_seed(
            self.seed ^ (int(batch_index) * 0x9E3779B9) & 0xFFFFFFFF
        )
        return generator

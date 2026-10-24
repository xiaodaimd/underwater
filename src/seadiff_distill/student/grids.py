"""渐进蒸馏的嵌套时间网格。

从 G(1000) = [999, ..., 0] 开始，每级取父网格偶数下标点并保留终点 0：

    G(1000) -> G(500) -> ... -> G(4) -> G(2) -> G(1)

性质：
- ``len(G(N)) == N``；
- 除 G(1)=[999] 外，起点 999、终点 0；
- 学生网格完全嵌套在教师网格中；
- 学生每个区间 ``[t -> s]`` 的教师链由父网格中 ``s < p <= t`` 的点组成。
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

TRAIN_TIMESTEPS = 1000
# 这里记录网格点数；各级首点都为999，并非把实际t依次设成500、250等。
GRID_SIZES: Tuple[int, ...] = (1000, 500, 250, 125, 63, 32, 16, 8, 4, 2, 1)


def _coarsen(grid: Sequence[int]) -> List[int]:
    """取偶数下标点；多于一格时保留父网格终点。"""
    source = list(grid)
    if len(source) < 2:
        raise ValueError("父网格至少需要 2 个点。")

    coarse = source[0::2]
    if len(coarse) > 1 and coarse[-1] != source[-1]:
        # 替换最后一个抽样点以保留0，而非追加0；这样点数仍为ceil(N/2)。
        coarse[-1] = source[-1]
    if len(coarse) != (len(source) + 1) // 2:
        raise AssertionError("网格减半后的点数不符合 ceil(N/2)。")
    return coarse


def build_nested_grids() -> Dict[int, List[int]]:
    """返回 ``{1000: [...], 500: [...], ..., 1: [999]}``。"""
    grids: Dict[int, List[int]] = {
        TRAIN_TIMESTEPS: list(range(TRAIN_TIMESTEPS - 1, -1, -1))
    }
    for parent_size, size in zip(GRID_SIZES[:-1], GRID_SIZES[1:]):
        if size == 1:
            # G(1)只存起点；[999->0]训练区间由目标provider显式补出。
            grids[size] = [grids[parent_size][0]]
        else:
            grids[size] = _coarsen(grids[parent_size])
    return grids


@dataclass(frozen=True)
class IntervalCoverage:
    """一个学生区间对应的教师链覆盖。"""

    start: int
    end: int
    chain: Tuple[int, ...]
    teacher_updates: int
    strict_bisection: bool


def interval_coverage(
    teacher_grid: Sequence[int],
    student_grid: Sequence[int],
) -> Tuple[IntervalCoverage, ...]:
    """记录每个学生区间的教师链。"""
    teacher = list(teacher_grid)
    student = list(student_grid)

    if student[0] != teacher[0]:
        raise ValueError("学生网格起点必须与教师网格一致。")
    if any(point not in set(teacher) for point in student):
        raise ValueError("学生网格必须完全落在教师网格内。")

    records: List[IntervalCoverage] = []
    # 本函数只枚举相邻网格点；G(1)没有相邻对，终端单跳由provider处理。
    # teacher_updates计DDIM状态更新，未计终端t=0额外的去噪前向。
    for start, end in zip(student[:-1], student[1:]):
        chain = [
            point for point in teacher if end < point <= start
        ] + [end]
        updates = len(chain) - 1
        records.append(
            IntervalCoverage(
                start=start,
                end=end,
                chain=tuple(chain),
                teacher_updates=updates,
                strict_bisection=updates <= 2,
            )
        )
    return tuple(records)


def verify_nested_grids(
    grids: Dict[int, List[int]],
) -> Dict[str, object]:
    """全量校验网格结构，返回可写入日志的摘要。"""
    sizes = tuple(sorted(grids.keys(), reverse=True))
    if sizes != GRID_SIZES:
        raise AssertionError(f"网格大小集合不符：{sizes} vs {GRID_SIZES}")

    for size, grid in grids.items():
        if len(grid) != size:
            raise AssertionError(f"G({size}) 点数 {len(grid)} != {size}")
        if len(set(grid)) != len(grid):
            raise AssertionError(f"G({size}) 包含重复点")
        if any(not 0 <= point < TRAIN_TIMESTEPS for point in grid):
            raise AssertionError(f"G({size}) 包含越界点")
        if grid != sorted(grid, reverse=True):
            raise AssertionError(f"G({size}) 不是严格递减")
        if grid[0] != TRAIN_TIMESTEPS - 1:
            raise AssertionError(f"G({size}) 起点不是 999")
        if size > 1 and grid[-1] != 0:
            raise AssertionError(f"G({size}) 终点不是 0")
        if size == 1 and grid != [TRAIN_TIMESTEPS - 1]:
            raise AssertionError("G(1) 必须是单点 [999]")

    stages: Dict[str, object] = {}
    for parent_size, student_size in zip(sizes[:-1], sizes[1:]):
        parent = grids[parent_size]
        student = grids[student_size]
        missing = [p for p in student if p not in set(parent)]
        if missing:
            raise AssertionError(
                f"G({student_size}) 未嵌套在 G({parent_size})：{missing}"
            )

        coverages = interval_coverage(parent, student)
        for record in coverages:
            chain = record.chain
            if chain[0] != record.start or chain[-1] != record.end:
                raise AssertionError("覆盖链首尾与区间不一致。")
            if any(
                not record.end < point < record.start
                for point in chain[1:-1]
            ):
                raise AssertionError("覆盖链中间点未夹在区间内。")
            if record.teacher_updates < 1:
                raise AssertionError("每个学生区间至少覆盖 1 次教师更新。")

        stages[f"{parent_size}->{student_size}"] = {
            "teacher_grid_size": parent_size,
            "student_grid_size": student_size,
            "max_teacher_updates_per_interval": max(
                (r.teacher_updates for r in coverages), default=0
            ),
        }

    return {
        "schema_version": 1,
        "grid_sizes": list(GRID_SIZES),
        "stages": stages,
    }

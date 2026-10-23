"""Top-k 位置选择与稳定并列规则。"""

from typing import Sequence, Tuple


def select_top_k(
    site_names: Sequence[str],
    gate_values: Sequence[float],
    k: int,
) -> Tuple[str, ...]:
    """按门控值降序选 k 个位置；并列按 canonical 顺序（site_names 顺序）确定。

    返回位置名元组，长度恰为 k。
    """
    names = tuple(site_names)
    values = tuple(float(v) for v in gate_values)

    if len(names) != len(values):
        raise ValueError("site_names 与 gate_values 长度不一致。")
    if len(set(names)) != len(names):
        raise ValueError("site_names 不能重复。")
    if type(k) is not int or not 1 <= k <= len(names):
        raise ValueError("k 必须介于 1 与候选位置数之间。")
    if any(not (v == v) for v in values):  # NaN
        raise ValueError("门控值包含 NaN。")

    indexed = list(enumerate(zip(names, values)))
    indexed.sort(key=lambda item: (-item[1][1], item[0]))
    return tuple(item[1][0] for item in indexed[:k])

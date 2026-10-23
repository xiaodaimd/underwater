"""重建 400k 教师共用的确定性初始化。

同一 seed、同一构造顺序、同一代码会得到逐位相同的 state_dict。正式 run
从这份初始化开始训练，因此它是 400k 权重可复现的必要输入。脚本只保存
模型参数，不包含优化器、step 或 RNG 状态。
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from seadiff_distill.determinism import configure_determinism  # noqa: E402
from seadiff_distill.models.seadiff_unet import DocDiff  # noqa: E402

EXPECTED_SHA256 = (
    "f6c228b944a548edaf98a7194e294d767d99f9a068242f4572b83699e0dc00ec"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(seed: int, channels: int, ch_mults: tuple, n_blocks: int):
    configure_determinism(seed, strict=True)
    return DocDiff(
        6,
        3,
        channels,
        ch_mults,
        n_blocks,
        beta_use_norm=True,
    ).cpu().float()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="weights/shared_init/teacher_beta_norm_seed3407.pt",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument(
        "--ch-mults", type=int, nargs="+", default=[1, 2, 3, 4]
    )
    parser.add_argument("--n-blocks", type=int, default=1)
    parser.add_argument("--expected-sha256", default=EXPECTED_SHA256)
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    first = build(args.seed, args.channels, tuple(args.ch_mults), args.n_blocks)
    second = build(
        args.seed, args.channels, tuple(args.ch_mults), args.n_blocks
    )
    second_state = second.state_dict()
    differences = [
        name
        for name, value in first.state_dict().items()
        if not torch.equal(value, second_state[name])
    ]
    if differences:
        raise RuntimeError(f"两次重建不一致：{differences[:5]}")
    if any(
        value.dtype != torch.float32
        for value in first.state_dict().values()
    ):
        raise RuntimeError("state_dict 含非 FP32 参数。")

    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(first.state_dict(), temporary)
    os.replace(temporary, output)
    actual_sha = sha256_file(output)
    if actual_sha != args.expected_sha256:
        raise RuntimeError(
            "生成文件 SHA256 与正式 run 不一致，禁止继续：\n"
            f"期望 {args.expected_sha256}\n实际 {actual_sha}"
        )

    metadata = {
        "kind": "shared_teacher_initialization",
        "seed": args.seed,
        "architecture": {
            "input_channels": 6,
            "output_channels": 3,
            "channels": args.channels,
            "ch_mults": list(args.ch_mults),
            "n_blocks": args.n_blocks,
            "beta_use_norm": True,
        },
        "state_dict_sha256": actual_sha,
        "bytes": output.stat().st_size,
        "parameters": sum(p.numel() for p in first.parameters()),
        "rebuild_bitwise_identical": True,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""训练/推理脚本共用的参数与配置解析。"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from seadiff_distill.config import load_config, resolve_run_config  # noqa: E402


def add_common_arguments(
    parser: argparse.ArgumentParser, *, default_config: str
) -> None:
    parser.add_argument(
        "--config", default=default_config, help="YAML/JSON 配置文件"
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="覆盖 data.data_root；清单路径仍为相对路径",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto / cpu / cuda / cuda:0",
    )


def resolve_config(args: argparse.Namespace) -> Dict[str, Any]:
    """读取配置并把相对路径解析到仓库根目录。"""
    config = load_config(args.config)
    return resolve_run_config(
        config, root=ROOT, data_root=args.data_root
    )


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置了 CUDA 设备，但当前环境不可用。")
    return device


def training_seed(config: Dict[str, Any]) -> int:
    """训练脚本用 training.seed，其余脚本退回顶层 seed。"""
    training = config.get("training") or {}
    if "seed" in training:
        return int(training["seed"])
    return int(config.get("seed", 3407))

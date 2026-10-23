"""YAML/JSON 配置读取与路径解析。"""

import copy
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import yaml


class ConfigError(ValueError):
    """配置字段缺失或取值非法。"""


def _read(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件顶层必须是映射：{path}")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """递归合并，override 优先；不修改输入。"""
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(
    path: Union[str, Path],
    *,
    overrides: Union[None, Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """读取配置，并记录配置文件绝对路径。"""
    config_path = Path(path).expanduser().resolve(strict=True)
    config = _read(config_path)
    if overrides:
        config = deep_merge(config, overrides)
    config["_config_path"] = str(config_path)
    return config


def resolve_path(value: Union[str, Path], root: Union[str, Path]) -> Path:
    """解析配置中的路径；绝对路径保持不变。"""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path(root).expanduser() / path).resolve()


def resolve_run_config(
    config: Mapping[str, Any],
    *,
    root: Union[str, Path],
    data_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """把配置中的输入/输出路径转为绝对路径。

    ``root`` 通常是仓库根目录；``data_root`` 可在命令行覆盖，数据清单
    内部的路径仍保持相对于 ``data_root`` 的 POSIX 形式。
    """
    resolved = copy.deepcopy(dict(config))
    resolved["output_dir"] = str(
        resolve_path(resolved["output_dir"], root)
    )

    data_cfg = resolved["data"]
    data_value = data_root if data_root is not None else data_cfg.get(
        "data_root", "data"
    )
    data_cfg["data_root"] = str(resolve_path(data_value, root))
    for key in (
        "manifest",
        "inner_manifest",
        "dev_manifest",
    ):
        if key in data_cfg:
            data_cfg[key] = str(resolve_path(data_cfg[key], root))

    teacher_cfg = resolved.get("teacher")
    if teacher_cfg is not None:
        for key in ("denoiser_path", "beta_predictor_path"):
            if key in teacher_cfg:
                teacher_cfg[key] = str(
                    resolve_path(teacher_cfg[key], root)
                )

    model_cfg = resolved.get("model", {})
    for key in ("init_checkpoint", "denoiser_init", "beta_init"):
        if key in model_cfg:
            model_cfg[key] = str(resolve_path(model_cfg[key], root))

    inference_cfg = resolved.get("inference")
    if inference_cfg is not None:
        for key in ("denoiser", "beta_predictor"):
            if key in inference_cfg:
                inference_cfg[key] = str(
                    resolve_path(inference_cfg[key], root)
                )
    return resolved


def save_config_snapshot(config: Mapping[str, Any], path: Union[str, Path]) -> None:
    """把已解析配置写入 checkpoint 旁的快照文件。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in config.items() if not k.startswith("_resolved")}
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

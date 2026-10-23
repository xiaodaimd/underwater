"""E3 动态物理教师微调。

E3 只改条件槽位，不改网络结构与扩散过程：

- 从 400k 教师加载 denoiser 与 beta；
- beta 冻结（eval + requires_grad=False），只训练 denoiser；
- 10,000 iterations，batch 5，AdamW lr 1e-5，wd 1e-4；
- 每步 ``H_slot = 0``、``J_t = U + w(t)(J - U)``；
- 输入 ``[x_t, U, 0, J_t]``；
- 损失仍为 ``MSE + 0.1 * VGG19``。

``w(t) = 0.1 + 0.9 * (1 - alpha_bar_t)^2``，与 ``teacher.conditions`` 共用。
"""

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import torch

from ..data.dataset import SeaDiffDataset
from ..data.packing import INPUT_ORDER, pack_paper_order
from ..data.physical import compute_physical_image
from ..determinism import capture_rng_state, restore_rng_state
from ..diffusion.schedule import TRAIN_TIMESTEPS, TrainingDiffusion
from ..losses.perceptual import VGG19PerceptualLoss
from ..losses.teacher_loss import TeacherLoss
from ..models.seadiff_unet import DocDiff
from ..training.checkpoint import to_jsonable, write_json
from .conditions import (
    canonical_alpha_bar,
    describe,
    dynamic_physical_slots,
    dynamic_weight,
)

CHECKPOINT_KIND = "seadiff_e3_teacher_checkpoint"
SCHEMA_VERSION = 1
# 原始 E3 正式训练检查点使用的 kind；保留该值以支持官方 10k 训练续跑。
LEGACY_CHECKPOINT_KIND = "seadiff_condition_finetune_checkpoint"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_bytes(module: torch.nn.Module) -> bytes:
    chunks = []
    for name, parameter in module.state_dict().items():
        if parameter.dtype != torch.float32:
            raise ValueError(f"期望 FP32 权重：{name}")
        chunks.append(
            parameter.detach().cpu().contiguous().numpy().tobytes()
        )
    return b"".join(chunks)


class E3TeacherTrainer:
    """冻结 beta、只微调 denoiser 的 E3 教师训练器。"""

    def __init__(self, config: Mapping[str, Any], device: torch.device):
        self.config = dict(config)
        self.device = device

        data_cfg = config["data"]
        self.dataset = SeaDiffDataset(
            manifest_path=data_cfg["manifest"],
            data_root=data_cfg["data_root"],
            image_size=tuple(data_cfg.get("image_size", [336, 336])),
        )
        self.batch_size = int(data_cfg.get("batch_size", 5))
        if self.batch_size != 5:
            raise ValueError("E3 教师要求 batch_size=5。")
        self.steps_per_epoch = math.ceil(len(self.dataset) / self.batch_size)

        condition_cfg = dict(config.get("condition") or {})
        if condition_cfg.get("mode") != "dynamic_physical":
            raise ValueError(
                "E3 配置必须为 condition.mode=dynamic_physical。"
            )
        self.condition_spec = describe()
        self.alpha_bar = canonical_alpha_bar().to(device)

        model_cfg = config["model"]
        self.beta_use_norm = bool(model_cfg.get("beta_norm", True))
        if not self.beta_use_norm:
            raise ValueError("E3 教师必须启用 beta_norm。")
        self.network = DocDiff(
            input_channels=6,
            output_channels=3,
            n_channels=int(model_cfg.get("channels", 32)),
            ch_mults=tuple(model_cfg.get("ch_mults", [1, 2, 3, 4])),
            n_blocks=int(model_cfg.get("n_blocks", 1)),
            beta_use_norm=self.beta_use_norm,
        ).to(device).float()

        self.denoiser_init = Path(
            str(model_cfg["denoiser_init"]).strip()
        ).expanduser().resolve(strict=True)
        self.beta_init = Path(
            str(model_cfg["beta_init"]).strip()
        ).expanduser().resolve(strict=True)
        self.denoiser_init_sha256 = self._load_state(
            self.network.denoiser,
            self.denoiser_init,
            model_cfg.get("denoiser_init_sha256"),
            "denoiser",
        )
        self.beta_init_sha256 = self._load_state(
            self.network.beta_predictor,
            self.beta_init,
            model_cfg.get("beta_init_sha256"),
            "beta",
        )

        self.network.beta_predictor.eval()
        for parameter in self.network.beta_predictor.parameters():
            parameter.requires_grad_(False)
        self.beta_bytes = _tensor_bytes(self.network.beta_predictor)
        self.beta_content_sha256 = hashlib.sha256(
            self.beta_bytes
        ).hexdigest()

        self.diffusion = TrainingDiffusion(TRAIN_TIMESTEPS).to(device)
        training_cfg = config["training"]
        self.iterations = int(training_cfg.get("iterations", 10000))
        self.lr = float(training_cfg.get("lr", 1e-5))
        self.betas = tuple(
            float(v) for v in training_cfg.get("betas", [0.9, 0.999])
        )
        self.eps = float(training_cfg.get("eps", 1e-8))
        self.weight_decay = float(
            training_cfg.get("weight_decay", 1e-4)
        )
        self.perceptual_weight = float(
            training_cfg.get("perceptual_weight", 0.1)
        )
        if self.perceptual_weight != 0.1:
            raise ValueError("E3 感知损失权重必须为 0.1。")
        self.log_every = int(training_cfg.get("log_every", 50))
        self.save_every = int(training_cfg.get("save_every", 2000))
        self.input_order = str(
            training_cfg.get("input_order", INPUT_ORDER)
        )
        if self.input_order != INPUT_ORDER:
            raise ValueError(f"只允许 input_order={INPUT_ORDER}")

        self.perceptual = VGG19PerceptualLoss(device=device)
        self.loss_module = TeacherLoss(
            perceptual_weight=self.perceptual_weight,
            perceptual=self.perceptual,
        )
        denoiser_parameters = list(self.network.denoiser.parameters())
        if not denoiser_parameters:
            raise ValueError("Denoiser 没有参数。")
        self.optimizer = torch.optim.AdamW(
            denoiser_parameters,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )

        self.output_dir = Path(config["output_dir"])
        self.weights_dir = self.output_dir / "weights"
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        self.base_seed = int(config.get("seed", 3407))
        self.data_identity = {
            "manifest": str(Path(data_cfg["manifest"]).resolve()),
            "manifest_sha256": _sha256_file(
                Path(data_cfg["manifest"]).resolve()
            ),
            "samples": len(self.dataset),
            "batch_size": self.batch_size,
            "steps_per_epoch": self.steps_per_epoch,
            "dataset": self.dataset.describe(),
        }

        self.w_anchors = {
            str(t): float(dynamic_weight(self.alpha_bar[t]).item())
            for t in (0, 500, 999)
        }

        self.iteration = 0
        self.epoch = 0
        self.position = 0
        self.history: List[Dict[str, Any]] = []
        self.started = time.time()

    def _load_state(
        self,
        module: torch.nn.Module,
        path: Path,
        expected_sha: Optional[str],
        label: str,
    ) -> str:
        actual = _sha256_file(path)
        if expected_sha and actual != str(expected_sha):
            raise ValueError(
                f"{label} 初始化权重 SHA256 不匹配："
                f"{actual} vs {expected_sha}"
            )
        module.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True),
            strict=True,
        )
        return actual

    # ------------------------------------------------------------------
    # 数据
    # ------------------------------------------------------------------
    def _permutation(self, epoch: int) -> List[int]:
        generator = torch.Generator()
        generator.manual_seed(self.base_seed + epoch)
        return torch.randperm(
            len(self.dataset), generator=generator
        ).tolist()

    def _fetch(self, indices: List[int]) -> Dict[str, Any]:
        samples = [self.dataset[index] for index in indices]
        return {
            "sample_id": [s["sample_id"] for s in samples],
            "underwater": torch.stack(
                [s["underwater"] for s in samples]
            ).to(self.device),
            "reference": torch.stack(
                [s["reference"] for s in samples]
            ).to(self.device),
            "depth": torch.stack([s["depth"] for s in samples]).to(
                self.device
            ),
            "histogram": torch.stack(
                [s["histogram"] for s in samples]
            ).to(self.device),
        }

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------
    def _train_step(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        self.network.train()
        # 即使外层切到 train，冻结 beta 始终保持 eval。
        self.network.beta_predictor.eval()
        self.optimizer.zero_grad(set_to_none=True)

        underwater = batch["underwater"]
        reference = batch["reference"]
        depth = batch["depth"]
        histogram = batch["histogram"]

        with torch.no_grad():
            predicted_beta, physical, preclamp = compute_physical_image(
                underwater=underwater,
                depth=depth,
                beta_predictor=self.network.beta_predictor,
                return_preclamp=True,
            )

        timesteps = torch.randint(
            0, TRAIN_TIMESTEPS, (underwater.shape[0],), device=self.device
        ).long()
        noisy, _ = self.diffusion.noisy_image(timesteps, reference)

        histogram_slot, physical_slot = dynamic_physical_slots(
            underwater,
            histogram,
            physical,
            timesteps=timesteps,
            alpha_bar=self.alpha_bar,
        )
        packed = pack_paper_order(
            noisy, underwater, histogram_slot, physical_slot
        )
        prediction = self.network.denoiser(packed, timesteps)

        loss, parts = self.loss_module(prediction, reference)
        loss.backward()

        beta_grad_norm = math.sqrt(
            sum(
                float(parameter.grad.detach().pow(2).sum().item())
                for parameter in self.network.beta_predictor.parameters()
                if parameter.grad is not None
            )
        )
        denoiser_grad_norm = math.sqrt(
            sum(
                float(parameter.grad.detach().pow(2).sum().item())
                for parameter in self.network.denoiser.parameters()
                if parameter.grad is not None
            )
        )
        if self.iteration + 1 == 1 and not denoiser_grad_norm > 0.0:
            raise RuntimeError(
                "第 1 个 iteration 的 Denoiser 梯度范数不大于 0，"
                "拒绝继续训练。"
            )
        self.optimizer.step()

        weight = dynamic_weight(self.alpha_bar[timesteps])
        return {
            "loss": float(loss.detach().item()),
            "mse": float(parts["mse"].detach().item()),
            "perceptual": float(parts["perceptual"].detach().item()),
            "pred_min": float(prediction.detach().min().item()),
            "pred_max": float(prediction.detach().max().item()),
            "denoiser_grad_norm": denoiser_grad_norm,
            "beta_grad_norm": beta_grad_norm,
            "beta_frozen": bool(beta_grad_norm == 0.0),
            "physical_min": float(physical.min().item()),
            "physical_max": float(physical.max().item()),
            "physical_clamp_fraction": float(
                ((preclamp < 0.0) | (preclamp > 1.0))
                .float()
                .mean()
                .item()
            ),
            "beta_mean": float(predicted_beta.mean().item()),
            "w_min": float(weight.min().item()),
            "w_max": float(weight.max().item()),
            "w_mean": float(weight.mean().item()),
            "j_t_min": float(physical_slot.min().item()),
            "j_t_max": float(physical_slot.max().item()),
            "gpu_peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(self.device))
                if self.device.type == "cuda" else None
            ),
        }

    # ------------------------------------------------------------------
    # checkpoint
    # ------------------------------------------------------------------
    def _payload(self) -> Dict[str, Any]:
        return {
            "kind": CHECKPOINT_KIND,
            "schema_version": SCHEMA_VERSION,
            "iteration": self.iteration,
            "epoch": self.epoch,
            "position": self.position,
            "mode": "dynamic_physical",
            "condition_spec": self.condition_spec,
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": to_jsonable(self.config),
            "input_order": self.input_order,
            "batch_size": self.batch_size,
            "beta_norm": self.beta_use_norm,
            "denoiser_init": str(self.denoiser_init),
            "denoiser_init_sha256": self.denoiser_init_sha256,
            "beta_init": str(self.beta_init),
            "beta_init_sha256": self.beta_init_sha256,
            "beta_content_sha256": self.beta_content_sha256,
            "data_identity": self.data_identity,
            "w_anchors": self.w_anchors,
            "rng": capture_rng_state(),
            "history": self.history[-200:],
            "perceptual": self.perceptual.describe(),
            "diffusion": {
                "train_timesteps": TRAIN_TIMESTEPS,
                "schedule": "linear",
                "beta_start": 1e-6,
                "beta_end": 0.02,
                "prediction_type": "x0",
            },
        }

    def _save(self, filename: str) -> Path:
        path = self.output_dir / filename
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(self._payload(), temporary)
        os.replace(temporary, path)
        return path

    def _verify_beta_frozen(self) -> None:
        current = _tensor_bytes(self.network.beta_predictor)
        if current != self.beta_bytes:
            raise RuntimeError("Beta 权重在训练中发生了变化，冻结被破坏。")

    def _save_weights(self) -> Dict[str, Any]:
        self._verify_beta_frozen()
        denoiser_path = (
            self.weights_dir / f"model_denoiser_{self.iteration}.pth"
        )
        beta_path = (
            self.weights_dir / f"model_beta_frozen_{self.iteration}.pth"
        )
        torch.save(self.network.denoiser.state_dict(), denoiser_path)
        torch.save(self.network.beta_predictor.state_dict(), beta_path)
        record = {
            "iteration": self.iteration,
            "mode": "dynamic_physical",
            "denoiser": {
                "path": str(denoiser_path),
                "sha256": _sha256_file(denoiser_path),
            },
            "beta_frozen": {
                "path": str(beta_path),
                "sha256": _sha256_file(beta_path),
                "content_sha256": self.beta_content_sha256,
            },
            "denoiser_init_sha256": self.denoiser_init_sha256,
        }
        with (self.output_dir / "weights_manifest.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def _load(self, path: str | Path) -> None:
        payload = torch.load(
            Path(path).expanduser().resolve(strict=True),
            map_location="cpu",
            weights_only=False,
        )
        if payload.get("kind") not in (
            CHECKPOINT_KIND,
            LEGACY_CHECKPOINT_KIND,
        ):
            raise ValueError("不是 E3 教师检查点。")
        if (
            payload.get("kind") == CHECKPOINT_KIND
            and payload.get("schema_version") != SCHEMA_VERSION
        ):
            raise ValueError("E3 检查点版本不支持。")
        if payload.get("mode") != "dynamic_physical":
            raise ValueError("检查点不是 dynamic_physical 模式。")
        if int(payload.get("batch_size", -1)) != self.batch_size:
            raise ValueError("检查点 batch_size 与当前配置不一致。")
        if bool(payload.get("beta_norm", False)) != self.beta_use_norm:
            raise ValueError("检查点 beta_norm 与当前配置不一致。")
        if payload.get("denoiser_init_sha256") != self.denoiser_init_sha256:
            raise ValueError("检查点 denoiser 初始化与当前配置不一致。")
        if payload.get("beta_init_sha256") != self.beta_init_sha256:
            raise ValueError("检查点 beta 初始化与当前配置不一致。")

        self.network.load_state_dict(payload["network"], strict=True)
        self.network.to(self.device)
        self.network.beta_predictor.eval()
        for parameter in self.network.beta_predictor.parameters():
            parameter.requires_grad_(False)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.iteration = int(payload["iteration"])
        self.epoch = int(payload["epoch"])
        self.position = int(payload["position"])
        self.history = list(payload.get("history", []))
        if payload.get("rng") is not None:
            restore_rng_state(payload["rng"])

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def train(
        self,
        *,
        resume: Optional[str] = None,
        max_iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        if resume:
            self._load(resume)
        target = (
            int(max_iterations)
            if max_iterations is not None
            else self.iterations
        )
        if target > self.iterations:
            raise ValueError("max_iterations 不能超过配置预算。")
        if target < self.iteration:
            raise ValueError("target 小于已完成 iteration。")

        while self.iteration < target:
            permutation = self._permutation(self.epoch)
            batch_count = math.ceil(len(permutation) / self.batch_size)
            if self.position >= batch_count:
                self.epoch += 1
                self.position = 0
                continue

            for position in range(self.position, batch_count):
                if self.iteration >= target:
                    break
                indices = permutation[
                    position * self.batch_size:(position + 1) * self.batch_size
                ]
                batch = self._fetch(indices)
                logs = self._train_step(batch)
                self.iteration += 1
                self.position = position + 1

                if self.iteration == 1:
                    first = {
                        "event": "first_update",
                        "iteration": 1,
                        "epoch": self.epoch,
                        "position": self.position,
                        "elapsed_seconds": round(
                            time.time() - self.started, 3
                        ),
                        **logs,
                    }
                    self.history.append(first)
                    with (self.output_dir / "events.jsonl").open(
                        "a", encoding="utf-8"
                    ) as handle:
                        handle.write(
                            json.dumps(first, ensure_ascii=False) + "\n"
                        )
                    write_json(self.output_dir / "first_update.json", first)

                if self.log_every and self.iteration % self.log_every == 0:
                    record = {
                        "event": "train",
                        "iteration": self.iteration,
                        "epoch": self.epoch,
                        "position": self.position,
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "elapsed_seconds": round(
                            time.time() - self.started, 1
                        ),
                        **logs,
                    }
                    self.history.append(record)
                    with (self.output_dir / "events.jsonl").open(
                        "a", encoding="utf-8"
                    ) as handle:
                        handle.write(
                            json.dumps(record, ensure_ascii=False) + "\n"
                        )

                if self.save_every and self.iteration % self.save_every == 0:
                    self._save("finetune_last.pt")
                    self._save_weights()
                    print(
                        f"[e3-teacher] iteration {self.iteration} "
                        "checkpoint saved",
                        flush=True,
                    )

            if self.position >= batch_count:
                self.epoch += 1
                self.position = 0

        full = self.iteration == self.iterations
        self._save("finetune_last.pt")
        weights = self._save_weights()
        if full:
            self._save("finetune_final.pt")
        summary = {
            "status": "completed" if full else "partial_not_formal",
            "mode": "dynamic_physical",
            "iterations_completed": self.iteration,
            "configured_iterations": self.iterations,
            "epoch": self.epoch,
            "position": self.position,
            "final_weights": weights,
            "denoiser_init": str(self.denoiser_init),
            "denoiser_init_sha256": self.denoiser_init_sha256,
            "beta_init": str(self.beta_init),
            "beta_init_sha256": self.beta_init_sha256,
            "beta_content_sha256": self.beta_content_sha256,
            "beta_frozen_verified": True,
            "w_anchors": self.w_anchors,
            "condition_spec": self.condition_spec,
            "input_order": self.input_order,
            "batch_size": self.batch_size,
            "elapsed_seconds": round(time.time() - self.started, 1),
            "data_identity": self.data_identity,
            "perceptual": self.perceptual.describe(),
        }
        write_json(self.output_dir / "finetune_summary.json", summary)
        return summary

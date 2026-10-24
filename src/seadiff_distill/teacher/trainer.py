"""SeaDiff 400k 教师训练。

固定协议：
- UIEB 600 对训练图，batch 5，400,000 iterations；
- AdamW，lr 1e-4，betas (0.9, 0.999)，eps 1e-8，weight decay 1e-4；
- 输入 ``[x_t, U, H, J]``，12 通道；
- 损失 ``MSE + 0.1 * VGG19(relu1_2/relu2_2/relu3_3)``；
- 线性 beta 日程，T=1000，x0 预测。

Beta 网络与去噪网络一起训练；``beta_norm=True`` 是官方实现上的数值
稳定性补充（logits 夹紧到 ±8，反向直通）。
"""

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import torch
from torch import Tensor

from ..data.dataset import SeaDiffDataset
from ..data.packing import INPUT_ORDER, pack_paper_order
from ..data.physical import compute_physical_image
from ..determinism import capture_rng_state, restore_rng_state
from ..diffusion.schedule import TRAIN_TIMESTEPS, TrainingDiffusion
from ..losses.perceptual import VGG19PerceptualLoss
from ..losses.teacher_loss import TeacherLoss
from ..models.seadiff_unet import DocDiff
from ..training.checkpoint import to_jsonable, write_json

CHECKPOINT_KIND = "seadiff_teacher_checkpoint"
SCHEMA_VERSION = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TeacherTrainer:
    """400k 教师训练器（单进程、FP32、可断点恢复）。"""

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
            raise ValueError("400k 教师要求 batch_size=5。")
        self.steps_per_epoch = math.ceil(len(self.dataset) / self.batch_size)

        model_cfg = config["model"]
        self.beta_use_norm = bool(model_cfg.get("beta_norm", True))
        if not self.beta_use_norm:
            raise ValueError("教师必须启用 beta_norm。")
        self.network = DocDiff(
            input_channels=6,
            output_channels=3,
            n_channels=int(model_cfg.get("channels", 32)),
            ch_mults=tuple(model_cfg.get("ch_mults", [1, 2, 3, 4])),
            n_blocks=int(model_cfg.get("n_blocks", 1)),
            beta_use_norm=self.beta_use_norm,
        ).to(device).float()

        # 可选共享初始化：只加载 state_dict，不加载优化器或步数。
        self.init_checkpoint = None
        self.init_sha256 = None
        init_value = model_cfg.get("init_checkpoint")
        if init_value:
            init_path = Path(str(init_value)).expanduser().resolve(strict=True)
            actual_sha = _sha256_file(init_path)
            expected_sha = model_cfg.get("init_checkpoint_sha256")
            if expected_sha and actual_sha != str(expected_sha):
                raise ValueError(
                    "init_checkpoint SHA256 不匹配："
                    f"{actual_sha} vs {expected_sha}"
                )
            init_state = torch.load(
                init_path, map_location="cpu", weights_only=True
            )
            if (
                isinstance(init_state, dict)
                and "network" in init_state
                and isinstance(init_state["network"], dict)
            ):
                init_state = init_state["network"]
            self.network.load_state_dict(init_state, strict=True)
            self.init_checkpoint = str(init_path)
            self.init_sha256 = actual_sha

        self.diffusion = TrainingDiffusion(TRAIN_TIMESTEPS).to(device)
        training_cfg = config["training"]
        self.iterations = int(training_cfg.get("iterations", 400000))
        self.lr = float(training_cfg.get("lr", 1e-4))
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
            raise ValueError("教师感知损失权重必须为 0.1。")
        self.input_order = str(
            training_cfg.get("input_order", INPUT_ORDER)
        )
        if self.input_order != INPUT_ORDER:
            raise ValueError(f"只允许 input_order={INPUT_ORDER}")

        self.log_every = int(training_cfg.get("log_every", 50))
        self.save_every = int(training_cfg.get("save_every", 50000))

        self.perceptual = VGG19PerceptualLoss(device=device)
        self.loss_module = TeacherLoss(
            perceptual_weight=self.perceptual_weight,
            perceptual=self.perceptual,
        )
        # 原始 400k 训练同时优化 Beta 与 Denoiser；E3 微调另有独立训练器。
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
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

        self.iteration = 0
        self.epoch = 0
        self.position = 0
        self.history: List[Dict[str, Any]] = []
        self.started = time.time()
        # Beta 梯度连续为 0 时立即停止，避免静默训练死分支。
        self.beta_grad_zero_streak = 0
        self.beta_grad_zero_limit = 200

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
    def _train_step(self, batch: Mapping[str, Any]) -> Dict[str, float]:
        self.network.train()
        self.optimizer.zero_grad(set_to_none=True)

        underwater = batch["underwater"]
        reference = batch["reference"]
        depth = batch["depth"]
        histogram = batch["histogram"]

        # 此处必须保留计算图：损失→Denoiser→J→Beta，才能联合学习物理衰减。
        # 深度/直方图是输入 U 的条件；reference 不参与 J 的生成。
        predicted_beta, physical, preclamp = compute_physical_image(
            underwater=underwater,
            depth=depth,
            beta_predictor=self.network.beta_predictor,
            return_preclamp=True,
        )
        # 从清晰参考图 Y 生成随机时间步 x_t，模拟不同噪声水平的训练状态。
        timesteps = torch.randint(
            0, TRAIN_TIMESTEPS, (underwater.shape[0],), device=self.device
        ).long()
        noisy, _ = self.diffusion.noisy_image(timesteps, reference)
        packed = pack_paper_order(
            noisy, underwater, histogram, physical
        )
        # 骨干预测 x0_hat，直接与 Y 计算图像 MSE 和 VGG19 特征损失。
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
        self._guard_beta_gradient(predicted_beta, preclamp, beta_grad_norm)
        self.optimizer.step()

        beta_values = predicted_beta.detach()
        clamp_low = (preclamp.detach() < 0.0).float().mean()
        clamp_high = (preclamp.detach() > 1.0).float().mean()
        return {
            "loss": float(loss.detach().item()),
            "mse": float(parts["mse"].detach().item()),
            "perceptual": float(parts["perceptual"].detach().item()),
            "beta_min": float(beta_values.min().item()),
            "beta_max": float(beta_values.max().item()),
            "beta_mean": float(beta_values.mean().item()),
            "physical_clamp_fraction": float(
                (clamp_low + clamp_high).item()
            ),
            "beta_grad_norm": beta_grad_norm,
            "denoiser_grad_norm": denoiser_grad_norm,
            "gpu_peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(self.device))
                if self.device.type == "cuda" else None
            ),
        }

    def _guard_beta_gradient(
        self,
        predicted_beta: Tensor,
        preclamp: Tensor,
        beta_grad_norm: float,
    ) -> None:
        # 此诊断适用于应被训练的基础 Beta；E3 冻结 Beta 时零梯度是预期行为。
        if beta_grad_norm == 0.0:
            self.beta_grad_zero_streak += 1
        else:
            self.beta_grad_zero_streak = 0
        if self.beta_grad_zero_streak < self.beta_grad_zero_limit:
            return

        beta_values = predicted_beta.detach()
        dead = {
            "event": "beta_grad_dead",
            "iteration": self.iteration + 1,
            "streak": self.beta_grad_zero_streak,
            "beta_min": float(beta_values.min().item()),
            "beta_max": float(beta_values.max().item()),
            "beta_mean": float(beta_values.mean().item()),
            "physical_clamp_fraction": float(
                ((preclamp.detach() < 0.0) | (preclamp.detach() > 1.0))
                .float()
                .mean()
                .item()
            ),
        }
        write_json(self.output_dir / "beta_grad_dead.json", dead)
        raise RuntimeError(
            "Beta 梯度连续 200 步精确为 0，Beta 子网已停止更新。"
        )

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
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": to_jsonable(self.config),
            "input_order": self.input_order,
            "batch_size": self.batch_size,
            "beta_norm": self.beta_use_norm,
            "init_checkpoint": self.init_checkpoint,
            "init_sha256": self.init_sha256,
            "data_identity": self.data_identity,
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

    def _save_weights(self) -> Dict[str, Any]:
        beta_path = (
            self.weights_dir / f"model_beta_predictor_{self.iteration}.pth"
        )
        denoiser_path = (
            self.weights_dir / f"model_denoiser_{self.iteration}.pth"
        )
        torch.save(self.network.beta_predictor.state_dict(), beta_path)
        torch.save(self.network.denoiser.state_dict(), denoiser_path)
        record = {
            "iteration": self.iteration,
            "beta_predictor": {
                "path": str(beta_path),
                "sha256": _sha256_file(beta_path),
            },
            "denoiser": {
                "path": str(denoiser_path),
                "sha256": _sha256_file(denoiser_path),
            },
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
            "seadiff_teacher_paper_checkpoint",
        ):
            raise ValueError("不是教师检查点。")
        if (
            payload.get("kind") == CHECKPOINT_KIND
            and payload.get("schema_version") != SCHEMA_VERSION
        ):
            raise ValueError("教师检查点版本不支持。")
        if payload.get("input_order") != self.input_order:
            raise ValueError("检查点 input_order 与当前配置不一致。")
        if int(payload.get("batch_size", -1)) != self.batch_size:
            raise ValueError("检查点 batch_size 与当前配置不一致。")
        if bool(payload.get("beta_norm", False)) != self.beta_use_norm:
            raise ValueError("检查点 beta_norm 与当前配置不一致。")

        self.network.load_state_dict(payload["network"], strict=True)
        self.network.to(self.device)
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.iteration = int(payload["iteration"])
        self.epoch = int(payload.get("epoch", 0))
        self.position = int(payload.get("position", 0))
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
                    self._save("teacher_last.pt")
                    self._save_weights()
                    print(
                        f"[teacher] iteration {self.iteration} checkpoint saved",
                        flush=True,
                    )

            if self.position >= batch_count:
                self.epoch += 1
                self.position = 0

        self._save("teacher_last.pt")
        weights = self._save_weights()
        summary = {
            "iterations_completed": self.iteration,
            "configured_iterations": self.iterations,
            "epoch": self.epoch,
            "position": self.position,
            "final_weights": weights,
            "input_order": self.input_order,
            "batch_size": self.batch_size,
            "init_checkpoint": self.init_checkpoint,
            "init_sha256": self.init_sha256,
            "elapsed_seconds": round(time.time() - self.started, 1),
            "data_identity": self.data_identity,
            "perceptual": self.perceptual.describe(),
        }
        write_json(self.output_dir / "teacher_summary.json", summary)
        return summary

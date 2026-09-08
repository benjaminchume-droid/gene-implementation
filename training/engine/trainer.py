from __future__ import annotations

import math
import random
import time
from pathlib import Path

import torch

from .batching import TokenBatcher
from .checkpoint import TrainingCheckpoint
from .config import TrainingConfig
from .dataset import JSONLTextDataset


class GeneTrainer:

    def __init__(
        self,
        model,
        tokenizer,
        config: TrainingConfig | None = None,
        device: str | None = None,
    ) -> None:

        self.model = model
        self.tokenizer = tokenizer

        self.config = config or TrainingConfig()
        self.config.validate()
        self.config.prepare_dirs()

        self.device = (
            device
            or (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.model.to(self.device)

        self.batch_builder = TokenBatcher(
            tokenizer=tokenizer,
            max_sequence_length=
                self.config.max_sequence_length,
        )

        self.checkpoints = TrainingCheckpoint(
            self.config.output_dir
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = self._build_scheduler()

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(
                self.config.mixed_precision
                and self.device == "cuda"
            ),
        )

        self.step = 0
        self.epoch = 0
        self.best_validation_loss = None

        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

    def _build_scheduler(self):
        def lr_lambda(step: int):
            if step < self.config.warmup_steps:
                return (
                    float(step + 1)
                    / max(
                        1,
                        self.config.warmup_steps,
                    )
                )

            if self.config.max_steps is None:
                return 1.0

            progress = (
                step
                - self.config.warmup_steps
            ) / max(
                1,
                self.config.max_steps
                - self.config.warmup_steps,
            )

            progress = max(
                0.0,
                min(1.0, progress),
            )

            return (
                0.5
                * (
                    1.0
                    + math.cos(
                        math.pi * progress
                    )
                )
            )

        return torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda,
        )

    def _prepare_batch(self, batch):
        input_ids = batch[
            "input_ids"
        ].to(self.device)

        attention_mask = batch[
            "attention_mask"
        ].to(self.device)

        labels = input_ids.clone()

        labels[
            attention_mask == 0
        ] = -100

        return (
            input_ids,
            attention_mask,
            labels,
        )

    def train_step(self, batch) -> float:
        input_ids, attention_mask, labels = (
            self._prepare_batch(batch)
        )

        use_amp = (
            self.config.mixed_precision
            and self.device == "cuda"
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs["loss"]

            scaled_loss = (
                loss
                / self.config.gradient_accumulation_steps
            )

        self.scaler.scale(
            scaled_loss
        ).backward()

        return float(
            loss.detach().cpu()
        )

    def optimizer_step(self):
        self.scaler.unscale_(
            self.optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )

        self.scaler.step(
            self.optimizer
        )

        self.scaler.update()

        self.optimizer.zero_grad(
            set_to_none=True
        )

        self.scheduler.step()
        self.step += 1

    def evaluate(
        self,
        validation_file: str,
        max_batches: int | None = None,
    ) -> dict:

        self.model.eval()

        dataset = JSONLTextDataset(
            validation_file
        )

        losses = []

        with torch.no_grad():
            for index, batch in enumerate(
                self.batch_builder.batch(
                    dataset,
                    self.config.batch_size,
                )
            ):

                input_ids, attention_mask, labels = (
                    self._prepare_batch(batch)
                )

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                losses.append(
                    float(
                        outputs["loss"]
                        .detach()
                        .cpu()
                    )
                )

                if (
                    max_batches is not None
                    and index + 1 >= max_batches
                ):
                    break

        self.model.train()

        if not losses:
            return {
                "loss": None,
                "perplexity": None,
                "batches": 0,
            }

        average = sum(losses) / len(losses)

        return {
            "loss": average,
            "perplexity": math.exp(
                min(average, 20)
            ),
            "batches": len(losses),
        }

    def train(
        self,
        train_file: str,
        validation_file: str | None = None,
    ) -> dict:

        start = time.time()

        dataset = JSONLTextDataset(
            train_file
        )

        self.model.train()

        self.optimizer.zero_grad(
            set_to_none=True
        )

        history = []

        batch_count = 0

        for epoch in range(
            self.config.epochs
        ):

            self.epoch = epoch

            for batch in self.batch_builder.batch(
                dataset,
                self.config.batch_size,
            ):

                loss = self.train_step(
                    batch
                )

                batch_count += 1

                if (
                    batch_count
                    % self.config.gradient_accumulation_steps
                    == 0
                ):
                    self.optimizer_step()

                    record = {
                        "step": self.step,
                        "epoch": epoch,
                        "loss": loss,
                        "learning_rate":
                            self.optimizer.param_groups[0]["lr"],
                    }

                    history.append(record)

                    if (
                        self.step
                        % self.config.log_interval
                        == 0
                    ):
                        print(record)

                    if (
                        validation_file
                        and self.step
                        % self.config.validation_interval
                        == 0
                    ):

                        validation = self.evaluate(
                            validation_file
                        )

                        if (
                            validation["loss"]
                            is not None
                            and (
                                self.best_validation_loss
                                is None
                                or validation["loss"]
                                < self.best_validation_loss
                            )
                        ):
                            self.best_validation_loss = (
                                validation["loss"]
                            )

                    if (
                        self.step
                        % self.config.checkpoint_interval
                        == 0
                    ):
                        self.checkpoints.save(
                            self.model,
                            self.optimizer,
                            self.scheduler,
                            self.scaler,
                            epoch,
                            self.step,
                            self.best_validation_loss,
                            self.config,
                        )

                    if (
                        self.config.max_steps is not None
                        and self.step
                        >= self.config.max_steps
                    ):
                        break

            if (
                self.config.max_steps is not None
                and self.step
                >= self.config.max_steps
            ):
                break

        final_checkpoint = (
            self.checkpoints.save(
                self.model,
                self.optimizer,
                self.scheduler,
                self.scaler,
                self.epoch,
                self.step,
                self.best_validation_loss,
                self.config,
            )
        )

        result = {
            "success": True,
            "steps": self.step,
            "epochs": self.epoch + 1,
            "elapsed_seconds":
                time.time() - start,
            "final_checkpoint":
                final_checkpoint,
            "best_validation_loss":
                self.best_validation_loss,
            "history": history,
        }

        run_dir = Path(
            self.config.run_dir
        )

        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (run_dir / f"run-{self.step}.json").write_text(
            __import__("json").dumps(
                result,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        return result

    def resume(
        self,
        checkpoint_path: str,
    ) -> dict:

        payload = self.checkpoints.load(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            map_location=self.device,
        )

        self.epoch = payload.get(
            "epoch",
            0,
        )

        self.step = payload.get(
            "step",
            0,
        )

        self.best_validation_loss = payload.get(
            "best_validation_loss"
        )

        return {
            "success": True,
            "epoch": self.epoch,
            "step": self.step,
            "best_validation_loss":
                self.best_validation_loss,
        }

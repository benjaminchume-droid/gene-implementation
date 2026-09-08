from __future__ import annotations

import json
import time

from dataclasses import dataclass, asdict
from pathlib import Path

import torch


@dataclass
class TrainingProfile:

    model_name: str

    device: str

    parameter_count: int

    trainable_parameter_count: int

    parameter_memory_mb: float

    peak_memory_mb: float

    input_tokens: int

    batch_size: int

    sequence_length: int

    forward_seconds: float

    backward_seconds: float

    optimizer_seconds: float

    total_step_seconds: float

    tokens_per_second: float

    loss: float

    gradient_norm: float

    checkpoint_size_mb: float = 0.0

    metadata: dict = None

    def to_dict(self):
        return asdict(self)


class HardwareProfiler:

    def __init__(
        self,
        model,
        device: str | None = None,
    ) -> None:

        self.model = model

        self.device = (
            device
            or (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.model.to(
            self.device
        )

    @staticmethod
    def _parameter_memory(
        model,
    ) -> int:

        total = 0

        for parameter in (
            model.parameters()
        ):

            total += (
                parameter.numel()
                * parameter.element_size()
            )

        for buffer in (
            model.buffers()
        ):

            total += (
                buffer.numel()
                * buffer.element_size()
            )

        return total

    def _reset_memory(self):

        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def _peak_memory(self) -> float:

        if self.device == "cuda":
            return (
                torch.cuda.max_memory_allocated()
                / 1024**2
            )

        return 0.0

    def profile_step(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
    ) -> TrainingProfile:

        self.model.train()

        input_ids = input_ids.to(
            self.device
        )

        if attention_mask is not None:
            attention_mask = (
                attention_mask.to(
                    self.device
                )
            )

        if labels is None:
            labels = input_ids.clone()

        labels = labels.to(
            self.device
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-5,
            weight_decay=0.0,
        )

        self._reset_memory()

        optimizer.zero_grad(
            set_to_none=True
        )

        start_total = time.perf_counter()

        forward_start = (
            time.perf_counter()
        )

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        if self.device == "cuda":
            torch.cuda.synchronize()

        forward_seconds = (
            time.perf_counter()
            - forward_start
        )

        loss = output["loss"]

        backward_start = (
            time.perf_counter()
        )

        loss.backward()

        if self.device == "cuda":
            torch.cuda.synchronize()

        backward_seconds = (
            time.perf_counter()
            - backward_start
        )

        gradient_norm = 0.0

        for parameter in (
            self.model.parameters()
        ):

            if parameter.grad is None:
                continue

            gradient_norm += float(
                parameter.grad.detach()
                .float()
                .norm()
                .cpu()
            )

        optimizer_start = (
            time.perf_counter()
        )

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            1.0,
        )

        optimizer.step()

        if self.device == "cuda":
            torch.cuda.synchronize()

        optimizer_seconds = (
            time.perf_counter()
            - optimizer_start
        )

        total_seconds = (
            time.perf_counter()
            - start_total
        )

        batch_size = (
            input_ids.shape[0]
        )

        sequence_length = (
            input_ids.shape[1]
        )

        token_count = (
            batch_size
            * sequence_length
        )

        tokens_per_second = (
            token_count
            / total_seconds
            if total_seconds > 0
            else 0.0
        )

        parameter_bytes = (
            self._parameter_memory(
                self.model
            )
        )

        return TrainingProfile(
            model_name=(
                self.model.config.model_name
            ),

            device=self.device,

            parameter_count=(
                self.model.parameter_count()
            ),

            trainable_parameter_count=(
                self.model.trainable_parameter_count()
            ),

            parameter_memory_mb=(
                parameter_bytes
                / 1024**2
            ),

            peak_memory_mb=(
                self._peak_memory()
            ),

            input_tokens=(
                token_count
            ),

            batch_size=(
                batch_size
            ),

            sequence_length=(
                sequence_length
            ),

            forward_seconds=(
                forward_seconds
            ),

            backward_seconds=(
                backward_seconds
            ),

            optimizer_seconds=(
                optimizer_seconds
            ),

            total_step_seconds=(
                total_seconds
            ),

            tokens_per_second=(
                tokens_per_second
            ),

            loss=float(
                loss.detach().cpu()
            ),

            gradient_norm=(
                gradient_norm
            ),

            metadata={
                "cuda_available":
                    torch.cuda.is_available(),

                "cuda_device":
                    (
                        torch.cuda.get_device_name(0)
                        if torch.cuda.is_available()
                        else None
                    ),

                "torch_version":
                    torch.__version__,
            },
        )


class ProfileStore:

    def __init__(
        self,
        path: str = (
            "gene/data/training/"
            "profiles/profiles.jsonl"
        ),
    ) -> None:

        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        profile: TrainingProfile,
    ) -> None:

        with self.path.open(
            "a",
            encoding="utf-8",
        ) as handle:

            handle.write(
                json.dumps(
                    profile.to_dict(),
                    ensure_ascii=False,
                )
                + "\n"
            )

    def list(
        self,
    ) -> list[dict]:

        if not self.path.exists():
            return []

        records = []

        with self.path.open(
            "r",
            encoding="utf-8-sig",
        ) as handle:

            for line in handle:

                if line.strip():
                    records.append(
                        json.loads(line)
                    )

        return records

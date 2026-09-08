from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time

from pathlib import Path
from typing import Iterator

import torch

from gene.model.configs import gene_200m_config
from gene.model.neural import GeneTransformer
from gene.training.packing import CorpusPacker
from gene.training.tokenizer import GeneTokenizer
from .checkpoint_manager import CheckpointManager


PACKED_PATH = (
    "gene/data/training/packed/"
    "train.bin"
)

TOKENIZER_PATH = (
    "gene/data/training/tokenizer/"
    "gene-tokenizer.json"
)

PACK_MANIFEST_PATH = (
    "gene/data/training/packed/"
    "train.manifest.json"
)

OUTPUT_DIR = (
    "gene/data/training/production_v2"
)


def sha256_file(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:

        while True:

            chunk = handle.read(
                8 * 1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def atomic_json(
    path: Path,
    payload: dict,
) -> None:

    temporary = path.with_suffix(
        ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    temporary.replace(
        path
    )


def free_disk_bytes(
    path: Path,
) -> int:

    return shutil.disk_usage(
        path
    ).free


def system_memory_gb() -> float:

    if os.name != "nt":
        return 0.0

    try:

        import ctypes

        class MEMORYSTATUSEX(
            ctypes.Structure
        ):
            _fields_ = [
                (
                    "dwLength",
                    ctypes.c_ulong,
                ),
                (
                    "dwMemoryLoad",
                    ctypes.c_ulong,
                ),
                (
                    "ullTotalPhys",
                    ctypes.c_ulonglong,
                ),
                (
                    "ullAvailPhys",
                    ctypes.c_ulonglong,
                ),
                (
                    "ullTotalPageFile",
                    ctypes.c_ulonglong,
                ),
                (
                    "ullAvailPageFile",
                    ctypes.c_ulonglong,
                ),
                (
                    "ullTotalVirtual",
                    ctypes.c_ulonglong,
                ),
                (
                    "ullAvailVirtual",
                    ctypes.c_ulonglong,
                ),
                (
                    "ullAvailExtendedVirtual",
                    ctypes.c_ulonglong,
                ),
            ]

        status = MEMORYSTATUSEX()

        status.dwLength = (
            ctypes.sizeof(
                status
            )
        )

        if not (
            ctypes.windll.kernel32
            .GlobalMemoryStatusEx(
                ctypes.byref(status)
            )
        ):
            return 0.0

        return (
            status.ullTotalPhys
            / 1024**3
        )

    except Exception:
        return 0.0


def configure_runtime() -> int:

    logical = (
        os.cpu_count()
        or 1
    )

    threads = min(
        4,
        logical,
    )

    torch.set_num_threads(
        threads
    )

    try:

        torch.set_num_interop_threads(
            max(
                1,
                min(
                    2,
                    threads,
                ),
            )
        )

    except RuntimeError:
        pass

    return threads


def choose_context(
    requested: int | None,
) -> int:

    if requested is not None:

        if requested not in (
            256,
            512,
            1024,
        ):
            raise ValueError(
                "Context must be 256, 512, "
                "or 1024 for this CPU trainer."
            )

        return requested

    memory = (
        system_memory_gb()
    )

    if memory >= 32:
        return 1024

    if memory >= 16:
        return 512

    return 256


def read_manifest() -> dict:

    path = Path(
        PACK_MANIFEST_PATH
    )

    if not path.exists():
        raise FileNotFoundError(
            str(path)
        )

    return json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )


def manifest_summary() -> dict:

    payload = read_manifest()

    stats = payload.get(
        "stats",
        {},
    )

    return {
        "blocks":
            int(
                stats.get(
                    "blocks_written",
                    0,
                )
            ),
        "tokens":
            int(
                stats.get(
                    "tokens_seen",
                    0,
                )
            ),
        "records":
            int(
                stats.get(
                    "records_seen",
                    0,
                )
            ),
        "block_size":
            int(
                stats.get(
                    "block_size",
                    8192,
                )
            ),
        "eos_token_id":
            stats.get(
                "eos_token_id"
            ),
    }


def iter_block_sequences(
    *,
    packed_path: str,
    block_size: int,
    context_length: int,
    first_block: int = 0,
    last_block: int | None = None,
) -> Iterator[
    tuple[int, int, list[int]]
]:

    for block_index, block in enumerate(
        CorpusPacker.read_blocks(
            packed_path,
            block_size,
        )
    ):

        if block_index < first_block:
            continue

        if (
            last_block is not None
            and block_index >= last_block
        ):
            break

        usable = (
            len(block)
            // context_length
        ) * context_length

        sequence_number = 0

        for offset in range(
            0,
            usable,
            context_length,
        ):

            sequence = block[
                offset:
                offset + context_length
            ]

            yield (
                block_index,
                sequence_number,
                sequence,
            )

            sequence_number += 1


def calculate_lr(
    *,
    step: int,
    base_lr: float,
    warmup_steps: int,
    total_schedule_steps: int,
) -> float:

    if warmup_steps > 0 and (
        step <= warmup_steps
    ):

        return (
            base_lr
            * step
            / warmup_steps
        )

    if total_schedule_steps <= warmup_steps:
        return base_lr

    progress = (
        step
        - warmup_steps
    ) / (
        total_schedule_steps
        - warmup_steps
    )

    progress = min(
        1.0,
        max(
            0.0,
            progress,
        ),
    )

    return (
        0.5
        * base_lr
        * (
            1.0
            + math.cos(
                math.pi * progress
            )
        )
    )


def set_learning_rate(
    optimizer,
    learning_rate: float,
) -> None:

    for group in (
        optimizer.param_groups
    ):
        group["lr"] = (
            learning_rate
        )


class ProductionTrainerV2:

    def __init__(
        self,
        *,
        context_length: int | None = None,
        batch_size: int = 1,
        learning_rate: float = 2e-4,
        warmup_steps: int = 1000,
        schedule_steps: int = 1_600_000,
        validation_fraction: float = 0.01,
        validation_steps: int = 10,
        checkpoint_every: int = 25,
        full_checkpoint_every: int = 250,
        keep_weight_checkpoints: int = 3,
        seed: int = 42,
        output_dir: str = OUTPUT_DIR,
    ) -> None:

        self.context_length = (
            choose_context(
                context_length
            )
        )

        if batch_size != 1:
            raise ValueError(
                "Only batch_size=1 is supported "
                "for the current CPU trainer."
            )

        self.batch_size = batch_size

        self.learning_rate = (
            learning_rate
        )

        self.warmup_steps = (
            warmup_steps
        )

        self.schedule_steps = (
            schedule_steps
        )

        self.validation_fraction = (
            validation_fraction
        )

        self.validation_steps = (
            validation_steps
        )

        self.checkpoint_every = (
            checkpoint_every
        )

        self.full_checkpoint_every = (
            full_checkpoint_every
        )

        self.keep_weight_checkpoints = (
            keep_weight_checkpoints
        )

        self.seed = seed

        self.output_dir = Path(
            output_dir
        )

        self.checkpoint_dir = (
            self.output_dir
            / "checkpoints"
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.checkpoints = CheckpointManager(
            str(self.checkpoint_dir),
            keep_weights=self.keep_weight_checkpoints,
            keep_resume=1,
            minimum_free_gb=6.0,
        )

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.threads = (
            configure_runtime()
        )

        self.model = None
        self.optimizer = None

        self.total_blocks = 0
        self.validation_blocks = 0
        self.validation_start_block = 0
        self.train_sequence_count = 0

        self.history: list[dict] = []

    def preflight(self) -> None:

        packed = Path(
            PACKED_PATH
        )

        tokenizer = Path(
            TOKENIZER_PATH
        )

        if not packed.exists():
            raise FileNotFoundError(
                str(packed)
            )

        if not tokenizer.exists():
            raise FileNotFoundError(
                str(tokenizer)
            )

        manifest = manifest_summary()

        self.total_blocks = (
            manifest["blocks"]
        )

        if self.total_blocks <= 0:
            raise ValueError(
                "Packed corpus contains "
                "no blocks."
            )

        self.validation_blocks = max(
            1,
            int(
                self.total_blocks
                * self.validation_fraction
            ),
        )

        self.validation_start_block = (
            self.total_blocks
            - self.validation_blocks
        )

        block_size = (
            manifest["block_size"]
        )

        sequences_per_block = (
            block_size
            // self.context_length
        )

        train_blocks = (
            self.validation_start_block
        )

        self.train_sequence_count = (
            train_blocks
            * sequences_per_block
        )

        tok = GeneTokenizer.load(
            TOKENIZER_PATH
        )

        config = (
            gene_200m_config()
        )

        if (
            tok.vocab_size_actual()
            != config.vocab_size
        ):
            raise ValueError(
                "Tokenizer/model vocabulary mismatch."
            )

        print(
            json.dumps(
                {
                    "device":
                        self.device,
                    "memory_gb":
                        system_memory_gb(),
                    "logical_cpus":
                        os.cpu_count(),
                    "threads":
                        self.threads,
                    "context":
                        self.context_length,
                    "train_tokens":
                        (
                            train_blocks
                            * block_size
                        ),
                    "validation_tokens":
                        (
                            self.validation_blocks
                            * block_size
                        ),
                    "total_tokens":
                        manifest["tokens"],
                    "train_sequences_per_epoch":
                        self.train_sequence_count,
                    "parameters":
                        195551712,
                    "tokenizer_vocab":
                        tok.vocab_size_actual(),
                    "packed_sha256":
                        sha256_file(
                            packed
                        ),
                    "tokenizer_sha256":
                        sha256_file(
                            tokenizer
                        ),
                },
                indent=2,
            )
        )

    def build(self) -> None:

        config = (
            gene_200m_config()
        )

        self.model = GeneTransformer(
            config
        )

        self.model.to(
            self.device
        )

        self.model.train()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.0,
            betas=(
                0.9,
                0.95,
            ),
            weight_decay=0.1,
        )

    def estimate_weights_bytes(self) -> int:

        return sum(
            parameter.numel()
            * parameter.element_size()
            for parameter
            in self.model.parameters()
        )

    def estimate_full_bytes(self) -> int:

        # Conservative:
        # weights + two Adam moments + margin.
        return int(
            self.estimate_weights_bytes()
            * 3.5
            + 512 * 1024 * 1024
        )

    def ensure_space(
        self,
        full: bool,
    ) -> None:

        free = free_disk_bytes(
            self.checkpoint_dir
        )

        required = (
            self.estimate_full_bytes()
            if full
            else self.estimate_weights_bytes()
        )

        safety = (
            768 * 1024 * 1024
        )

        if free < (
            required + safety
        ):

            raise RuntimeError(
                "Insufficient disk space "
                "for checkpoint. "
                f"Free={free/1e9:.2f}GB, "
                f"Required="
                f"{(required+safety)/1e9:.2f}GB"
            )

    def set_step_lr(
        self,
        step: int,
    ) -> float:

        lr = calculate_lr(
            step=step,
            base_lr=self.learning_rate,
            warmup_steps=self.warmup_steps,
            total_schedule_steps=
                self.schedule_steps,
        )

        set_learning_rate(
            self.optimizer,
            lr,
        )

        return lr

    def save_weights(
        self,
        *,
        step: int,
        epoch: int,
        sequence_index: int,
        validation_loss: float | None,
    ) -> str:

        self.ensure_space(
            full=False
        )

        path = (
            self.checkpoint_dir
            / f"weights-step-{step}.pt"
        )

        temporary = path.with_suffix(
            ".tmp"
        )

        payload = {
            "model_state_dict":
                self.model.state_dict(),
            "step":
                step,
            "epoch":
                epoch,
            "sequence_index":
                sequence_index,
            "validation_loss":
                validation_loss,
            "model_config":
                gene_200m_config().to_dict(),
            "tokenizer_path":
                TOKENIZER_PATH,
            "packed_path":
                PACKED_PATH,
            "training_type":
                "gene_200m_pretraining_v2",
        }

        torch.save(
            payload,
            temporary,
        )

        if (
            not temporary.exists()
            or temporary.stat().st_size
            == 0
        ):

            raise RuntimeError(
                "Weight checkpoint was not written."
            )

        temporary.replace(
            path
        )

        self.prune_weights()

        return str(path)

    def save_resume(
        self,
        *,
        step: int,
        epoch: int,
        sequence_index: int,
        validation_loss: float | None,
    ) -> str:

        self.ensure_space(
            full=True
        )

        path = (
            self.checkpoint_dir
            / f"resume-step-{step}.pt"
        )

        temporary = path.with_suffix(
            ".tmp"
        )

        payload = {
            "model_state_dict":
                self.model.state_dict(),
            "optimizer_state_dict":
                self.optimizer.state_dict(),
            "step":
                step,
            "epoch":
                epoch,
            "sequence_index":
                sequence_index,
            "validation_loss":
                validation_loss,
            "learning_rate":
                self.optimizer
                .param_groups[0]["lr"],
            "base_learning_rate":
                self.learning_rate,
            "warmup_steps":
                self.warmup_steps,
            "schedule_steps":
                self.schedule_steps,
            "checkpoint_every":
                self.checkpoint_every,
            "full_checkpoint_every":
                self.full_checkpoint_every,
            "validation_steps":
                self.validation_steps,
            "model_config":
                gene_200m_config().to_dict(),
            "tokenizer_path":
                TOKENIZER_PATH,
            "packed_path":
                PACKED_PATH,
            "training_type":
                "gene_200m_pretraining_v2",
            "seed":
                self.seed,
        }

        torch.save(
            payload,
            temporary,
        )

        if (
            not temporary.exists()
            or temporary.stat().st_size
            == 0
        ):

            raise RuntimeError(
                "Resume checkpoint was not written."
            )

        temporary.replace(
            path
        )

        self.prune_resume_checkpoints(
            keep=1
        )

        return str(path)

    def prune_weights(self) -> None:

        files = sorted(
            self.checkpoint_dir.glob(
                "weights-step-*.pt"
            ),
            key=lambda path:
                path.stat().st_mtime,
            reverse=True,
        )

        for path in files[
            self.keep_weight_checkpoints:
        ]:

            try:
                path.unlink()
            except OSError:
                pass

    def prune_resume_checkpoints(
        self,
        keep: int = 1,
    ) -> None:

        files = sorted(
            self.checkpoint_dir.glob(
                "resume-step-*.pt"
            ),
            key=lambda path:
                path.stat().st_mtime,
            reverse=True,
        )

        for path in files[
            keep:
        ]:

            try:
                path.unlink()
            except OSError:
                pass

    def update_latest(
        self,
        *,
        step: int,
        epoch: int,
        sequence_index: int,
        weights: str,
        resume: str | None,
    ) -> None:

        atomic_json(
            self.checkpoint_dir
            / "latest.json",
            {
                "step":
                    step,
                "epoch":
                    epoch,
                "sequence_index":
                    sequence_index,
                "weights_checkpoint":
                    weights,
                "resume_checkpoint":
                    resume,
                "free_disk_gb":
                    free_disk_bytes(
                        self.checkpoint_dir
                    ) / 1e9,
            },
        )

    def validate(
        self,
    ) -> float:

        self.model.eval()

        total = 0.0
        count = 0

        iterator = iter_block_sequences(
            packed_path=PACKED_PATH,
            block_size=8192,
            context_length=
                self.context_length,
            first_block=
                self.validation_start_block,
        )

        with torch.no_grad():

            for (
                _,
                _,
                sequence,
            ) in iterator:

                input_ids = torch.tensor(
                    [sequence],
                    dtype=torch.long,
                    device=self.device,
                )

                labels = input_ids.clone()

                output = self.model(
                    input_ids=input_ids,
                    labels=labels,
                )

                loss = output["loss"]

                if not torch.isfinite(
                    loss
                ):
                    raise RuntimeError(
                        "Validation produced "
                        "non-finite loss."
                    )

                total += float(
                    loss.detach().cpu()
                )

                count += 1

                if (
                    count
                    >= self.validation_steps
                ):
                    break

        self.model.train()

        if count == 0:
            raise RuntimeError(
                "Validation produced no sequences."
            )

        return total / count

    def save_metrics(
        self,
        record: dict,
    ) -> None:

        path = (
            self.output_dir
            / "metrics.jsonl"
        )

        with path.open(
            "a",
            encoding="utf-8",
        ) as handle:

            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    def load_resume(
        self,
    ) -> dict | None:

        latest = (
            self.checkpoint_dir
            / "latest.json"
        )

        if not latest.exists():
            return None

        metadata = json.loads(
            latest.read_text(
                encoding="utf-8-sig"
            )
        )

        path_value = (
            metadata.get(
                "resume_checkpoint"
            )
        )

        if not path_value:
            return None

        path = Path(
            path_value
        )

        if not path.exists():
            return None

        checkpoint = torch.load(
            path,
            map_location="cpu",
        )

        required = (
            "model_state_dict",
            "optimizer_state_dict",
            "step",
            "epoch",
            "sequence_index",
        )

        missing = [
            key
            for key in required
            if key not in checkpoint
        ]

        if missing:
            raise ValueError(
                "Invalid resume checkpoint. "
                "Missing: "
                + ", ".join(missing)
            )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        checkpoint_warmup = checkpoint.get(
            "warmup_steps"
        )

        checkpoint_schedule = checkpoint.get(
            "schedule_steps"
        )

        if (
            checkpoint_warmup is not None
            and checkpoint_schedule is not None
        ):

            self.warmup_steps = int(
                checkpoint_warmup
            )

            self.schedule_steps = int(
                checkpoint_schedule
            )

            self.learning_rate = float(
                checkpoint.get(
                    "base_learning_rate",
                    self.learning_rate,
                )
            )

        return checkpoint

    def train(
        self,
        *,
        steps: int,
        resume: bool,
        validate_every: int,
    ) -> dict:

        configure_runtime()

        self.preflight()
        self.build()

        start_step = 0
        epoch = 0
        sequence_index = 0
        last_validation = None

        if resume:

            checkpoint = (
                self.load_resume()
            )

            if checkpoint:

                start_step = int(
                    checkpoint["step"]
                )

                epoch = int(
                    checkpoint["epoch"]
                )

                sequence_index = int(
                    checkpoint[
                        "sequence_index"
                    ]
                )

                print(
                    "RESUMED FROM STEP",
                    start_step,
                    "EPOCH",
                    epoch,
                    "SEQUENCE",
                    sequence_index,
                )

        target_step = (
            start_step + steps
        )

        global_step = start_step

        while global_step < target_step:

            if (
                sequence_index
                >= self.train_sequence_count
            ):

                epoch += 1
                sequence_index = 0

            iterator = iter_block_sequences(
                packed_path=PACKED_PATH,
                block_size=8192,
                context_length=
                    self.context_length,
                first_block=0,
                last_block=
                    self.validation_start_block,
            )

            skipped = 0

            if sequence_index:

                for _ in iterator:

                    skipped += 1

                    if (
                        skipped
                        >= sequence_index
                    ):
                        break

            for (
                block_index,
                sequence_number,
                sequence,
            ) in iterator:

                if global_step >= target_step:
                    break

                global_step += 1

                sequence_index += 1

                lr = (
                    self.set_step_lr(
                        global_step
                    )
                )

                input_ids = torch.tensor(
                    [sequence],
                    dtype=torch.long,
                    device=self.device,
                )

                labels = (
                    input_ids.clone()
                )

                self.optimizer.zero_grad(
                    set_to_none=True
                )

                started = (
                    time.perf_counter()
                )

                output = self.model(
                    input_ids=input_ids,
                    labels=labels,
                )

                loss = output["loss"]

                if not torch.isfinite(
                    loss
                ):
                    raise RuntimeError(
                        f"Non-finite loss at "
                        f"step {global_step}."
                    )

                loss.backward()

                gradient_norm = (
                    torch.nn.utils
                    .clip_grad_norm_(
                        self.model.parameters(),
                        1.0,
                    )
                )

                self.optimizer.step()

                elapsed = (
                    time.perf_counter()
                    - started
                )

                record = {
                    "step":
                        global_step,
                    "epoch":
                        epoch,
                    "sequence_index":
                        sequence_index,
                    "block_index":
                        block_index,
                    "loss":
                        float(
                            loss.detach()
                            .cpu()
                        ),
                    "learning_rate":
                        lr,
                    "gradient_norm":
                        float(
                            gradient_norm
                        ),
                    "seconds":
                        elapsed,
                    "tokens_per_second":
                        (
                            self.context_length
                            / elapsed
                        ),
                }

                if (
                    global_step
                    % validate_every
                    == 0
                ):

                    last_validation = (
                        self.validate()
                    )

                    record[
                        "validation_loss"
                    ] = (
                        last_validation
                    )

                self.history.append(
                    record
                )

                self.save_metrics(
                    record
                )

                print(
                    f"step={global_step} "
                    f"loss={record['loss']:.6f} "
                    f"lr={lr:.8f} "
                    f"tok/s="
                    f"{record['tokens_per_second']:.2f}"
                    + (
                        f" val="
                        f"{last_validation:.6f}"
                        if last_validation
                        is not None
                        else ""
                    )
                )

                if (
                    global_step
                    % self.checkpoint_every
                    == 0
                ):

                    weights = (
                        self.save_weights(
                            step=global_step,
                            epoch=epoch,
                            sequence_index=
                                sequence_index,
                            validation_loss=
                                last_validation,
                        )
                    )

                    resume_path = None

                    if (
                        global_step
                        % self.full_checkpoint_every
                        == 0
                    ):

                        resume_path = (
                            self.save_resume(
                                step=
                                    global_step,
                                epoch=epoch,
                                sequence_index=
                                    sequence_index,
                                validation_loss=
                                    last_validation,
                            )
                        )

                    self.update_latest(
                        step=
                            global_step,
                        epoch=epoch,
                        sequence_index=
                            sequence_index,
                        weights=
                            weights,
                        resume=
                            resume_path,
                    )

                    print(
                        "WEIGHTS:",
                        weights,
                    )

                    if resume_path:
                        print(
                            "RESUME:",
                            resume_path,
                        )

                if (
                    global_step
                    >= target_step
                ):
                    break

            if (
                global_step
                >= target_step
            ):
                break

        final_weights = (
            self.save_weights(
                step=global_step,
                epoch=epoch,
                sequence_index=
                    sequence_index,
                validation_loss=
                    last_validation,
            )
        )

        report = {
            "success":
                True,
            "steps_completed":
                global_step
                - start_step,
            "global_step":
                global_step,
            "epoch":
                epoch,
            "sequence_index":
                sequence_index,
            "context_length":
                self.context_length,
            "threads":
                self.threads,
            "device":
                self.device,
            "first_loss":
                (
                    self.history[0]["loss"]
                    if self.history
                    else None
                ),
            "last_loss":
                (
                    self.history[-1]["loss"]
                    if self.history
                    else None
                ),
            "loss_reduced":
                (
                    bool(self.history)
                    and
                    self.history[-1]["loss"]
                    <
                    self.history[0]["loss"]
                ),
            "last_validation_loss":
                last_validation,
            "checkpoint":
                final_weights,
        }

        atomic_json(
            self.output_dir
            / "training_report.json",
            report,
        )

        return report



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--steps",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--context",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--schedule-steps",
        type=int,
        default=1600000,
    )

    parser.add_argument(
        "--validate-every",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--validation-steps",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--full-checkpoint-every",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
    )

    args = parser.parse_args()

    trainer = ProductionTrainerV2(
        context_length=args.context,
        warmup_steps=args.warmup_steps,
        schedule_steps=args.schedule_steps,
        validation_steps=args.validation_steps,
        checkpoint_every=args.checkpoint_every,
        full_checkpoint_every=args.full_checkpoint_every,
    )

    result = trainer.train(
        steps=args.steps,
        resume=not args.no_resume,
        validate_every=args.validate_every,
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "GENE 200M PRODUCTION V2"
    )

    print(
        "=" * 72
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

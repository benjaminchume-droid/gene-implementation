from __future__ import annotations

import json
import os
import random
import time

from pathlib import Path

import torch

from gene.model.configs import gene_200m_config
from gene.model.neural import GeneTransformer
from gene.training.packing import CorpusPacker
from gene.training.tokenizer import GeneTokenizer


DEFAULT_PACKED = (
    "gene/data/training/packed/"
    "train.bin"
)

DEFAULT_TOKENIZER = (
    "gene/data/training/tokenizer/"
    "gene-tokenizer.json"
)

DEFAULT_OUTPUT = (
    "gene/data/training/production"
)


def detect_memory_gb() -> float:

    if os.name != "nt":
        return 0.0

    try:

        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = (
            ctypes.sizeof(status)
        )

        ctypes.windll.kernel32.GlobalMemoryStatusEx(
            ctypes.byref(status)
        )

        return (
            status.ullTotalPhys
            / 1024**3
        )

    except Exception:
        return 0.0


def choose_config(
    requested_context: int | None = None,
) -> dict:

    memory_gb = detect_memory_gb()

    logical_cpus = (
        os.cpu_count()
        or 1
    )

    # Conservative CPU-safe defaults.
    if memory_gb >= 32:
        default_context = 1024
    elif memory_gb >= 16:
        default_context = 512
    else:
        default_context = 256

    context = (
        requested_context
        or default_context
    )

    if context not in (
        256,
        512,
        1024,
    ):
        raise ValueError(
            "Initial production context must "
            "be 256, 512, or 1024."
        )

    # The benchmark showed 4 threads was best
    # on the current machine, but a new machine
    # must be measured rather than assumed.
    threads = min(
        4,
        logical_cpus,
    )

    return {
        "memory_gb":
            memory_gb,
        "logical_cpus":
            logical_cpus,
        "threads":
            threads,
        "context_length":
            context,
        "micro_batch_size":
            1,
        "gradient_accumulation":
            1,
    }


def set_cpu_runtime(
    threads: int,
) -> None:

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
        # Interop thread count cannot be changed
        # after parallel work has started.
        pass


def seed_everything(
    seed: int,
) -> None:

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def atomic_write_json(
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


def load_training_blocks(
    packed_path: str,
    *,
    context_length: int,
):
    """
    Converts packed 8192-token blocks into
    smaller causal-LM sequences.
    """

    for block_index, block in enumerate(
        CorpusPacker.read_blocks(
            packed_path,
            8192,
        )
    ):

        usable = (
            len(block)
            // context_length
        ) * context_length

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
                sequence,
            )


class ProductionPretrainer:

    def __init__(
        self,
        *,
        packed_path: str = DEFAULT_PACKED,
        tokenizer_path: str = DEFAULT_TOKENIZER,
        output_dir: str = DEFAULT_OUTPUT,
        context_length: int | None = None,
        learning_rate: float = 2e-4,
        seed: int = 42,
        checkpoint_every: int = 25,
        full_checkpoint_every: int = 250,
    ) -> None:

        self.packed_path = Path(
            packed_path
        )

        self.tokenizer_path = Path(
            tokenizer_path
        )

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

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.config = choose_config(
            context_length
        )

        self.learning_rate = (
            learning_rate
        )

        self.seed = seed

        self.checkpoint_every = (
            checkpoint_every
        )

        self.full_checkpoint_every = (
            full_checkpoint_every
        )

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.tokenizer = None

    def preflight(self) -> None:

        if not self.packed_path.exists():
            raise FileNotFoundError(
                str(
                    self.packed_path
                )
            )

        if not self.tokenizer_path.exists():
            raise FileNotFoundError(
                str(
                    self.tokenizer_path
                )
            )

        model_config = (
            gene_200m_config()
        )

        if (
            self.config["context_length"]
            > model_config.max_position_embeddings
        ):
            raise ValueError(
                "Requested context exceeds "
                "model context."
            )

        tokenizer = GeneTokenizer.load(
            str(
                self.tokenizer_path
            )
        )

        if (
            tokenizer.vocab_size_actual()
            != model_config.vocab_size
        ):
            raise ValueError(
                "Tokenizer/model vocabulary mismatch."
            )

        if self.device == "cpu":
            print(
                "WARNING: CPU-only training."
            )

        print(
            json.dumps(
                {
                    "device":
                        self.device,
                    "hardware":
                        self.config,
                    "parameters":
                        195551712,
                    "context":
                        self.config[
                            "context_length"
                        ],
                    "learning_rate":
                        self.learning_rate,
                },
                indent=2,
            )
        )

    def build(self) -> None:

        config = gene_200m_config()

        self.tokenizer = GeneTokenizer.load(
            str(
                self.tokenizer_path
            )
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
            lr=self.learning_rate,
            betas=(
                0.9,
                0.95,
            ),
            weight_decay=0.1,
        )

        self.scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=100000,
            )
        )

    def _free_disk_bytes(self) -> int:

        try:

            import shutil

            return shutil.disk_usage(
                self.output_dir
            ).free

        except Exception:
            return 0

    def _estimate_full_checkpoint_bytes(
        self,
    ) -> int:

        total = 0

        for parameter in (
            self.model.parameters()
        ):

            total += (
                parameter.numel()
                * parameter.element_size()
            )

        # Conservative estimate for AdamW:
        # model weights + gradients + two optimizer
        # moment tensors + safety margin.
        return int(
            total
            * 5.0
            * 1.25
        )

    def _ensure_checkpoint_space(
        self,
        *,
        full: bool,
    ) -> None:

        free_bytes = (
            self._free_disk_bytes()
        )

        if full:

            required = (
                self._estimate_full_checkpoint_bytes()
            )

        else:

            required = sum(
                parameter.numel()
                * parameter.element_size()
                for parameter
                in self.model.parameters()
            )

            required = int(
                required
                * 1.15
            )

        safety = (
            512 * 1024 * 1024
        )

        if free_bytes < (
            required + safety
        ):

            raise RuntimeError(
                "Insufficient disk space for "
                "checkpoint. "
                f"free={free_bytes / 1e9:.2f}GB "
                f"required={(required + safety) / 1e9:.2f}GB"
            )

    def _atomic_torch_save(
        self,
        payload: dict,
        path: Path,
    ) -> None:

        temporary = (
            path.with_suffix(
                ".tmp"
            )
        )

        if temporary.exists():

            try:
                temporary.unlink()
            except OSError:
                pass

        torch.save(
            payload,
            temporary,
        )

        if (
            not temporary.exists()
            or temporary.stat().st_size <= 0
        ):

            raise RuntimeError(
                f"Checkpoint write failed: "
                f"{temporary}"
            )

        temporary.replace(
            path
        )

    def save_weights_checkpoint(
        self,
        *,
        step: int,
        block_index: int,
        sequence_index: int,
        history: list[dict],
    ) -> str:

        self._ensure_checkpoint_space(
            full=False
        )

        path = (
            self.checkpoint_dir
            / f"weights-step-{step}.pt"
        )

        payload = {
            "model_state_dict":
                self.model.state_dict(),

            "step":
                step,

            "block_index":
                block_index,

            "sequence_index":
                sequence_index,

            "history_tail":
                history[-10:],

            "model_config":
                gene_200m_config().to_dict(),

            "tokenizer_path":
                str(
                    self.tokenizer_path
                ),

            "packed_path":
                str(
                    self.packed_path
                ),

            "training_type":
                "gene_200m_pretraining",
        }

        self._atomic_torch_save(
            payload,
            path,
        )

        return str(path)

    def save_full_checkpoint(
        self,
        *,
        step: int,
        block_index: int,
        sequence_index: int,
        history: list[dict],
    ) -> str:

        self._ensure_checkpoint_space(
            full=True
        )

        path = (
            self.checkpoint_dir
            / f"resume-step-{step}.pt"
        )

        payload = {
            "model_state_dict":
                self.model.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            "scheduler_state_dict":
                self.scheduler.state_dict(),

            "step":
                step,

            "block_index":
                block_index,

            "sequence_index":
                sequence_index,

            "history":
                history,

            "seed":
                self.seed,

            "config":
                self.config,

            "model_config":
                gene_200m_config().to_dict(),

            "tokenizer_path":
                str(
                    self.tokenizer_path
                ),

            "packed_path":
                str(
                    self.packed_path
                ),

            "training_type":
                "gene_200m_pretraining",
        }

        self._atomic_torch_save(
            payload,
            path,
        )

        return str(path)

    def save_checkpoint(
        self,
        *,
        step: int,
        block_index: int,
        sequence_index: int,
        history: list[dict],
    ) -> dict:

        weights_path = (
            self.save_weights_checkpoint(
                step=step,
                block_index=block_index,
                sequence_index=sequence_index,
                history=history,
            )
        )

        full_path = None

        if (
            step % self.full_checkpoint_every
            == 0
        ):

            full_path = (
                self.save_full_checkpoint(
                    step=step,
                    block_index=block_index,
                    sequence_index=sequence_index,
                    history=history,
                )
            )

        latest = (
            self.checkpoint_dir
            / "latest.json"
        )

        atomic_write_json(
            latest,
            {
                "weights_checkpoint":
                    weights_path,

                "resume_checkpoint":
                    full_path,

                "step":
                    step,

                "block_index":
                    block_index,

                "sequence_index":
                    sequence_index,

                "free_disk_gb":
                    self._free_disk_bytes()
                    / 1e9,
            },
        )

        return {
            "weights":
                weights_path,
            "resume":
                full_path,
        }

    def load_latest(self):

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

        checkpoint_value = (
            metadata.get(
                "resume_checkpoint"
            )
            or metadata.get(
                "checkpoint"
            )
        )

        if not checkpoint_value:

            print(
                "No full resume checkpoint "
                "is available yet."
            )

            return None

        path = Path(
            checkpoint_value
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Resume checkpoint not found: "
                f"{path}"
            )

        checkpoint = torch.load(
            path,
            map_location="cpu",
        )

        required = (
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "step",
        )

        missing = [
            key
            for key in required
            if key not in checkpoint
        ]

        if missing:

            raise ValueError(
                "Checkpoint is not a full "
                "resume checkpoint. Missing: "
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

        self.scheduler.load_state_dict(
            checkpoint[
                "scheduler_state_dict"
            ]
        )

        return checkpoint

    def train(
        self,
        *,
        max_steps: int,
        resume: bool = True,
    ) -> dict:

        seed_everything(
            self.seed
        )

        set_cpu_runtime(
            self.config[
                "threads"
            ]
        )

        self.preflight()
        self.build()

        checkpoint = (
            self.load_latest()
            if resume
            else None
        )

        start_step = 0

        if checkpoint:

            start_step = int(
                checkpoint[
                    "step"
                ]
            )

            print(
                "RESUMED FROM STEP",
                start_step,
            )

        history = []

        sequences_seen = 0

        step = start_step

        for (
            block_index,
            sequence,
        ) in load_training_blocks(
            str(
                self.packed_path
            ),
            context_length=
                self.config[
                    "context_length"
                ],
        ):

            if step >= max_steps:
                break

            step += 1

            input_ids = torch.tensor(
                [sequence],
                dtype=torch.long,
                device=self.device,
            )

            labels = input_ids.clone()

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
                    f"Non-finite loss at step {step}."
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

            self.scheduler.step()

            elapsed = (
                time.perf_counter()
                - started
            )

            sequences_seen += 1

            record = {
                "step":
                    step,
                "loss":
                    float(
                        loss.detach()
                        .cpu()
                    ),
                "seconds":
                    elapsed,
                "tokens":
                    self.config[
                        "context_length"
                    ],
                "tokens_per_second":
                    (
                        self.config[
                            "context_length"
                        ]
                        / elapsed
                    ),
                "gradient_norm":
                    float(
                        gradient_norm
                    ),
                "block_index":
                    block_index,
            }

            history.append(
                record
            )

            print(
                f"step={step} "
                f"loss={record['loss']:.6f} "
                f"tok/s="
                f"{record['tokens_per_second']:.2f}"
            )

            if (
                step % self.checkpoint_every
                == 0
            ):

                checkpoint = (
                    self.save_checkpoint(
                        step=step,
                        block_index=
                            block_index,
                        sequence_index=
                            sequences_seen,
                        history=history,
                    )
                )

                print(
                    "WEIGHTS CHECKPOINT:",
                    checkpoint["weights"],
                )

                if checkpoint["resume"]:
                    print(
                        "RESUME CHECKPOINT:",
                        checkpoint["resume"],
                    )

        final_checkpoint = (
            self.save_checkpoint(
                step=step,
                block_index=block_index
                if "block_index"
                in locals()
                else 0,
                sequence_index=
                    sequences_seen,
                history=history,
            )
        )

        final_path = (
            final_checkpoint["weights"]
        )

        report = {
            "success":
                True,
            "step":
                step,
            "checkpoint":
                final_path,
            "context_length":
                self.config[
                    "context_length"
                ],
            "threads":
                self.config[
                    "threads"
                ],
            "device":
                self.device,
            "first_loss":
                history[0]["loss"]
                if history
                else None,
            "last_loss":
                history[-1]["loss"]
                if history
                else None,
            "loss_reduced":
                (
                    bool(history)
                    and
                    history[-1]["loss"]
                    < history[0]["loss"]
                ),
            "history":
                history,
        }

        atomic_write_json(
            self.output_dir
            / "training_report.json",
            report,
        )

        return report


if __name__ == "__main__":

    import argparse

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

    trainer = ProductionPretrainer(
        context_length=
            args.context,
        checkpoint_every=
            args.checkpoint_every,
        full_checkpoint_every=
            args.full_checkpoint_every,
    )

    result = trainer.train(
        max_steps=args.steps,
        resume=not args.no_resume,
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "GENE 200M PRODUCTION PILOT"
    )

    print(
        "=" * 72
    )

    print(
        f"Steps: {result['step']}"
    )

    print(
        f"First loss: "
        f"{result['first_loss']}"
    )

    print(
        f"Last loss: "
        f"{result['last_loss']}"
    )

    print(
        f"Loss reduced: "
        f"{result['loss_reduced']}"
    )

    print(
        f"Checkpoint: "
        f"{result['checkpoint']}"
    )

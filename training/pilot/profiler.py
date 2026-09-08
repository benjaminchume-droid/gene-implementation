from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from gene.model.configs import gene_200m_config
from gene.model.neural import GeneTransformer
from gene.training.packing import CorpusPacker


def process_memory_mb() -> float:
    """
    Windows working-set measurement with a graceful fallback.
    """

    if os.name != "nt":
        return 0.0

    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(
            ctypes.Structure
        ):
            _fields_ = [
                (
                    "cb",
                    wintypes.DWORD,
                ),
                (
                    "PageFaultCount",
                    wintypes.DWORD,
                ),
                (
                    "PeakWorkingSetSize",
                    ctypes.c_size_t,
                ),
                (
                    "WorkingSetSize",
                    ctypes.c_size_t,
                ),
                (
                    "QuotaPeakPagedPoolUsage",
                    ctypes.c_size_t,
                ),
                (
                    "QuotaPagedPoolUsage",
                    ctypes.c_size_t,
                ),
                (
                    "QuotaPeakNonPagedPoolUsage",
                    ctypes.c_size_t,
                ),
                (
                    "QuotaNonPagedPoolUsage",
                    ctypes.c_size_t,
                ),
                (
                    "PagefileUsage",
                    ctypes.c_size_t,
                ),
                (
                    "PeakPagefileUsage",
                    ctypes.c_size_t,
                ),
            ]

        process = ctypes.windll.kernel32.GetCurrentProcess()

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(
            counters
        )

        ctypes.windll.psapi.GetProcessMemoryInfo(
            process,
            ctypes.byref(counters),
            counters.cb,
        )

        return (
            counters.WorkingSetSize
            / 1024**2
        )

    except Exception:
        return 0.0


def read_block(
    path: str,
    block_size: int,
):
    block = next(
        CorpusPacker.read_blocks(
            path,
            block_size,
        )
    )

    return torch.tensor(
        block,
        dtype=torch.long,
    )


def run_pilot(
    *,
    packed_path: str,
    steps: int = 2,
    sequence_length: int = 256,
    learning_rate: float = 1e-5,
    num_threads: int | None = None,
    output_path: str = (
        "gene/data/training/pilot/"
        "200m_pilot_report.json"
    ),
) -> dict:

    if steps <= 0:
        raise ValueError(
            "steps must be positive."
        )

    if sequence_length <= 0:
        raise ValueError(
            "sequence_length must be positive."
        )

    if not Path(
        packed_path
    ).exists():

        raise FileNotFoundError(
            packed_path
        )

    if num_threads is not None:

        if num_threads <= 0:
            raise ValueError(
                "num_threads must be positive."
            )

        torch.set_num_threads(
            num_threads
        )

    config = gene_200m_config()

    if (
        sequence_length
        > config.max_position_embeddings
    ):
        raise ValueError(
            "sequence_length exceeds "
            "model context."
        )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Creating Gene 200M model..."
    )

    model = GeneTransformer(
        config
    )

    model.to(device)
    model.train()

    print(
        f"Parameters: "
        f"{model.parameter_count():,}"
    )

    print(
        f"Device: {device}"
    )

    print(
        "Loading real packed corpus..."
    )

    full_block = read_block(
        packed_path,
        config.max_position_embeddings,
    )

    if len(full_block) < sequence_length:
        raise RuntimeError(
            "Packed corpus does not contain "
            "enough tokens for the requested "
            "sequence length."
        )

    input_ids = (
        full_block[
            :sequence_length
        ]
        .unsqueeze(0)
        .to(device)
    )

    labels = input_ids.clone()

    # We are doing a causal language-model step.
    attention_mask = torch.ones_like(
        input_ids,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )

    memory_before = (
        process_memory_mb()
    )

    parameter_bytes = sum(
        parameter.numel()
        * parameter.element_size()
        for parameter
        in model.parameters()
    )

    parameter_memory_mb = (
        parameter_bytes
        / 1024**2
    )

    print(
        f"Parameter memory: "
        f"{parameter_memory_mb:.2f} MB"
    )

    step_records = []

    for step in range(1, steps + 1):

        print(
            f"\nSTEP {step}/{steps}"
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        total_start = (
            time.perf_counter()
        )

        forward_start = (
            time.perf_counter()
        )

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        if device == "cuda":
            torch.cuda.synchronize()

        forward_seconds = (
            time.perf_counter()
            - forward_start
        )

        loss = output["loss"]

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                f"Non-finite loss at step {step}: "
                f"{loss}"
            )

        backward_start = (
            time.perf_counter()
        )

        loss.backward()

        if device == "cuda":
            torch.cuda.synchronize()

        backward_seconds = (
            time.perf_counter()
            - backward_start
        )

        gradient_norm = 0.0

        gradient_tensors = 0

        for parameter in (
            model.parameters()
        ):

            if parameter.grad is None:
                continue

            gradient_tensors += 1

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
            model.parameters(),
            1.0,
        )

        optimizer.step()

        if device == "cuda":
            torch.cuda.synchronize()

        optimizer_seconds = (
            time.perf_counter()
            - optimizer_start
        )

        total_seconds = (
            time.perf_counter()
            - total_start
        )

        token_count = (
            input_ids.shape[0]
            * input_ids.shape[1]
        )

        tokens_per_second = (
            token_count
            / total_seconds
        )

        memory_now = (
            process_memory_mb()
        )

        cuda_peak_mb = 0.0

        if device == "cuda":
            cuda_peak_mb = (
                torch.cuda
                .max_memory_allocated()
                / 1024**2
            )

        record = {
            "step":
                step,
            "loss":
                float(
                    loss.detach()
                    .cpu()
                ),
            "forward_seconds":
                forward_seconds,
            "backward_seconds":
                backward_seconds,
            "optimizer_seconds":
                optimizer_seconds,
            "total_seconds":
                total_seconds,
            "tokens":
                token_count,
            "tokens_per_second":
                tokens_per_second,
            "gradient_norm":
                gradient_norm,
            "gradient_tensors":
                gradient_tensors,
            "ram_mb":
                memory_now,
            "cuda_peak_mb":
                cuda_peak_mb,
        }

        step_records.append(
            record
        )

        print(
            f"loss={record['loss']:.6f} "
            f"time={record['total_seconds']:.3f}s "
            f"tok/s={record['tokens_per_second']:.2f} "
            f"RAM={record['ram_mb']:.2f}MB"
        )

    memory_after = (
        process_memory_mb()
    )

    average_step = (
        sum(
            record["total_seconds"]
            for record
            in step_records
        )
        / len(step_records)
    )

    average_tokens_per_second = (
        sum(
            record["tokens_per_second"]
            for record
            in step_records
        )
        / len(step_records)
    )

    first_loss = (
        step_records[0]["loss"]
    )

    last_loss = (
        step_records[-1]["loss"]
    )

    report = {
        "model": config.to_dict(),
        "parameter_count":
            model.parameter_count(),
        "trainable_parameter_count":
            model.trainable_parameter_count(),
        "device":
            device,
        "torch_version":
            torch.__version__,
        "cpu_threads":
            torch.get_num_threads(),
        "packed_path":
            packed_path,
        "sequence_length":
            sequence_length,
        "batch_size":
            1,
        "steps":
            steps,
        "learning_rate":
            learning_rate,
        "parameter_memory_mb":
            parameter_memory_mb,
        "memory_before_mb":
            memory_before,
        "memory_after_mb":
            memory_after,
        "average_step_seconds":
            average_step,
        "average_tokens_per_second":
            average_tokens_per_second,
        "first_loss":
            first_loss,
        "last_loss":
            last_loss,
        "loss_reduced":
            last_loss < first_loss,
        "steps_detail":
            step_records,
    }

    output = Path(
        output_path
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return report


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--packed",
        default=(
            "gene/data/training/packed/"
            "train.bin"
        ),
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output",
        default=(
            "gene/data/training/pilot/"
            "200m_pilot_report.json"
        ),
    )

    args = parser.parse_args()

    report = run_pilot(
        packed_path=args.packed,
        steps=args.steps,
        sequence_length=
            args.sequence_length,
        learning_rate=
            args.learning_rate,
        num_threads=args.threads,
        output_path=args.output,
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "GENE 200M PILOT PROFILE"
    )

    print(
        "=" * 72
    )

    print(
        f"Parameters: "
        f"{report['parameter_count']:,}"
    )

    print(
        f"Sequence length: "
        f"{report['sequence_length']}"
    )

    print(
        f"Steps: "
        f"{report['steps']}"
    )

    print(
        f"Average step: "
        f"{report['average_step_seconds']:.3f}s"
    )

    print(
        f"Average tokens/sec: "
        f"{report['average_tokens_per_second']:.2f}"
    )

    print(
        f"First loss: "
        f"{report['first_loss']:.6f}"
    )

    print(
        f"Last loss: "
        f"{report['last_loss']:.6f}"
    )

    print(
        f"Loss reduced: "
        f"{report['loss_reduced']}"
    )

    print(
        f"RAM before: "
        f"{report['memory_before_mb']:.2f} MB"
    )

    print(
        f"RAM after: "
        f"{report['memory_after_mb']:.2f} MB"
    )

    print(
        f"Report: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()

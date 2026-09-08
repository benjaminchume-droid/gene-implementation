from __future__ import annotations

import ctypes
import json
import os
import time

from pathlib import Path

import torch

from gene.model.configs import gene_200m_config
from gene.model.neural import GeneTransformer
from gene.training.packing import CorpusPacker


DEFAULT_OUTPUT = (
    "gene/data/training/pilot/"
    "context_scaling.json"
)


def process_memory_mb() -> float:

    if os.name != "nt":
        return 0.0

    try:

        class PROCESS_MEMORY_COUNTERS(
            ctypes.Structure
        ):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)

        process = ctypes.windll.kernel32.GetCurrentProcess()

        get_memory = (
            ctypes.windll.psapi.GetProcessMemoryInfo
        )

        get_memory.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(
                PROCESS_MEMORY_COUNTERS
            ),
            ctypes.c_ulong,
        ]

        get_memory.restype = ctypes.c_bool

        if not get_memory(
            process,
            ctypes.byref(counters),
            counters.cb,
        ):
            return 0.0

        return (
            counters.WorkingSetSize
            / 1024**2
        )

    except Exception:
        return 0.0


def load_tokens(
    packed_path: str,
    count: int,
):

    tokens = []

    for block in CorpusPacker.read_blocks(
        packed_path,
        8192,
    ):

        tokens.extend(block)

        if len(tokens) >= count:
            break

    if len(tokens) < count:
        raise RuntimeError(
            f"Corpus contains only "
            f"{len(tokens)} tokens, requested "
            f"{count}."
        )

    return torch.tensor(
        tokens[:count],
        dtype=torch.long,
    )


def save_report(
    report: dict,
    output_path: str,
) -> None:

    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = output.with_suffix(
        ".tmp"
    )

    temporary.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    temporary.replace(output)


def run_one_context(
    *,
    packed_path: str,
    context_length: int,
    learning_rate: float = 1e-5,
) -> dict:

    config = gene_200m_config()

    if (
        context_length
        > config.max_position_embeddings
    ):
        raise ValueError(
            f"context_length={context_length} "
            f"exceeds model maximum "
            f"{config.max_position_embeddings}"
        )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Creating fresh model for "
        f"context {context_length}..."
    )

    model = GeneTransformer(
        config
    )

    model.to(device)
    model.train()

    tokens = load_tokens(
        packed_path,
        context_length,
    )

    input_ids = (
        tokens
        .unsqueeze(0)
        .to(device)
    )

    labels = input_ids.clone()

    attention_mask = torch.ones_like(
        input_ids,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    memory_before = process_memory_mb()

    total_start = time.perf_counter()

    optimizer.zero_grad(
        set_to_none=True
    )

    forward_start = time.perf_counter()

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

    if not torch.isfinite(loss):
        raise RuntimeError(
            f"Non-finite loss: {loss}"
        )

    backward_start = time.perf_counter()

    loss.backward()

    if device == "cuda":
        torch.cuda.synchronize()

    backward_seconds = (
        time.perf_counter()
        - backward_start
    )

    gradient_norm = 0.0
    gradient_tensors = 0

    for parameter in model.parameters():

        if parameter.grad is None:
            continue

        gradient_tensors += 1

        gradient_norm += float(
            parameter.grad.detach()
            .float()
            .norm()
            .cpu()
        )

    optimizer_start = time.perf_counter()

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

    memory_after = process_memory_mb()

    cuda_peak_mb = 0.0

    if device == "cuda":
        cuda_peak_mb = (
            torch.cuda.max_memory_allocated()
            / 1024**2
        )

    return {
        "context_length":
            context_length,

        "loss":
            float(
                loss.detach().cpu()
            ),

        "forward_seconds":
            forward_seconds,

        "backward_seconds":
            backward_seconds,

        "optimizer_seconds":
            optimizer_seconds,

        "total_seconds":
            total_seconds,

        "tokens_per_second":
            (
                context_length
                / total_seconds
            ),

        "gradient_norm":
            gradient_norm,

        "gradient_tensors":
            gradient_tensors,

        "memory_before_mb":
            memory_before,

        "memory_after_mb":
            memory_after,

        "memory_delta_mb":
            max(
                0.0,
                memory_after
                - memory_before,
            ),

        "cuda_peak_mb":
            cuda_peak_mb,

        "device":
            device,
    }


def run_scaling_benchmark(
    *,
    packed_path: str = (
        "gene/data/training/packed/"
        "train.bin"
    ),
    contexts: tuple[int, ...] = (
        256,
        512,
        1024,
        2048,
    ),
    output_path: str = DEFAULT_OUTPUT,
) -> dict:

    if not Path(
        packed_path
    ).exists():
        raise FileNotFoundError(
            packed_path
        )

    output = Path(
        output_path
    )

    existing = {
        "model":
            gene_200m_config().to_dict(),

        "packed_path":
            packed_path,

        "device":
            (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            ),

        "torch_version":
            torch.__version__,

        "cpu_threads":
            torch.get_num_threads(),

        "results":
            [],
    }

    # Resume previously completed contexts.
    if output.exists():

        try:

            existing = json.loads(
                output.read_text(
                    encoding="utf-8-sig"
                )
            )

            print(
                "Existing benchmark results:",
                [
                    item["context_length"]
                    for item
                    in existing.get(
                        "results",
                        [],
                    )
                ],
            )

        except Exception:

            print(
                "Existing report could not be "
                "loaded; starting fresh."
            )

    completed = {
        item["context_length"]
        for item
        in existing.get(
            "results",
            [],
        )
    }

    save_report(
        existing,
        output_path,
    )

    for context_length in contexts:

        if context_length in completed:

            print(
                f"\n=== CONTEXT "
                f"{context_length}: SKIPPED "
                f"(already complete) ==="
            )

            continue

        print(
            f"\n=== CONTEXT "
            f"{context_length} ==="
        )

        started = time.perf_counter()

        try:

            result = run_one_context(
                packed_path=packed_path,
                context_length=
                    context_length,
            )

            result[
                "wall_clock_seconds"
            ] = (
                time.perf_counter()
                - started
            )

            existing[
                "results"
            ].append(
                result
            )

            # CRITICAL:
            # Save immediately after each successful
            # context. A later failure cannot erase
            # previous measurements.
            save_report(
                existing,
                output_path,
            )

            print(
                f"loss="
                f"{result['loss']:.6f} "
                f"time="
                f"{result['total_seconds']:.3f}s "
                f"tok/s="
                f"{result['tokens_per_second']:.2f} "
                f"RAM delta="
                f"{result['memory_delta_mb']:.2f}MB"
            )

            print(
                f"SAVED CONTEXT "
                f"{context_length}"
            )

        except KeyboardInterrupt:

            print(
                "\nBenchmark interrupted."
            )

            save_report(
                existing,
                output_path,
            )

            raise

        except Exception as exc:

            failure = {
                "context_length":
                    context_length,
                "error":
                    type(exc).__name__,
                "message":
                    str(exc),
            }

            existing.setdefault(
                "failures",
                [],
            ).append(
                failure
            )

            # Save even when the current context fails.
            save_report(
                existing,
                output_path,
            )

            print(
                f"FAILED CONTEXT "
                f"{context_length}: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            print(
                "Previous successful measurements "
                "remain saved."
            )

            break

    existing[
        "completed_contexts"
    ] = [
        item["context_length"]
        for item
        in existing.get(
            "results",
            [],
        )
    ]

    existing[
        "failed_contexts"
    ] = [
        item["context_length"]
        for item
        in existing.get(
            "failures",
            [],
        )
    ]

    save_report(
        existing,
        output_path,
    )

    return existing


if __name__ == "__main__":

    report = run_scaling_benchmark()

    print(
        "\n"
        + "=" * 72
    )

    print(
        "GENE 200M CONTEXT SCALING"
    )

    print(
        "=" * 72
    )

    print(
        "Completed:",
        report.get(
            "completed_contexts",
            [],
        ),
    )

    print(
        "Failed:",
        report.get(
            "failed_contexts",
            [],
        ),
    )

    for result in report.get(
        "results",
        [],
    ):

        print(
            f"{result['context_length']:>5} "
            f"tokens | "
            f"{result['total_seconds']:>8.3f}s | "
            f"{result['tokens_per_second']:>8.2f} tok/s | "
            f"RAM Δ="
            f"{result['memory_delta_mb']:.2f}MB | "
            f"loss="
            f"{result['loss']:.6f}"
        )

    print(
        f"\nReport: {DEFAULT_OUTPUT}"
    )

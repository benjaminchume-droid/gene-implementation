from __future__ import annotations

import json
import os
import time

from pathlib import Path

import torch

from gene.model.configs import gene_200m_config
from gene.model.neural import GeneTransformer
from gene.training.packing import CorpusPacker


def process_memory_mb() -> float:
    if os.name != "nt":
        return 0.0

    try:
        import ctypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
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
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            ctypes.c_ulong,
        ]

        get_memory.restype = ctypes.c_bool

        if not get_memory(
            process,
            ctypes.byref(counters),
            counters.cb,
        ):
            return 0.0

        return counters.WorkingSetSize / 1024**2

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
            f"Corpus has {len(tokens)} tokens; "
            f"{count} requested."
        )

    return torch.tensor(
        tokens[:count],
        dtype=torch.long,
    )


def benchmark(
    packed_path: str,
    context_length: int,
    threads: int,
) -> dict:

    torch.set_num_threads(threads)

    config = gene_200m_config()

    model = GeneTransformer(
        config
    )

    model.to("cpu")
    model.train()

    tokens = load_tokens(
        packed_path,
        context_length,
    )

    input_ids = tokens.unsqueeze(0)
    labels = input_ids.clone()
    attention_mask = torch.ones_like(
        input_ids
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        weight_decay=0.0,
    )

    # Warm-up to separate thread/runtime initialization
    # from the measured step.
    optimizer.zero_grad(
        set_to_none=True
    )

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    output["loss"].backward()
    optimizer.step()

    model.zero_grad(
        set_to_none=True
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    memory_before = process_memory_mb()

    start = time.perf_counter()

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    forward_seconds = (
        time.perf_counter() - start
    )

    loss = output["loss"]

    backward_start = time.perf_counter()

    loss.backward()

    backward_seconds = (
        time.perf_counter() - backward_start
    )

    optimizer_start = time.perf_counter()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0,
    )

    optimizer.step()

    optimizer_seconds = (
        time.perf_counter()
        - optimizer_start
    )

    total_seconds = (
        forward_seconds
        + backward_seconds
        + optimizer_seconds
    )

    memory_after = process_memory_mb()

    return {
        "threads":
            threads,
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
            context_length
            / total_seconds,
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
    }


def run(
    packed_path: str = (
        "gene/data/training/packed/"
        "train.bin"
    ),
    contexts=(512, 1024),
    threads=None,
) -> dict:

    cpu_count = os.cpu_count() or 1

    if threads is None:
        thread_values = sorted(
            {
                1,
                min(2, cpu_count),
                min(4, cpu_count),
                cpu_count,
            }
        )
    else:
        thread_values = [
            value
            for value in threads
            if value > 0
            and value <= cpu_count
        ]

    results = []

    output = Path(
        "gene/data/training/resources/"
        "thread_scaling.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for context in contexts:

        for thread_count in thread_values:

            print(
                f"\n=== CONTEXT {context} | "
                f"THREADS {thread_count} ==="
            )

            try:

                result = benchmark(
                    packed_path=packed_path,
                    context_length=context,
                    threads=thread_count,
                )

                results.append(result)

                print(
                    f"loss={result['loss']:.6f} "
                    f"time={result['total_seconds']:.3f}s "
                    f"tok/s={result['tokens_per_second']:.2f} "
                    f"RAM Δ="
                    f"{result['memory_delta_mb']:.2f}MB"
                )

            except Exception as exc:

                results.append(
                    {
                        "threads":
                            thread_count,
                        "context_length":
                            context,
                        "failed":
                            True,
                        "error":
                            type(exc).__name__,
                        "message":
                            str(exc),
                    }
                )

                print(
                    f"FAILED: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

    report = {
        "logical_cpus":
            cpu_count,
        "thread_values":
            thread_values,
        "contexts":
            list(contexts),
        "results":
            results,
    }

    output.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    return report


if __name__ == "__main__":

    report = run()

    print(
        "\n"
        + "=" * 72
    )

    print(
        "GENE CPU THREAD SCALING"
    )

    print(
        "=" * 72
    )

    for result in report["results"]:

        if result.get("failed"):
            print(
                f"context={result['context_length']} "
                f"threads={result['threads']} "
                f"FAILED"
            )
            continue

        print(
            f"context={result['context_length']:>4} "
            f"threads={result['threads']:>2} "
            f"tok/s="
            f"{result['tokens_per_second']:.2f}"
        )

    print(
        "\nReport:",
        "gene/data/training/resources/"
        "thread_scaling.json",
    )

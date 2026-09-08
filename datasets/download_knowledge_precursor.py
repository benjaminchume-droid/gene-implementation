import json
import os
import sys
import time
from pathlib import Path

from datasets import load_dataset

ROOT = Path("gene/datasets/knowledge_precursor")
ROOT.mkdir(parents=True, exist_ok=True)

# Keep every locally-created shard below 500 MB.
CHUNK_BYTES = 400 * 1024 * 1024

DATASETS = [
    {
        "name": "fineweb_edu",
        "repo": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "split": "train",
        "target_bytes": 400 * 1024 * 1024,
    },
    {
        "name": "fineweb",
        "repo": "HuggingFaceFW/fineweb",
        "config": "sample-10BT",
        "split": "train",
        "target_bytes": 250 * 1024 * 1024,
    },
    {
        "name": "wikipedia",
        "repo": "wikimedia/wikipedia",
        "config": "20231101.en",
        "split": "train",
        "target_bytes": 300 * 1024 * 1024,
    },
    {
        "name": "openwebmath",
        "repo": "open-web-math/open-web-math",
        "config": None,
        "split": "train",
        "target_bytes": 300 * 1024 * 1024,
    },
    {
        "name": "cosmopedia_v2",
        "repo": "HuggingFaceTB/smollm-corpus",
        "config": "cosmopedia-v2",
        "split": "train",
        "target_bytes": 250 * 1024 * 1024,
    },
    {
        "name": "tulu3_sft",
        "repo": "allenai/tulu-3-sft-mixture",
        "config": None,
        "split": "train",
        "target_bytes": 300 * 1024 * 1024,
    },
]

TOTAL_LIMIT = 2 * 1024 * 1024 * 1024


def current_total():
    total = 0
    for path in ROOT.rglob("*"):
        if path.is_file() and not path.name.endswith(".json"):
            total += path.stat().st_size
    return total


def normalize_item(item):
    """
    Preserve the source fields rather than pretending different datasets
    share the same schema.
    """
    return item


def write_stream(dataset_cfg):
    name = dataset_cfg["name"]
    target = min(
        dataset_cfg["target_bytes"],
        CHUNK_BYTES,
    )

    existing = sorted(
        ROOT.joinpath(name).glob("chunk_*.jsonl")
    )

    for path in existing:
        if path.stat().st_size >= int(target * 0.98):
            print(
                f"[SKIP] {name}: completed chunk "
                f"{path.name} "
                f"({path.stat().st_size / 1024 / 1024:.1f} MB)"
            )
            return True

    out_dir = ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk = out_dir / "chunk_000.jsonl"

    # Resume by truncating any incomplete final JSON line.
    if chunk.exists():
        data = chunk.read_bytes()

        if data and not data.endswith(b"\n"):
            last_newline = data.rfind(b"\n")

            if last_newline >= 0:
                with chunk.open("r+b") as f:
                    f.truncate(last_newline + 1)

    print("")
    print("=" * 72)
    print(f"DATASET: {name}")
    print(f"REPOSITORY: {dataset_cfg['repo']}")
    print(f"TARGET CHUNK: {target / 1024 / 1024:.0f} MB")
    print("=" * 72)

    kwargs = {
        "path": dataset_cfg["repo"],
        "split": dataset_cfg["split"],
        "streaming": True,
    }

    if dataset_cfg["config"]:
        kwargs["name"] = dataset_cfg["config"]

    try:
        ds = load_dataset(**kwargs)
    except Exception as exc:
        print(f"[FAILED OPEN] {name}: {exc}")
        return False

    written = chunk.stat().st_size if chunk.exists() else 0

    try:
        with chunk.open("ab") as f:
            for item in ds:
                record = normalize_item(item)

                line = (
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")

                if written + len(line) > target:
                    break

                f.write(line)
                written += len(line)

                if written % (25 * 1024 * 1024) < len(line):
                    print(
                        f"  {written / 1024 / 1024:.0f} MB"
                    )

                if current_total() >= TOTAL_LIMIT:
                    print("[STOP] 2 GB global limit reached.")
                    return False

        print(
            f"[DONE] {name}: "
            f"{written / 1024 / 1024:.1f} MB"
        )
        return True

    except KeyboardInterrupt:
        print(
            f"[PAUSED] {name}: "
            f"{written / 1024 / 1024:.1f} MB"
        )
        print("Re-run the same command to continue.")
        return False

    except Exception as exc:
        print(
            f"[NETWORK/READ ERROR] {name}: {exc}"
        )
        print(
            "Re-run the same command to continue."
        )
        return False


def main():
    print("=" * 72)
    print("GENE KNOWLEDGE PRECURSOR ACQUISITION")
    print("=" * 72)
    print("Global payload limit: 2 GB")
    print("Local chunks: <= 400 MB")
    print("Streaming download: ENABLED")
    print("Resume strategy: local partial-file reuse")
    print("")

    for cfg in DATASETS:

        total = current_total()

        if total >= TOTAL_LIMIT:
            print("2 GB limit reached.")
            break

        write_stream(cfg)

    total = current_total()

    print("")
    print("=" * 72)
    print("ACQUISITION STATUS")
    print("=" * 72)

    for path in sorted(ROOT.rglob("*.jsonl")):
        size = path.stat().st_size
        print(
            f"{size / 1024 / 1024:8.1f} MB  "
            f"{path.relative_to(ROOT)}"
        )

    print("")
    print(
        f"TOTAL: {total / 1024 / 1024:.1f} MB "
        f"({total / 1024 / 1024 / 1024:.2f} GB)"
    )

    manifest = ROOT / "manifest.json"

    manifest.write_text(
        json.dumps(
            {
                "version": "1.0",
                "global_limit_bytes": TOTAL_LIMIT,
                "chunk_limit_bytes": CHUNK_BYTES,
                "datasets": [
                    {
                        "name": cfg["name"],
                        "repository": cfg["repo"],
                        "config": cfg["config"],
                        "split": cfg["split"],
                    }
                    for cfg in DATASETS
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()

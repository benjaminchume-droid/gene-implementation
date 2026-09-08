from pathlib import Path
import json

from datasets import load_dataset

BASE = Path(__file__).resolve().parent

RAW = BASE / "raw"
SPECIAL = BASE / "specializations"
MANIFESTS = BASE / "manifests"


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0

    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def save_manifest(
    name: str,
    repo: str,
    output: Path,
    requested: int,
    downloaded: int,
    status: str,
    error: str | None = None,
):
    manifest = {
        "name": name,
        "repository": repo,
        "requested": requested,
        "downloaded": downloaded,
        "status": status,
        "output": str(output),
    }

    if error:
        manifest["error"] = error

    (MANIFESTS / f"{name}.json").write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def acquire(
    name: str,
    repo: str,
    split: str,
    output: Path,
    limit: int,
    config: str | None = None,
):
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = count_lines(output)

    if existing >= limit:
        print(
            f"SKIP {name}: "
            f"{existing:,}/{limit:,} rows already present"
        )

        save_manifest(
            name,
            repo,
            output,
            limit,
            existing,
            "complete",
        )
        return

    print()
    print("=" * 70)
    print(f"DOWNLOADING: {name}")
    print("=" * 70)
    print(f"Repository: {repo}")
    print(f"Target:     {limit:,}")

    kwargs = {
        "split": split,
        "streaming": True,
    }

    if config:
        kwargs["name"] = config

    try:
        dataset = load_dataset(
            repo,
            **kwargs,
        )

        count = 0

        with output.open(
            "w",
            encoding="utf-8",
        ) as f:
            for row in dataset:

                f.write(
                    json.dumps(
                        {
                            "dataset": name,
                            "source": repo,
                            "data": dict(row),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                count += 1

                if count >= limit:
                    break

        save_manifest(
            name,
            repo,
            output,
            limit,
            count,
            "complete",
        )

        print(
            f"SUCCESS: {count:,} rows"
        )
        print(
            f"Saved: {output}"
        )

    except Exception as error:

        save_manifest(
            name,
            repo,
            output,
            limit,
            0,
            "failed",
            str(error),
        )

        print(
            f"FAILED: {error}"
        )


def main():

    print()
    print("=" * 70)
    print("VELOCITY AI — FINAL DATA ACQUISITION")
    print("=" * 70)

    # ---------------------------------------------------------
    # LIMA
    # Direct file because the repository still exposes a
    # legacy dataset script incompatible with current Datasets.
    # ---------------------------------------------------------

    lima = RAW / "core" / "lima" / "train.jsonl"

    if lima.exists():

        rows = count_lines(lima)

        print()
        print(
            f"LIMA: existing file detected "
            f"({rows:,} rows)"
        )

        save_manifest(
            "lima",
            "GAIR/lima",
            lima,
            1030,
            rows,
            "complete",
        )

    else:

        print()
        print("LIMA is missing.")
        print(
            "Run:"
        )
        print(
            "hf download GAIR/lima train.jsonl "
            "--repo-type dataset "
            "--local-dir GENE\\datasets\\raw\\core\\lima"
        )

    # =========================================================
    # GENE CORE
    # =========================================================

    acquire(
        "ultrachat",
        "HuggingFaceH4/ultrachat_200k",
        "train_sft",
        RAW / "core" / "ultrachat.jsonl",
        5000,
    )

    acquire(
        "openorca",
        "Open-Orca/OpenOrca",
        "train",
        RAW / "core" / "openorca.jsonl",
        10000,
    )

    acquire(
        "oasst1",
        "OpenAssistant/oasst1",
        "train",
        RAW / "core" / "oasst1.jsonl",
        5000,
    )

    # =========================================================
    # GENE AGENTS
    # =========================================================

    acquire(
        "glaive_function_calling",
        "glaiveai/glaive-function-calling-v2",
        "train",
        RAW / "agent" / "glaive_function_calling.jsonl",
        10000,
    )

    acquire(
        "magicoder",
        "ise-uiuc/Magicoder-OSS-Instruct-75K",
        "train",
        RAW / "future_agents" / "magicoder.jsonl",
        5000,
    )

    # =========================================================
    # LUMEN MIND
    # =========================================================

    acquire(
        "mind2web",
        "osunlp/Mind2Web",
        "train",
        SPECIAL / "lumen_mind" / "mind2web.jsonl",
        1000,
    )

    acquire(
        "weblinx",
        "McGill-NLP/WebLINX",
        "train",
        SPECIAL / "lumen_mind" / "weblinx.jsonl",
        5000,
    )

    acquire(
        "ms_marco",
        "microsoft/ms_marco",
        "train",
        SPECIAL / "lumen_mind" / "ms_marco.jsonl",
        10000,
        config="v1.1",
    )

    # =========================================================
    # LUMEN FORGE
    # =========================================================

    acquire(
        "webnlg",
        "GEM/web_nlg",
        "train",
        SPECIAL / "lumen_forge" / "webnlg.jsonl",
        5000,
        config="en",
    )

    # =========================================================
    # EXAMFORGE
    # =========================================================

    acquire(
        "sciq",
        "allenai/sciq",
        "train",
        SPECIAL / "examforge" / "sciq.jsonl",
        10000,
    )

    acquire(
        "arc_challenge",
        "allenai/ai2_arc",
        "train",
        SPECIAL / "examforge" / "arc_challenge.jsonl",
        1119,
        config="ARC-Challenge",
    )

    # =========================================================
    # RELAY
    # =========================================================

    acquire(
        "daily_dialog",
        "OpenRL/daily_dialog",
        "train",
        SPECIAL / "relay" / "daily_dialog.jsonl",
        10000,
    )

    acquire(
        "soda",
        "allenai/soda",
        "train",
        SPECIAL / "relay" / "soda.jsonl",
        10000,
    )

    # =========================================================
    # PIXA
    # Prompt/metadata only — NO images.
    # =========================================================

    acquire(
        "diffusiondb",
        "poloclub/diffusiondb",
        "train",
        SPECIAL / "pixa" / "diffusiondb.jsonl",
        5000,
        config="2k_first_prompts",
    )

    print()
    print("=" * 70)
    print("DATA ACQUISITION FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()


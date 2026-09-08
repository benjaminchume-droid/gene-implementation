from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from gene.training.tokenizer import GeneTokenizer


@dataclass
class PackingStats:
    source_files: list[str]
    output_file: str
    block_size: int

    records_seen: int = 0
    records_tokenized: int = 0
    records_skipped: int = 0

    tokens_seen: int = 0
    blocks_written: int = 0

    trailing_tokens: int = 0
    eos_token_id: int = 3

    created_at: str = ""


class CorpusPacker:

    def __init__(
        self,
        tokenizer: GeneTokenizer,
        block_size: int = 8192,
        eos_token: str = "<eos>",
    ) -> None:

        if block_size <= 1:
            raise ValueError(
                "block_size must be greater than 1."
            )

        self.tokenizer = tokenizer
        self.block_size = block_size

        eos_id = tokenizer.token_id(
            eos_token
        )

        if eos_id is None:
            raise ValueError(
                f"Tokenizer does not contain {eos_token}."
            )

        self.eos_token_id = eos_id

    @staticmethod
    def _iter_jsonl_text(
        files: list[str],
        text_key: str = "text",
    ) -> Iterator[str]:

        for file_name in files:

            path = Path(file_name)

            if not path.exists():
                raise FileNotFoundError(
                    str(path)
                )

            with path.open(
                "r",
                encoding="utf-8-sig",
            ) as handle:

                for line in handle:

                    line = line.strip()

                    if not line:
                        continue

                    try:
                        item = json.loads(
                            line
                        )
                    except json.JSONDecodeError:
                        continue

                    if not isinstance(
                        item,
                        dict,
                    ):
                        continue

                    value = item.get(
                        text_key
                    )

                    if (
                        isinstance(value, str)
                        and value.strip()
                    ):
                        yield value

    def token_stream(
        self,
        files: list[str],
        stats: PackingStats,
        text_key: str = "text",
    ) -> Iterator[int]:

        for text in self._iter_jsonl_text(
            files,
            text_key=text_key,
        ):

            stats.records_seen += 1

            try:
                ids = self.tokenizer.encode(
                    text
                )
            except Exception:
                stats.records_skipped += 1
                continue

            if not ids:
                stats.records_skipped += 1
                continue

            stats.records_tokenized += 1
            stats.tokens_seen += len(ids)

            for token_id in ids:
                yield token_id

            yield self.eos_token_id
            stats.tokens_seen += 1

    def pack(
        self,
        files: list[str],
        output_file: str,
        text_key: str = "text",
    ) -> dict:

        if not files:
            raise ValueError(
                "No source files supplied."
            )

        output = Path(output_file)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        stats = PackingStats(
            source_files=list(files),
            output_file=str(output),
            block_size=self.block_size,
            eos_token_id=self.eos_token_id,
            created_at=datetime.now(
                timezone.utc
            ).isoformat(),
        )

        # uint32 supports the tokenizer's integer vocabulary
        # safely and keeps the packed representation simple.
        import struct

        buffer: list[int] = []

        with output.open(
            "wb"
        ) as handle:

            for token_id in self.token_stream(
                files,
                stats,
                text_key=text_key,
            ):

                buffer.append(
                    token_id
                )

                if len(buffer) < self.block_size:
                    continue

                handle.write(
                    struct.pack(
                        f"<{self.block_size}I",
                        *buffer,
                    )
                )

                stats.blocks_written += 1
                buffer.clear()

        stats.trailing_tokens = len(
            buffer
        )

        manifest = {
            "format":
                "gene_packed_uint32_v1",

            "stats":
                asdict(stats),

            "tokenizer":
                str(
                    getattr(
                        self.tokenizer,
                        "manifest_path",
                        "",
                    )
                ),
        }

        manifest_path = (
            output.with_suffix(
                ".manifest.json"
            )
        )

        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return {
            "success": True,
            "output":
                str(output),
            "manifest":
                str(manifest_path),
            "blocks":
                stats.blocks_written,
            "tokens_seen":
                stats.tokens_seen,
            "trailing_tokens":
                stats.trailing_tokens,
            "records_seen":
                stats.records_seen,
            "records_tokenized":
                stats.records_tokenized,
            "records_skipped":
                stats.records_skipped,
        }

    @staticmethod
    def read_blocks(
        path: str,
        block_size: int,
    ) -> Iterator[list[int]]:

        import struct

        bytes_per_block = (
            block_size * 4
        )

        with Path(path).open(
            "rb"
        ) as handle:

            while True:

                payload = handle.read(
                    bytes_per_block
                )

                if not payload:
                    break

                if len(payload) != bytes_per_block:
                    raise ValueError(
                        "Packed corpus contains "
                        "an incomplete block."
                    )

                yield list(
                    struct.unpack(
                        f"<{block_size}I",
                        payload,
                    )
                )

    @staticmethod
    def sha256(
        path: str,
    ) -> str:

        digest = hashlib.sha256()

        with Path(path).open(
            "rb"
        ) as handle:

            for chunk in iter(
                lambda:
                handle.read(1024 * 1024),
                b"",
            ):
                digest.update(chunk)

        return digest.hexdigest()

from __future__ import annotations

import hashlib
import json

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


DEFAULT_SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
]


class GeneTokenizer:

    def __init__(
        self,
        vocab_size: int = 32768,
        min_frequency: int = 2,
        special_tokens: list[str] | None = None,
    ) -> None:

        if vocab_size <= 0:
            raise ValueError(
                "vocab_size must be positive."
            )

        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

        self.special_tokens = (
            special_tokens
            or DEFAULT_SPECIAL_TOKENS.copy()
        )

        self.tokenizer = Tokenizer(
            BPE(
                unk_token="<unk>"
            )
        )

        self.tokenizer.pre_tokenizer = (
            ByteLevel(
                add_prefix_space=True
            )
        )

        self.tokenizer.decoder = (
            ByteLevelDecoder()
        )

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

                for line_number, line in enumerate(
                    handle,
                    start=1,
                ):

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

                    text = item.get(
                        text_key
                    )

                    if (
                        isinstance(text, str)
                        and text.strip()
                    ):
                        yield text

    @staticmethod
    def _iter_plain_text(
        files: list[str],
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
                errors="replace",
            ) as handle:

                for line in handle:

                    line = line.strip()

                    if line:
                        yield line

    def train_jsonl(
        self,
        files: list[str],
        text_key: str = "text",
    ) -> dict:

        if not files:
            raise ValueError(
                "No training files supplied."
            )

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            show_progress=True,
        )

        self.tokenizer.train_from_iterator(
            self._iter_jsonl_text(
                files,
                text_key=text_key,
            ),
            trainer=trainer,
        )

        return self.metadata(
            source_files=files,
            source_format="jsonl",
        )

    def train_text(
        self,
        files: list[str],
    ) -> dict:

        if not files:
            raise ValueError(
                "No training files supplied."
            )

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            show_progress=True,
        )

        self.tokenizer.train_from_iterator(
            self._iter_plain_text(
                files
            ),
            trainer=trainer,
        )

        return self.metadata(
            source_files=files,
            source_format="text",
        )

    def save(
        self,
        path: str,
        metadata: dict | None = None,
    ) -> dict:

        target = Path(path)

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.tokenizer.save(
            str(target)
        )

        manifest_path = (
            target.with_suffix(
                ".manifest.json"
            )
        )

        manifest = {
            "created_at":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "tokenizer":
                str(target),

            "vocab_size":
                self.tokenizer.get_vocab_size(),

            "configured_vocab_size":
                self.vocab_size,

            "min_frequency":
                self.min_frequency,

            "special_tokens":
                self.special_tokens,

            **(metadata or {}),
        }

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
            "tokenizer":
                str(target),
            "manifest":
                str(manifest_path),
            "vocab_size":
                self.tokenizer.get_vocab_size(),
        }

    @classmethod
    def load(
        cls,
        path: str,
    ) -> "GeneTokenizer":

        target = Path(path)

        if not target.exists():
            raise FileNotFoundError(
                str(target)
            )

        instance = cls()

        instance.tokenizer = (
            Tokenizer.from_file(
                str(target)
            )
        )

        instance.vocab_size = (
            instance.tokenizer.get_vocab_size()
        )

        return instance

    def encode(
        self,
        text: str,
    ) -> list[int]:

        return self.tokenizer.encode(
            text
        ).ids

    def decode(
        self,
        ids: list[int],
    ) -> str:

        return self.tokenizer.decode(
            ids,
            skip_special_tokens=False,
        )

    def vocab_size_actual(self) -> int:

        return self.tokenizer.get_vocab_size()

    def token_id(
        self,
        token: str,
    ) -> int | None:

        return self.tokenizer.token_to_id(
            token
        )

    @staticmethod
    def file_hash(
        path: str,
    ) -> str:

        digest = hashlib.sha256()

        with Path(path).open(
            "rb"
        ) as handle:

            for chunk in iter(
                lambda:
                    handle.read(
                        1024 * 1024
                    ),
                b"",
            ):
                digest.update(
                    chunk
                )

        return digest.hexdigest()

    def metadata(
        self,
        source_files: list[str],
        source_format: str,
    ) -> dict:

        return {
            "source_files":
                source_files,

            "source_format":
                source_format,

            "requested_vocab_size":
                self.vocab_size,

            "actual_vocab_size":
                self.tokenizer.get_vocab_size(),

            "min_frequency":
                self.min_frequency,

            "special_tokens":
                self.special_tokens,
        }


def train_gene_tokenizer(
    files: list[str],
    output: str = (
        "gene/data/training/tokenizer/"
        "gene-tokenizer.json"
    ),
    vocab_size: int = 32768,
    min_frequency: int = 2,
) -> dict:

    tokenizer = GeneTokenizer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )

    metadata = tokenizer.train_jsonl(
        files
    )

    return tokenizer.save(
        output,
        metadata=metadata,
    )


class TokenizerTrainer:
    """
    Backward-compatible wrapper around GeneTokenizer.

    Existing callers can continue using TokenizerTrainer
    while the production tokenizer uses GeneTokenizer.
    """

    def __init__(
        self,
        vocab_size: int = 32768,
        min_frequency: int = 2,
    ) -> None:

        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

    def train(
        self,
        files: list[str],
        output: str = (
            "gene/data/training/tokenizer/"
            "gene-tokenizer.json"
        ),
    ) -> dict:

        tokenizer = GeneTokenizer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
        )

        metadata = tokenizer.train_jsonl(
            files
        )

        return tokenizer.save(
            output,
            metadata=metadata,
        )


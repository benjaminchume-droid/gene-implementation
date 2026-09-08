from __future__ import annotations

import inspect
import json

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from gene.model.configs import gene_200m_config
from gene.model.neural import GeneTransformer
from gene.training.tokenizer import GeneTokenizer


@dataclass
class ChatMessage:

    role: str
    content: str

    timestamp: str = field(
        default_factory=lambda:
        datetime.now(
            timezone.utc
        ).isoformat()
    )


@dataclass
class ChatSession:

    session_id: str

    messages: list[
        ChatMessage
    ] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def add(
        self,
        role: str,
        content: str,
    ) -> ChatMessage:

        message = ChatMessage(
            role=role,
            content=content,
        )

        self.messages.append(
            message
        )

        return message

    def clear(self) -> None:

        self.messages.clear()

    def to_dict(self) -> dict:

        return {
            "session_id":
                self.session_id,
            "messages": [
                asdict(message)
                for message
                in self.messages
            ],
            "metadata":
                self.metadata,
        }


class SessionStore:

    def __init__(
        self,
        root: str = (
            "gene/data/conversations"
        ),
    ) -> None:

        self.root = Path(root)

        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        session: ChatSession,
    ) -> str:

        path = (
            self.root
            / f"{session.session_id}.json"
        )

        temporary = path.with_suffix(
            ".tmp"
        )

        temporary.write_text(
            json.dumps(
                session.to_dict(),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        temporary.replace(path)

        return str(path)

    def load(
        self,
        session_id: str,
    ) -> ChatSession:

        path = (
            self.root
            / f"{session_id}.json"
        )

        payload = json.loads(
            path.read_text(
                encoding="utf-8-sig"
            )
        )

        session = ChatSession(
            session_id=
                payload["session_id"],
            metadata=
                payload.get(
                    "metadata",
                    {},
                ),
        )

        for item in payload.get(
            "messages",
            [],
        ):

            session.messages.append(
                ChatMessage(
                    role=item["role"],
                    content=item["content"],
                    timestamp=item.get(
                        "timestamp",
                        "",
                    ),
                )
            )

        return session


class GeneInferenceEngine:

    def __init__(
        self,
        *,
        tokenizer_path: str = (
            "gene/data/training/"
            "tokenizer/gene-tokenizer.json"
        ),
        checkpoint_path:
            str | None = None,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:

        self.tokenizer_path = (
            Path(tokenizer_path)
        )

        self.checkpoint_path = (
            Path(checkpoint_path)
            if checkpoint_path
            else None
        )

        self.device = (
            device
            or (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.dtype = (
            dtype
            or torch.float32
        )

        self.tokenizer = None
        self.model = None

        self.loaded = False

    def load(self) -> None:

        if not self.tokenizer_path.exists():

            raise FileNotFoundError(
                "Tokenizer not found: "
                f"{self.tokenizer_path}"
            )

        self.tokenizer = (
            GeneTokenizer.load(
                str(
                    self.tokenizer_path
                )
            )
        )

        config = gene_200m_config()

        tokenizer_vocab = (
            self.tokenizer
            .vocab_size_actual()
        )

        if (
            tokenizer_vocab
            != config.vocab_size
        ):

            raise ValueError(
                "Tokenizer/model vocabulary "
                f"mismatch: model="
                f"{config.vocab_size}, "
                f"tokenizer="
                f"{tokenizer_vocab}"
            )

        self.model = GeneTransformer(
            config
        )

        if self.checkpoint_path:

            if not self.checkpoint_path.exists():

                raise FileNotFoundError(
                    "Checkpoint not found: "
                    f"{self.checkpoint_path}"
                )

            self._load_checkpoint(
                self.checkpoint_path
            )

        self.model.to(
            self.device
        )

        self.model.eval()

        self.loaded = True

    def _load_checkpoint(
        self,
        path: Path,
    ) -> None:

        checkpoint = torch.load(
            str(path),
            map_location="cpu",
        )

        state_dict = None

        if isinstance(
            checkpoint,
            dict,
        ):

            for key in (
                "model_state_dict",
                "state_dict",
                "model",
            ):

                candidate = (
                    checkpoint.get(
                        key
                    )
                )

                if isinstance(
                    candidate,
                    dict,
                ):

                    state_dict = candidate
                    break

        if state_dict is None:

            if isinstance(
                checkpoint,
                dict,
            ):
                state_dict = checkpoint
            else:
                raise ValueError(
                    "Unsupported checkpoint format."
                )

        cleaned = {}

        for key, value in (
            state_dict.items()
        ):

            if key.startswith(
                "module."
            ):

                key = key[
                    len("module.") :
                ]

            cleaned[key] = value

        missing, unexpected = (
            self.model.load_state_dict(
                cleaned,
                strict=False,
            )
        )

        if missing:

            print(
                "Checkpoint missing keys:",
                len(missing),
            )

        if unexpected:

            print(
                "Checkpoint unexpected keys:",
                len(unexpected),
            )

    def _format_messages(
        self,
        messages: list[ChatMessage],
    ) -> str:

        lines = []

        for message in messages:

            role = (
                message.role.strip()
                or "user"
            )

            lines.append(
                f"{role}: "
                f"{message.content.strip()}"
            )

        lines.append(
            "assistant:"
        )

        return "\n".join(
            lines
        )

    def encode_conversation(
        self,
        session: ChatSession,
    ) -> list[int]:

        prompt = self._format_messages(
            session.messages
        )

        return self.tokenizer.encode(
            prompt
        )

    def _generate(
        self,
        input_ids: list[int],
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> list[int]:

        if self.model is None:

            raise RuntimeError(
                "Inference model is not loaded."
            )

        tensor = torch.tensor(
            [input_ids],
            dtype=torch.long,
            device=self.device,
        )

        generate = getattr(
            self.model,
            "generate",
            None,
        )

        if generate is None:
            raise RuntimeError(
                "GeneTransformer does not "
                "currently expose generate()."
            )

        signature = inspect.signature(
            generate
        )

        kwargs = {}

        if (
            "max_new_tokens"
            in signature.parameters
        ):
            kwargs[
                "max_new_tokens"
            ] = max_new_tokens

        if (
            "temperature"
            in signature.parameters
        ):
            kwargs[
                "temperature"
            ] = temperature

        if (
            "top_k"
            in signature.parameters
        ):
            kwargs[
                "top_k"
            ] = top_k

        if (
            "do_sample"
            in signature.parameters
        ):

            kwargs[
                "do_sample"
            ] = temperature > 0

        with torch.no_grad():

            output = generate(
                tensor,
                **kwargs,
            )

        if isinstance(
            output,
            tuple,
        ):
            output = output[0]

        if isinstance(
            output,
            torch.Tensor,
        ):

            return (
                output[0]
                .detach()
                .cpu()
                .tolist()
            )

        if isinstance(
            output,
            list,
        ):

            if output and isinstance(
                output[0],
                list,
            ):
                return output[0]

            return output

        raise TypeError(
            "Unsupported generation output: "
            f"{type(output)}"
        )

    def _remove_prompt(
        self,
        prompt_ids: list[int],
        output_ids: list[int],
    ) -> list[int]:

        if (
            len(output_ids)
            >= len(prompt_ids)
            and
            output_ids[
                :len(prompt_ids)
            ]
            == prompt_ids
        ):

            return output_ids[
                len(prompt_ids):
            ]

        return output_ids

    def generate_reply(
        self,
        session: ChatSession,
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 40,
    ) -> str:

        if not self.loaded:
            self.load()

        prompt_ids = (
            self.encode_conversation(
                session
            )
        )

        if not prompt_ids:

            raise ValueError(
                "Conversation produced no tokens."
            )

        if len(prompt_ids) >= (
            self.model.config
            .max_position_embeddings
        ):

            raise ValueError(
                "Conversation context exceeds "
                "the model context window."
            )

        output_ids = self._generate(
            prompt_ids,
            max_new_tokens=
                max_new_tokens,
            temperature=
                temperature,
            top_k=top_k,
        )

        new_ids = (
            self._remove_prompt(
                prompt_ids,
                output_ids,
            )
        )

        response = (
            self.tokenizer
            .decode(new_ids)
            .strip()
        )

        return response


def create_session(
    session_id: str | None = None,
) -> ChatSession:

    import uuid

    return ChatSession(
        session_id=(
            session_id
            or (
                "chat-"
                + uuid.uuid4().hex
            )
        )
    )

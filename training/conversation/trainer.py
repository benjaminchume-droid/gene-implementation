from __future__ import annotations

import json
import time

from pathlib import Path

import torch

from gene.model.configs import (
    gene_200m_config,
)
from gene.model.neural import (
    GeneTransformer,
)
from gene.training.tokenizer import (
    GeneTokenizer,
)


class ConversationTrainer:

    def __init__(
        self,
        *,
        tokenizer_path: str = (
            "gene/data/training/"
            "tokenizer/gene-tokenizer.json"
        ),
        output_dir: str = (
            "gene/data/training/"
            "conversation/checkpoints"
        ),
        device: str | None = None,
    ) -> None:

        self.tokenizer_path = Path(
            tokenizer_path
        )

        self.output_dir = Path(
            output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.device = (
            device
            or (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.tokenizer = None
        self.model = None

    def load(self) -> None:

        self.tokenizer = (
            GeneTokenizer.load(
                str(
                    self.tokenizer_path
                )
            )
        )

        config = gene_200m_config()

        if (
            config.vocab_size
            != self.tokenizer.vocab_size_actual()
        ):
            raise ValueError(
                "Tokenizer/model vocabulary mismatch."
            )

        self.model = GeneTransformer(
            config
        )

        self.model.to(
            self.device
        )

        self.model.train()

    @staticmethod
    def _join_messages(
        messages,
    ):
        pieces = []

        for message in messages:

            pieces.append(
                (
                    message["role"].strip()
                    + ": "
                    + message[
                        "content"
                    ].strip()
                )
            )

        return "\n".join(
            pieces
        )

    def encode_example(
        self,
        messages,
        max_length: int = 256,
    ):

        # Build the complete conversational sequence.
        full_text = self._join_messages(
            messages
        )

        full_ids = self.tokenizer.encode(
            full_text
        )

        if len(full_ids) > max_length:
            full_ids = full_ids[
                :max_length
            ]

        labels = [
            -100
            for _ in full_ids
        ]

        # Determine the assistant span for the final
        # assistant response.
        assistant_positions = []

        for index, message in enumerate(
            messages
        ):

            if (
                message["role"]
                == "assistant"
            ):
                assistant_positions.append(
                    index
                )

        if not assistant_positions:
            raise ValueError(
                "Conversation example has no "
                "assistant response."
            )

        target_index = (
            assistant_positions[-1]
        )

        prefix_messages = (
            messages[
                :target_index
            ]
        )

        prefix_text = (
            self._join_messages(
                prefix_messages
            )
            + "\nassistant:"
        )

        prefix_ids = (
            self.tokenizer.encode(
                prefix_text
            )
        )

        response_start = len(
            prefix_ids
        )

        response_start = min(
            response_start,
            len(full_ids),
        )

        for index in range(
            response_start,
            len(full_ids),
        ):
            labels[index] = (
                full_ids[index]
            )

        input_ids = torch.tensor(
            [full_ids],
            dtype=torch.long,
        )

        labels = torch.tensor(
            [labels],
            dtype=torch.long,
        )

        return (
            input_ids,
            labels,
        )

    def train(
        self,
        dataset_path: str,
        *,
        steps: int = 10,
        max_length: int = 256,
        learning_rate: float = 2e-5,
        checkpoint_every: int = 5,
    ) -> dict:

        if steps <= 0:
            raise ValueError(
                "steps must be positive."
            )

        self.load()

        dataset = []

        with Path(
            dataset_path
        ).open(
            "r",
            encoding="utf-8-sig",
        ) as handle:

            for line in handle:

                if line.strip():

                    dataset.append(
                        json.loads(line)
                    )

        if not dataset:
            raise ValueError(
                "Conversation dataset is empty."
            )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.0,
        )

        records = []

        print(
            f"Device: {self.device}"
        )

        print(
            "Parameters:",
            f"{self.model.parameter_count():,}",
        )

        for step in range(
            1,
            steps + 1,
        ):

            example = dataset[
                (step - 1)
                % len(dataset)
            ]

            input_ids, labels = (
                self.encode_example(
                    example[
                        "messages"
                    ],
                    max_length=
                        max_length,
                )
            )

            input_ids = input_ids.to(
                self.device
            )

            labels = labels.to(
                self.device
            )

            optimizer.zero_grad(
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

            optimizer.step()

            elapsed = (
                time.perf_counter()
                - started
            )

            record = {
                "step": step,
                "loss":
                    float(
                        loss.detach().cpu()
                    ),
                "gradient_norm":
                    float(
                        gradient_norm
                    ),
                "seconds":
                    elapsed,
            }

            records.append(
                record
            )

            print(
                f"step={step} "
                f"loss={record['loss']:.6f} "
                f"time={elapsed:.3f}s"
            )

            if (
                step % checkpoint_every
                == 0
            ):

                self.save_checkpoint(
                    step,
                    optimizer,
                    record,
                )

        final_path = (
            self.output_dir
            / "gene-conversation-baseline-final.pt"
        )

        torch.save(
            {
                "model_state_dict":
                    self.model.state_dict(),

                "model_config":
                    self.model.config.to_dict(),

                "steps":
                    steps,

                "training_records":
                    records,

                "tokenizer_path":
                    str(
                        self.tokenizer_path
                    ),

                "training_type":
                    "conversation_baseline",
            },
            final_path,
        )

        report = {
            "success":
                True,

            "checkpoint":
                str(final_path),

            "steps":
                steps,

            "first_loss":
                records[0]["loss"],

            "last_loss":
                records[-1]["loss"],

            "loss_reduced":
                (
                    records[-1]["loss"]
                    <
                    records[0]["loss"]
                ),

            "records":
                records,
        }

        report_path = (
            self.output_dir.parent
            / "conversation_baseline_report.json"
        )

        report_path.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return report

    def save_checkpoint(
        self,
        step,
        optimizer,
        metrics,
    ) -> str:

        path = (
            self.output_dir
            / f"step-{step}.pt"
        )

        torch.save(
            {
                "model_state_dict":
                    self.model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "step":
                    step,

                "metrics":
                    metrics,

                "model_config":
                    self.model.config.to_dict(),

                "tokenizer_path":
                    str(
                        self.tokenizer_path
                    ),

                "training_type":
                    "conversation_baseline",
            },
            path,
        )

        return str(path)

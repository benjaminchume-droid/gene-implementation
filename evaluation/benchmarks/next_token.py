from __future__ import annotations

import torch

from .base import Benchmark
from ..models import (
    EvaluationExample,
    EvaluationResult,
)


class NextTokenBenchmark(Benchmark):

    @property
    def name(self) -> str:
        return "next_token"

    def evaluate(
        self,
        model,
        tokenizer,
        examples,
    ) -> EvaluationResult:

        model.eval()

        correct = 0
        total = 0
        losses = []

        with torch.no_grad():

            for example in examples:

                ids = tokenizer.encode(
                    example.prompt
                )

                if len(ids) < 2:
                    continue

                input_ids = torch.tensor(
                    [ids[:-1]],
                    dtype=torch.long,
                )

                labels = torch.tensor(
                    [ids[1:]],
                    dtype=torch.long,
                )

                device = next(
                    model.parameters()
                ).device

                input_ids = input_ids.to(
                    device
                )

                labels = labels.to(
                    device
                )

                output = model(
                    input_ids=input_ids,
                    labels=labels,
                )

                loss = output["loss"]

                losses.append(
                    float(
                        loss.detach().cpu()
                    )
                )

                logits = output.get(
                    "logits"
                )

                if logits is None:
                    continue

                predictions = (
                    logits.argmax(
                        dim=-1
                    )
                )

                correct += int(
                    (
                        predictions
                        == labels
                    ).sum().item()
                )

                total += labels.numel()

        score = (
            correct / total
            if total
            else 0.0
        )

        average_loss = (
            sum(losses) / len(losses)
            if losses
            else None
        )

        return EvaluationResult(
            name=self.name,
            score=score,
            passed=(
                score > 0
                or bool(losses)
            ),
            examples=len(examples),
            metrics={
                "accuracy":
                    score,
                "loss":
                    average_loss,
                "tokens":
                    total,
            },
        )

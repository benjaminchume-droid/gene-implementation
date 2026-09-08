from __future__ import annotations

import json
from pathlib import Path

import torch

from gene.model.neural import (
    GeneTransformer,
    TransformerConfig,
)
from gene.training.tokenizer import (
    GeneTokenizer,
)
from gene.training.engine.batching import (
    TokenBatcher,
)


def build_overfit_corpus(
    path: Path,
) -> None:

    records = [
        {
            "text":
                "Gene learns from examples."
        },
        {
            "text":
                "Gene verifies successful procedures."
        },
        {
            "text":
                "Gene can compose capabilities."
        },
        {
            "text":
                "Gene uses evidence before learning."
        },
    ]

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "\n".join(
            json.dumps(record)
            for record in records
        ),
        encoding="utf-8",
    )


def build_tiny_model(
    vocab_size: int,
) -> GeneTransformer:

    config = TransformerConfig(
        vocab_size=vocab_size,
        hidden_size=96,
        intermediate_size=384,
        num_layers=2,
        num_heads=4,
        max_position_embeddings=128,
        dropout=0.0,
        model_name="gene-overfit-test",
    )

    config.validate()

    return GeneTransformer(
        config
    )


def train_overfit(
    corpus: Path,
    tokenizer: GeneTokenizer,
    *,
    steps: int = 300,
) -> dict:

    model = build_tiny_model(
        tokenizer.vocab_size_actual()
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)

    batcher = TokenBatcher(
        tokenizer=tokenizer,
        max_sequence_length=128,
    )

    texts = list(
        tokenizer._iter_jsonl_text(
            [str(corpus)]
        )
    )

    batches = list(
        batcher.batch(
            texts,
            batch_size=4,
        )
    )

    if not batches:
        raise RuntimeError(
            "Overfit corpus produced no batches."
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.0,
    )

    model.train()

    first_loss = None
    last_loss = None

    for step in range(steps):

        batch = batches[
            step % len(batches)
        ]

        input_ids = batch[
            "input_ids"
        ].to(device)

        attention_mask = batch[
            "attention_mask"
        ].to(device)

        labels = input_ids.clone()

        labels[
            attention_mask == 0
        ] = -100

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = output["loss"]

        if first_loss is None:
            first_loss = float(
                loss.detach().cpu()
            )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

        last_loss = float(
            loss.detach().cpu()
        )

    return {
        "device": device,
        "parameters":
            model.parameter_count(),
        "first_loss":
            first_loss,
        "last_loss":
            last_loss,
        "loss_reduced":
            last_loss < first_loss,
        "model":
            model,
        "batches":
            batches,
    }


def run_test(
    output_dir: str = (
        "gene/data/training/overfit"
    ),
) -> dict:

    root = Path(output_dir)

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    corpus = (
        root
        / "overfit.jsonl"
    )

    tokenizer_path = (
        root
        / "overfit-tokenizer.json"
    )

    build_overfit_corpus(
        corpus
    )

    tokenizer = GeneTokenizer(
        vocab_size=256,
        min_frequency=1,
    )

    tokenizer.train_jsonl(
        [str(corpus)]
    )

    tokenizer.save(
        str(tokenizer_path)
    )

    result = train_overfit(
        corpus,
        tokenizer,
    )

    assert result[
        "loss_reduced"
    ], (
        "Tiny model failed to reduce "
        "training loss."
    )

    return {
        key: value
        for key, value
        in result.items()
        if key != "model"
        and key != "batches"
    }


if __name__ == "__main__":

    result = run_test()

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

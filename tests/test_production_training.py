import json

from pathlib import Path

from gene.model.neural import (
    GeneTransformer,
    TransformerConfig,
)

from gene.training.production import (
    ProductionTrainer,
)

from gene.training.tokenizer import (
    GeneTokenizer,
)

from gene.training.engine import (
    TrainingConfig,
)


def test_production_manifest(
    tmp_path,
):

    corpus = (
        tmp_path
        / "train.jsonl"
    )

    corpus.write_text(
        '{"text":"Gene learns examples."}\n'
        '{"text":"Gene verifies procedures."}\n',
        encoding="utf-8",
    )

    tokenizer_path = (
        tmp_path
        / "tokenizer.json"
    )

    tokenizer = GeneTokenizer(
        vocab_size=128,
        min_frequency=1,
    )

    tokenizer.train_jsonl(
        [str(corpus)]
    )

    tokenizer.save(
        str(tokenizer_path)
    )

    config = TransformerConfig(
        vocab_size=
            tokenizer.vocab_size_actual(),
        hidden_size=64,
        intermediate_size=256,
        num_layers=2,
        num_heads=4,
        max_position_embeddings=64,
        dropout=0.0,
        model_name="production-controller-test",
    )

    model = GeneTransformer(
        config
    )

    training_config = TrainingConfig(
        output_dir=str(
            tmp_path
            / "checkpoints"
        ),
        run_dir=str(
            tmp_path
            / "runs"
        ),
        epochs=1,
        batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        max_sequence_length=64,
        checkpoint_interval=1000,
    )

    production = ProductionTrainer(
        model=model,
        tokenizer=tokenizer,
        config=training_config,
        tokenizer_path=
            str(tokenizer_path),
        train_data_path=
            str(corpus),
        run_root=str(
            tmp_path
            / "manifests"
        ),
    )

    manifest = (
        production.create_manifest()
    )

    assert manifest.run_id
    assert manifest.model_name == (
        "production-controller-test"
    )

    assert (
        manifest.metadata[
            "tokenizer_sha256"
        ]
    )

    assert (
        manifest.metadata[
            "train_data_sha256"
        ]
    )

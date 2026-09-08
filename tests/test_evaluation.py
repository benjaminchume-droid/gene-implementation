import torch

from gene.evaluation import (
    BenchmarkRegistry,
    EvaluationExample,
    EvaluationRunner,
    GenerationBenchmark,
    NextTokenBenchmark,
)

from gene.model.neural import (
    GeneTransformer,
    TransformerConfig,
)

from gene.training.tokenizer import (
    GeneTokenizer,
)


def build_model_and_tokenizer(
    tmp_path,
):

    source = (
        tmp_path
        / "eval.jsonl"
    )

    source.write_text(
        '{"text":"Gene learns examples."}\n'
        '{"text":"Gene verifies results."}\n',
        encoding="utf-8",
    )

    tokenizer = GeneTokenizer(
        vocab_size=128,
        min_frequency=1,
    )

    tokenizer.train_jsonl(
        [str(source)]
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
        model_name="evaluation-test",
    )

    model = GeneTransformer(
        config
    )

    return model, tokenizer


def test_benchmark_registry():

    registry = BenchmarkRegistry()

    registry.register(
        NextTokenBenchmark()
    )

    registry.register(
        GenerationBenchmark()
    )

    assert registry.status()[
        "count"
    ] == 2


def test_evaluation_runner(
    tmp_path,
):

    model, tokenizer = (
        build_model_and_tokenizer(
            tmp_path
        )
    )

    registry = BenchmarkRegistry()

    registry.register(
        NextTokenBenchmark()
    )

    runner = EvaluationRunner(
        registry
    )

    examples = [
        EvaluationExample(
            prompt=
                "Gene learns examples.",
        ),
    ]

    report = runner.run(
        model,
        tokenizer,
        examples,
    )

    assert report.model_name == (
        "evaluation-test"
    )

    assert report.results

    assert (
        report.results[0].examples
        == 1
    )


def test_generation_benchmark(
    tmp_path,
):

    model, tokenizer = (
        build_model_and_tokenizer(
            tmp_path
        )
    )

    benchmark = GenerationBenchmark()

    result = benchmark.evaluate(
        model,
        tokenizer,
        [
            EvaluationExample(
                prompt=
                    "Gene learns",
            )
        ],
    )

    assert result.examples == 1
    assert result.metrics


def test_evaluation_comparison():

    from gene.evaluation.comparator import (
        EvaluationComparator,
    )

    result = (
        EvaluationComparator().compare(
            {
                "aggregate_score":
                    0.4
            },
            {
                "aggregate_score":
                    0.7
            },
        )
    )

    assert result["improved"] is True
    assert result["delta"] == 0.3

from __future__ import annotations

import importlib
import json
import sys

from pathlib import Path

import torch


class NeuralReadinessAuditor:

    def __init__(self) -> None:

        self.checks: list[dict] = []

    def record(
        self,
        name: str,
        passed: bool,
        detail: str = "",
        *,
        critical: bool = True,
        metadata: dict | None = None,
    ) -> None:

        self.checks.append(
            {
                "name": name,
                "passed": passed,
                "critical": critical,
                "detail": detail,
                "metadata": metadata or {},
            }
        )

    def check_python_environment(self) -> None:

        self.record(
            "python_environment",
            sys.version_info >= (3, 13),
            f"Python {sys.version}",
            metadata={
                "executable":
                    sys.executable,
            },
        )

    def check_imports(self) -> None:

        modules = [
            "torch",
            "tokenizers",
            "gene.model.neural",
            "gene.model.configs",
            "gene.training.tokenizer",
            "gene.training.packing",
            "gene.training.factory",
            "gene.training.engine",
            "gene.training.production",
            "gene.training.profiling",
            "gene.evaluation",
        ]

        for module in modules:

            try:
                importlib.import_module(
                    module
                )

                self.record(
                    f"import:{module}",
                    True,
                    "import successful",
                )

            except Exception as exc:

                self.record(
                    f"import:{module}",
                    False,
                    (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                )

    def check_model_configuration(self) -> None:

        try:

            from gene.model.configs import (
                gene_200m_config,
            )

            config = gene_200m_config()

            config.validate()

            self.record(
                "model_configuration",
                True,
                "200M target configuration validated",
                metadata={
                    "config":
                        config.to_dict(),
                },
            )

        except Exception as exc:

            self.record(
                "model_configuration",
                False,
                (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

    def check_model_instantiation(self) -> dict | None:

        try:

            from gene.model.configs import (
                build_gene_200m,
            )

            model = build_gene_200m()

            parameters = (
                model.parameter_count()
            )

            trainable = (
                model.trainable_parameter_count()
            )

            passed = (
                parameters > 0
                and trainable > 0
            )

            self.record(
                "model_instantiation",
                passed,
                (
                    f"parameters={parameters}, "
                    f"trainable={trainable}"
                ),
                metadata={
                    "parameter_count":
                        parameters,
                    "trainable_parameter_count":
                        trainable,
                    "model_name":
                        model.config.model_name,
                },
            )

            return {
                "model":
                    model,
                "parameters":
                    parameters,
                "trainable":
                    trainable,
            }

        except Exception as exc:

            self.record(
                "model_instantiation",
                False,
                (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

            return None

    def check_tokenizer(
        self,
    ) -> object | None:

        candidates = [
            Path(
                "gene/data/training/tokenizer/"
                "gene-tokenizer.json"
            ),
            Path(
                "gene/data/training/tokenizer/"
                "gene-tokenizer.json"
            ),
        ]

        tokenizer_path = next(
            (
                path
                for path in candidates
                if path.exists()
            ),
            None,
        )

        if tokenizer_path is None:

            self.record(
                "production_tokenizer",
                False,
                (
                    "Production tokenizer file "
                    "was not found."
                ),
            )

            return None

        try:

            from gene.training.tokenizer import (
                GeneTokenizer,
            )

            tokenizer = (
                GeneTokenizer.load(
                    str(tokenizer_path)
                )
            )

            vocab_size = (
                tokenizer.vocab_size_actual()
            )

            passed = (
                vocab_size > 0
            )

            self.record(
                "production_tokenizer",
                passed,
                (
                    f"vocab_size={vocab_size}"
                ),
                metadata={
                    "path":
                        str(tokenizer_path),
                    "vocab_size":
                        vocab_size,
                },
            )

            return tokenizer

        except Exception as exc:

            self.record(
                "production_tokenizer",
                False,
                (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

            return None

    def check_vocab_compatibility(
        self,
        model_info,
        tokenizer,
    ) -> None:

        if (
            model_info is None
            or tokenizer is None
        ):
            self.record(
                "vocab_compatibility",
                False,
                "Model or tokenizer unavailable.",
            )
            return

        model_vocab = (
            model_info["model"]
            .config
            .vocab_size
        )

        tokenizer_vocab = (
            tokenizer
            .vocab_size_actual()
        )

        passed = (
            model_vocab
            == tokenizer_vocab
        )

        self.record(
            "vocab_compatibility",
            passed,
            (
                f"model_vocab={model_vocab}, "
                f"tokenizer_vocab={tokenizer_vocab}"
            ),
        )

    def check_training_stack(
        self,
    ) -> None:

        symbols = [
            (
                "gene.training.engine",
                "GeneTrainer",
            ),
            (
                "gene.training.factory",
                "TrainingDataFactory",
            ),
            (
                "gene.training.packing",
                "CorpusPacker",
            ),
            (
                "gene.training.production",
                "ProductionTrainer",
            ),
            (
                "gene.training.profiling",
                "HardwareProfiler",
            ),
            (
                "gene.evaluation",
                "EvaluationRunner",
            ),
        ]

        for module_name, symbol in symbols:

            try:

                module = importlib.import_module(
                    module_name
                )

                passed = hasattr(
                    module,
                    symbol,
                )

                self.record(
                    f"training_stack:{symbol}",
                    passed,
                    (
                        "available"
                        if passed
                        else "symbol missing"
                    ),
                )

            except Exception as exc:

                self.record(
                    f"training_stack:{symbol}",
                    False,
                    (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                )

    def check_training_data(
        self,
    ) -> None:

        tokenizer_path = Path(
            "gene/data/training/tokenizer/"
            "gene-tokenizer.json"
        )

        packed_candidates = [
            Path(
                "gene/data/training/packed/"
                "train.bin"
            ),
            Path(
                "gene/data/training/packed/"
                "train.jsonl"
            ),
        ]

        self.record(
            "tokenizer_artifact",
            tokenizer_path.exists(),
            (
                str(tokenizer_path)
                if tokenizer_path.exists()
                else "missing"
            ),
        )

        packed = next(
            (
                path
                for path
                in packed_candidates
                if path.exists()
            ),
            None,
        )

        if packed is None:

            self.record(
                "packed_training_data",
                False,
                "No packed training artifact found.",
            )

            return

        size = packed.stat().st_size

        self.record(
            "packed_training_data",
            size > 0,
            (
                f"path={packed}, "
                f"bytes={size}"
            ),
        )

    def check_smoke_artifact(
        self,
    ) -> None:

        path = Path(
            "gene/data/training/"
            "200m_smoke_test.json"
        )

        if not path.exists():

            self.record(
                "200m_smoke_test",
                False,
                "Smoke-test report not found.",
            )

            return

        try:

            report = json.loads(
                path.read_text(
                    encoding="utf-8-sig"
                )
            )

            passed = (
                report.get(
                    "forward"
                ) is True
                and
                report.get(
                    "backward"
                ) is True
                and
                report.get(
                    "optimizer_step"
                ) is True
                and
                report.get(
                    "changed_parameters",
                    0,
                ) > 0
            )

            self.record(
                "200m_smoke_test",
                passed,
                (
                    f"loss={report.get('loss')}, "
                    f"changed_parameters="
                    f"{report.get('changed_parameters')}"
                ),
                metadata=report,
            )

        except Exception as exc:

            self.record(
                "200m_smoke_test",
                False,
                (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

    def check_hardware(
        self,
    ) -> None:

        try:

            from gene.training.profiling import (
                hardware_report,
            )

            report = hardware_report()

            self.record(
                "hardware_profile",
                True,
                (
                    f"device="
                    f"{report.get('gpu_name') or 'CPU'}"
                ),
                metadata=report,
            )

        except Exception as exc:

            self.record(
                "hardware_profile",
                False,
                (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

    def check_production_controller(
        self,
    ) -> None:

        try:

            from gene.training.production import (
                ProductionTrainer,
            )

            self.record(
                "production_training_controller",
                True,
                (
                    f"{ProductionTrainer.__name__} "
                    "available"
                ),
            )

        except Exception as exc:

            self.record(
                "production_training_controller",
                False,
                (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

    def run(self) -> dict:

        self.check_python_environment()
        self.check_imports()
        self.check_model_configuration()

        model_info = (
            self.check_model_instantiation()
        )

        tokenizer = (
            self.check_tokenizer()
        )

        self.check_vocab_compatibility(
            model_info,
            tokenizer,
        )

        self.check_training_stack()
        self.check_training_data()
        self.check_smoke_artifact()
        self.check_hardware()
        self.check_production_controller()

        critical = [
            item
            for item
            in self.checks
            if item["critical"]
        ]

        failed_critical = [
            item
            for item
            in critical
            if not item["passed"]
        ]

        return {
            "ready":
                not failed_critical,

            "checks":
                self.checks,

            "passed":
                sum(
                    1
                    for item
                    in self.checks
                    if item["passed"]
                ),

            "failed":
                sum(
                    1
                    for item
                    in self.checks
                    if not item["passed"]
                ),

            "critical_failures":
                failed_critical,

            "python":
                sys.version,

            "torch":
                torch.__version__,
        }


def main() -> None:

    auditor = (
        NeuralReadinessAuditor()
    )

    report = auditor.run()

    print(
        "=" * 72
    )

    print(
        "GENE 200M NEURAL READINESS AUDIT"
    )

    print(
        "=" * 72
    )

    for check in report["checks"]:

        status = (
            "PASS"
            if check["passed"]
            else "FAIL"
        )

        print(
            f"[{status}] "
            f"{check['name']}: "
            f"{check['detail']}"
        )

    print(
        "-" * 72
    )

    print(
        f"Passed: {report['passed']}"
    )

    print(
        f"Failed: {report['failed']}"
    )

    print(
        "READY FOR 200M PILOT:",
        "YES"
        if report["ready"]
        else "NO",
    )

    output = Path(
        "gene/data/training/"
        "neural_readiness.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        "Report:",
        output,
    )

    if not report["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

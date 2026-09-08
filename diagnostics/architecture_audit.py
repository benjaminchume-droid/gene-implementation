from __future__ import annotations

import ast
import importlib
import json
import pkgutil
import sys

from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("gene")


class ArchitectureAuditor:

    REQUIRED_PACKAGES = [
        "gene.agent",
        "gene.browser",
        "gene.capabilities",
        "gene.discovery",
        "gene.evolution",
        "gene.experimentation",
        "gene.learning",
        "gene.planning",
        "gene.model",
        "gene.training",
        "gene.evaluation",
        "gene.mcp",
        "gene.desktop",
        "gene.voice",
        "gene.tools",
        "gene.memory",
        "gene.knowledge",
    ]

    REQUIRED_SYMBOLS = {
        "gene.agent": [
            "GeneAgent",
        ],
        "gene.discovery": [
            "DiscoveryEngine",
            "DiscoveryRegistry",
        ],
        "gene.experimentation": [
            "ExperimentRunner",
            "VerificationEngine",
        ],
        "gene.planning": [
            "CapabilityComposer",
            "PlanExecutor",
        ],
        "gene.learning": [
            "LearningOrchestrator",
        ],
        "gene.capabilities.runtime": [
            "CapabilityRuntime",
        ],
        "gene.training.factory": [
            "TrainingDataFactory",
        ],
        "gene.training.production": [
            "ProductionTrainer",
        ],
        "gene.evaluation": [
            "EvaluationRunner",
        ],
        "gene.mcp": [
            "MCPRuntime",
        ],
    }

    def __init__(self) -> None:

        self.errors: list[dict] = []
        self.warnings: list[dict] = []

        self.files: list[Path] = []

        self.imports: dict[
            str,
            set[str],
        ] = defaultdict(set)

        self.symbols: dict[
            str,
            set[str],
        ] = defaultdict(set)

    def error(
        self,
        category: str,
        message: str,
        **metadata,
    ) -> None:

        self.errors.append(
            {
                "category":
                    category,
                "message":
                    message,
                **metadata,
            }
        )

    def warning(
        self,
        category: str,
        message: str,
        **metadata,
    ) -> None:

        self.warnings.append(
            {
                "category":
                    category,
                "message":
                    message,
                **metadata,
            }
        )

    def collect_files(
        self,
    ) -> None:

        if not ROOT.exists():

            self.error(
                "filesystem",
                "gene package does not exist.",
            )

            return

        self.files = sorted(
            ROOT.rglob("*.py")
        )

    def check_package_structure(
        self,
    ) -> None:

        for path in self.files:

            if path.name != "__init__.py":
                continue

            # Package is fine.

        # Only directories that actually contain Python
        # source files are candidates for Python packages.
        # Data, cache, model, checkpoint, and other project
        # directories do not require __init__.py.
        package_dirs = {
            path.parent
            for path in self.files
            if path.name != "__init__.py"
        }

        for directory in sorted(
            package_dirs
        ):

            python_files = list(
                directory.glob("*.py")
            )

            if not python_files:
                continue

            if not (
                directory / "__init__.py"
            ).exists():

                self.warning(
                    "package",
                    "Python source directory lacks __init__.py.",
                    path=str(directory),
                )

    def parse_python_files(
        self,
    ) -> None:

        for path in self.files:

            try:

                source = path.read_text(
                    encoding="utf-8-sig"
                )

                tree = ast.parse(
                    source,
                    filename=str(path),
                )

            except Exception as exc:

                self.error(
                    "syntax",
                    str(exc),
                    path=str(path),
                )

                continue

            module = (
                "gene."
                + ".".join(
                    path.relative_to(
                        ROOT.parent
                    ).with_suffix("").parts
                )
            )

            if module.endswith(
                ".__init__"
            ):
                module = module[
                    :-len(".__init__")
                ]

            for node in ast.walk(tree):

                if isinstance(
                    node,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                        ast.ClassDef,
                    ),
                ):

                    self.symbols[
                        module
                    ].add(
                        node.name
                    )

                elif isinstance(
                    node,
                    ast.Import,
                ):

                    for alias in node.names:

                        self.imports[
                            module
                        ].add(
                            alias.name
                        )

                elif isinstance(
                    node,
                    ast.ImportFrom,
                ):

                    if node.module:

                        self.imports[
                            module
                        ].add(
                            node.module
                        )

    def check_stale_imports(
        self,
    ) -> None:

        existing_modules = set()

        for path in self.files:

            relative = path.relative_to(
                ROOT.parent
            ).with_suffix("")

            parts = list(
                relative.parts
            )

            if parts[-1] == "__init__":
                parts.pop()

            module = ".".join(parts)

            existing_modules.add(
                module
            )

        for module, imports in (
            self.imports.items()
        ):

            for imported in imports:

                if not imported.startswith(
                    "gene."
                ):
                    continue

                # Only inspect obvious direct Gene paths.
                base = imported

                if base in existing_modules:
                    continue

                # Check whether the import points to a
                # submodule that exists but isn't directly
                # represented by the parsed path.
                possible = [
                    candidate
                    for candidate
                    in existing_modules
                    if candidate == base
                    or candidate.startswith(
                        base + "."
                    )
                ]

                if not possible:

                    self.warning(
                        "stale_import",
                        (
                            f"{module} imports "
                            f"missing module {imported}"
                        ),
                        module=module,
                        imported=imported,
                    )

    def check_required_symbols(
        self,
    ) -> None:

        for module, required in (
            self.REQUIRED_SYMBOLS.items()
        ):

            try:

                loaded = importlib.import_module(
                    module
                )

            except Exception as exc:

                self.error(
                    "import",
                    (
                        f"Cannot import {module}: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    module=module,
                )

                continue

            for symbol in required:

                if not hasattr(
                    loaded,
                    symbol,
                ):

                    self.error(
                        "export",
                        (
                            f"{module} does not "
                            f"export {symbol}"
                        ),
                        module=module,
                        symbol=symbol,
                    )

    def check_duplicate_class_names(
        self,
    ) -> None:

        locations = defaultdict(list)

        for module, symbols in (
            self.symbols.items()
        ):

            for symbol in symbols:

                locations[symbol].append(
                    module
                )

        for symbol, modules in (
            locations.items()
        ):

            if (
                len(modules) <= 1
            ):
                continue

            # Duplicate names are not automatically bad.
            # Report them for human review.
            if symbol.endswith(
                (
                    "Engine",
                    "Runtime",
                    "Store",
                    "Registry",
                    "Manager",
                    "Trainer",
                )
            ):

                self.warning(
                    "duplicate_symbol",
                    (
                        f"{symbol} exists in "
                        f"{len(modules)} modules."
                    ),
                    symbol=symbol,
                    modules=modules,
                )

    def check_known_conflicts(
        self,
    ) -> None:

        conflicts = [
            (
                "gene.model.backends.neural",
                "gene.model.backends.neural",
            ),
        ]

        for left, right in conflicts:

            if left == right:

                path = (
                    ROOT
                    / "model"
                    / "backends"
                    / "neural.py"
                )

                sibling = (
                    ROOT
                    / "model"
                    / "backends"
                    / "neural"
                )

                if (
                    path.exists()
                    and sibling.exists()
                    and sibling.is_dir()
                ):

                    self.error(
                        "duplicate_module_path",
                        (
                            "neural.py and neural/ "
                            "both exist."
                        ),
                        file=str(path),
                        directory=str(sibling),
                    )

    def check_public_imports(
        self,
    ) -> None:

        for package in self.REQUIRED_PACKAGES:

            try:

                importlib.import_module(
                    package
                )

            except Exception as exc:

                self.error(
                    "package_import",
                    (
                        f"{package}: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    package=package,
                )

    def check_compilation(
        self,
    ) -> None:

        import py_compile

        for path in self.files:

            try:

                py_compile.compile(
                    str(path),
                    doraise=True,
                )

            except Exception as exc:

                self.error(
                    "compile",
                    str(exc),
                    path=str(path),
                )

    def check_training_chain(
        self,
    ) -> None:

        chain = [
            (
                "gene.training.tokenizer",
                "GeneTokenizer",
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
                "gene.training.engine",
                "GeneTrainer",
            ),
            (
                "gene.training.production",
                "ProductionTrainer",
            ),
        ]

        for module, symbol in chain:

            try:

                loaded = importlib.import_module(
                    module
                )

                if not hasattr(
                    loaded,
                    symbol,
                ):

                    self.error(
                        "training_chain",
                        (
                            f"{module} does not "
                            f"export {symbol}"
                        ),
                    )

            except Exception as exc:

                self.error(
                    "training_chain",
                    (
                        f"{module}: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )

    def check_model_chain(
        self,
    ) -> None:

        try:

            from gene.model.neural import (
                GeneTransformer,
                TransformerConfig,
            )

            config = TransformerConfig(
                vocab_size=128,
                hidden_size=32,
                intermediate_size=128,
                num_layers=1,
                num_heads=4,
                max_position_embeddings=64,
                dropout=0.0,
                model_name="architecture-audit",
            )

            model = GeneTransformer(
                config
            )

            parameter_count = (
                model.parameter_count()
            )

            if parameter_count <= 0:

                self.error(
                    "model",
                    "Neural model has no parameters.",
                )

        except Exception as exc:

            self.error(
                "model",
                (
                    f"Neural chain failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )

    def run(
        self,
    ) -> dict:

        self.collect_files()
        self.check_package_structure()
        self.parse_python_files()
        self.check_compilation()
        self.check_stale_imports()
        self.check_public_imports()
        self.check_required_symbols()
        self.check_duplicate_class_names()
        self.check_known_conflicts()
        self.check_training_chain()
        self.check_model_chain()

        report = {
            "success":
                not self.errors,

            "python_files":
                len(self.files),

            "errors":
                self.errors,

            "warnings":
                self.warnings,
        }

        return report


def main():

    auditor = ArchitectureAuditor()

    report = auditor.run()

    print(
        "=" * 72
    )

    print(
        "GENE ARCHITECTURE-WIDE INTEGRATION AUDIT"
    )

    print(
        "=" * 72
    )

    print(
        f"Python files: {report['python_files']}"
    )

    print(
        f"Errors:       {len(report['errors'])}"
    )

    print(
        f"Warnings:     {len(report['warnings'])}"
    )

    if report["errors"]:

        print(
            "\nERRORS"
        )

        for error in report["errors"]:

            print(
                f"  [{error['category']}] "
                f"{error['message']}"
            )

    if report["warnings"]:

        print(
            "\nWARNINGS"
        )

        for warning in report["warnings"]:

            print(
                f"  [{warning['category']}] "
                f"{warning['message']}"
            )

    output = Path(
        "gene/data/"
        "architecture_audit.json"
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
        "\nReport:",
        output,
    )

    print(
        "\nSTATUS:",
        "PASS" if report["success"]
        else "FAILED",
    )

    if not report["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import uuid

from pathlib import Path

from gene.training.engine import (
    GeneTrainer,
    TrainingConfig,
)
from gene.training.profiling import (
    hardware_report,
)

from .manifest import (
    RunManifest,
    RunManifestStore,
    sha256_file,
)


class ProductionTrainer:

    def __init__(
        self,
        model,
        tokenizer,
        config: TrainingConfig,
        *,
        tokenizer_path: str,
        train_data_path: str,
        validation_data_path: str | None = None,
        run_root: str = (
            "gene/data/training/runs"
        ),
    ) -> None:

        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        self.tokenizer_path = (
            tokenizer_path
        )

        self.train_data_path = (
            train_data_path
        )

        self.validation_data_path = (
            validation_data_path
        )

        self.run_id = (
            "run-"
            + uuid.uuid4().hex
        )

        self.manifests = (
            RunManifestStore(
                run_root
            )
        )

    def validate_inputs(self) -> None:

        files = [
            self.tokenizer_path,
            self.train_data_path,
        ]

        if self.validation_data_path:
            files.append(
                self.validation_data_path
            )

        for path in files:

            if not Path(path).exists():
                raise FileNotFoundError(
                    f"Required training input "
                    f"does not exist: {path}"
                )

        self.config.validate()

        if (
            self.model.config
            .vocab_size
            != self.tokenizer.vocab_size_actual()
        ):
            raise ValueError(
                "Model vocabulary size does not "
                "match tokenizer vocabulary size."
            )

    def create_manifest(self) -> RunManifest:

        self.validate_inputs()

        manifest = RunManifest(
            run_id=self.run_id,

            model_name=
                self.model.config.model_name,

            model_config=
                self.model.config.to_dict(),

            training_config=
                self.config.to_dict(),

            tokenizer_path=
                self.tokenizer_path,

            train_data_path=
                self.train_data_path,

            validation_data_path=
                self.validation_data_path,

            hardware=
                hardware_report(),

            metadata={
                "tokenizer_sha256":
                    sha256_file(
                        self.tokenizer_path
                    ),

                "train_data_sha256":
                    sha256_file(
                        self.train_data_path
                    ),

                "validation_data_sha256":
                    (
                        sha256_file(
                            self.validation_data_path
                        )
                        if self.validation_data_path
                        else None
                    ),
            },
        )

        self.manifests.save(
            manifest
        )

        return manifest

    def run(
        self,
    ) -> dict:

        manifest = (
            self.create_manifest()
        )

        trainer = GeneTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config,
        )

        result = trainer.train(
            train_file=
                self.train_data_path,
            validation_file=
                self.validation_data_path,
        )

        result["run_id"] = (
            manifest.run_id
        )

        result["manifest"] = (
            self.manifests.load(
                manifest.run_id
            )
        )

        output = Path(
            self.config.run_dir
        )

        output.mkdir(
            parents=True,
            exist_ok=True,
        )

        (output / f"{self.run_id}-result.json").write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )

        return result

    def resume(
        self,
        checkpoint_path: str,
    ) -> dict:

        self.validate_inputs()

        trainer = GeneTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config,
        )

        checkpoint = trainer.resume(
            checkpoint_path
        )

        return {
            "success": True,
            "run_id": self.run_id,
            "checkpoint":
                checkpoint,
        }

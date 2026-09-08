from __future__ import annotations

import json
from pathlib import Path

import torch

from gene.model.configs import (
    gene_200m_config,
)
from gene.model.neural import (
    GeneTransformer,
)


def run_smoke_test() -> dict:

    config = gene_200m_config()
    config.validate()

    model = GeneTransformer(
        config
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)
    model.train()

    batch_size = 1

    sequence_length = min(
        128,
        config.max_position_embeddings,
    )

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(
            batch_size,
            sequence_length,
        ),
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.ones(
        (
            batch_size,
            sequence_length,
        ),
        dtype=torch.long,
        device=device,
    )

    labels = input_ids.clone()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        weight_decay=0.0,
    )

    before = {
        name:
            parameter.detach().clone()
        for name, parameter
        in model.named_parameters()
        if parameter.requires_grad
    }

    optimizer.zero_grad(
        set_to_none=True
    )

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    loss = output.get(
        "loss"
    )

    if loss is None:
        raise RuntimeError(
            "GeneTransformer did not return a loss."
        )

    if not torch.isfinite(loss):
        raise RuntimeError(
            f"Non-finite loss: {loss}"
        )

    loss.backward()

    gradient_count = 0
    gradient_norm = 0.0

    for parameter in model.parameters():

        if parameter.grad is None:
            continue

        gradient_count += 1

        gradient_norm += float(
            parameter.grad.detach()
            .float()
            .norm()
            .cpu()
        )

    if gradient_count == 0:
        raise RuntimeError(
            "No trainable gradients were produced."
        )

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0,
    )

    optimizer.step()

    changed_parameters = 0

    for name, parameter in (
        model.named_parameters()
    ):

        if not parameter.requires_grad:
            continue

        previous = before[name]

        if not torch.equal(
            previous,
            parameter.detach(),
        ):
            changed_parameters += 1

    if changed_parameters == 0:
        raise RuntimeError(
            "Optimizer step did not change "
            "any trainable parameter."
        )

    result = {
        "model":
            config.model_name,

        "device":
            device,

        "parameter_count":
            model.parameter_count(),

        "trainable_parameter_count":
            model.trainable_parameter_count(),

        "context":
            config.max_position_embeddings,

        "sequence_length":
            sequence_length,

        "loss":
            float(
                loss.detach().cpu()
            ),

        "gradient_tensors":
            gradient_count,

        "gradient_norm":
            gradient_norm,

        "changed_parameters":
            changed_parameters,

        "forward":
            True,

        "backward":
            True,

        "optimizer_step":
            True,
    }

    output_path = Path(
        "gene/data/training/"
        "200m_smoke_test.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    return result


if __name__ == "__main__":

    print(
        json.dumps(
            run_smoke_test(),
            indent=2,
        )
    )

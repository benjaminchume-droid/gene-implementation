import torch

from gene.multimodal.projectors import (
    VisionProjector,
)


def test_vision_projector_shape():

    projector = VisionProjector()

    embedding = torch.randn(
        1,
        768,
    )

    result = projector(
        embedding
    )

    assert tuple(
        result.shape
    ) == (
        1,
        864,
    )


def test_vision_projector_batch():

    projector = VisionProjector()

    embedding = torch.randn(
        4,
        768,
    )

    result = projector(
        embedding
    )

    assert tuple(
        result.shape
    ) == (
        4,
        864,
    )


def test_gradient_flow():

    projector = VisionProjector()

    embedding = torch.randn(
        2,
        768,
        requires_grad=False,
    )

    output = projector(
        embedding
    )

    loss = output.pow(2).mean()

    loss.backward()

    gradients = [
        parameter.grad
        for parameter
        in projector.parameters()
        if parameter.requires_grad
    ]

    assert gradients
    assert any(
        gradient is not None
        for gradient in gradients
    )

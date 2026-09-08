from gene.model import (
    ModelBackend,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ModelRegistry,
    ModelRuntime,
)
from gene.model.backends import (
    NullModelBackend,
)


class TestBackend(ModelBackend):

    @property
    def capabilities(
        self,
    ) -> ModelCapabilities:

        return ModelCapabilities(
            name="test-model",
            backend="test",
            parameter_count=123,
            native_context=4096,
            modalities=("text",),
            supports_generation=True,
        )

    def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:

        return ModelResponse(
            success=True,
            text=request.prompt or "",
            model="test-model",
            backend="test",
        )


def test_model_contract():

    backend = TestBackend()

    assert backend.capabilities.name == (
        "test-model"
    )

    result = backend.generate(
        ModelRequest(
            prompt="hello"
        )
    )

    assert result.success is True
    assert result.text == "hello"


def test_model_registry(tmp_path):

    registry = ModelRegistry(
        str(
            tmp_path / "models.json"
        )
    )

    backend = TestBackend()

    registry.register(
        backend
    )

    assert (
        registry.capabilities(
            "test-model"
        ).parameter_count
        == 123
    )

    assert len(
        registry.list()
    ) == 1


def test_model_runtime(tmp_path):

    registry = ModelRegistry(
        str(
            tmp_path / "models.json"
        )
    )

    runtime = ModelRuntime(
        registry
    )

    runtime.register(
        TestBackend()
    )

    selected = runtime.select(
        "test-model"
    )

    assert selected["success"] is True

    response = runtime.generate(
        ModelRequest(
            prompt="Gene"
        )
    )

    assert response.success is True
    assert response.text == "Gene"

    assert runtime.health()[
        "healthy"
    ] is True


def test_null_backend():

    runtime = ModelRuntime(
        ModelRegistry()
    )

    runtime.register(
        NullModelBackend()
    )

    runtime.select("null")

    result = runtime.generate(
        ModelRequest(
            prompt="test"
        )
    )

    assert result.success is True
    assert result.model == "null"

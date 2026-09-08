from gene.core.capabilities.bootstrap import create_capability_system


def test_capability_system_boots() -> None:
    router = create_capability_system()

    names = router.registry.list()

    assert names
    assert router.registry.has("filesystem.read")
    assert router.registry.has("filesystem.list")
    assert router.registry.has("filesystem.search")

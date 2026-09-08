from gene.core.runtime import GeneRuntime


def main() -> None:

    runtime = GeneRuntime()

    print("=" * 70)
    print("GENE RUNTIME")
    print("=" * 70)

    health = runtime.health()

    print("Status:", health["status"])
    print("Identity:", health["identity"]["name"])
    print("Soul:", ", ".join(
        health["identity"]
        and runtime.soul.active_sparks()
    ))
    print("Genome:", health["genome"]["name"])
    print(
        "Genome modules:",
        health["genome"]["modules"],
    )
    print(
        "Memory records:",
        health["memory"]["memory_count"],
    )
    print(
        "Knowledge records:",
        health["knowledge"]["records"],
    )
    print(
        "Skills:",
        health["skills"]["total"],
    )
    print(
        "Effective context target:",
        health["context"]["effective_context_target"],
    )
    print(
        "Tools:",
        health["tools"],
    )

    print("=" * 70)
    print("GENE RUNTIME READY")


if __name__ == "__main__":
    main()

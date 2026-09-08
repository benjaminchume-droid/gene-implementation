from .base import EvolutionInvariantError


class NeuralWeightGuard:

    """
    Runtime invariant:

    External evolution may change external parameters only.
    It may not mutate a neural model's internal parameters.
    """

    def __init__(self):
        self.weight_mutation_attempts = 0

    def check_model_mutation(
        self,
        operation: str,
    ) -> None:

        if operation in {
            "load_checkpoint",
            "select_backend",
            "inference",
        }:
            return

        self.weight_mutation_attempts += 1

        raise EvolutionInvariantError(
            "External evolution cannot modify "
            "neural weights during runtime."
        )

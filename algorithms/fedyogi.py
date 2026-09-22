from flwr.serverapp.strategy import FedYogi
from algorithms.base import common_strategy_config


def create(config: dict) -> FedYogi:
    return FedYogi(
        eta=float(config.get("fedopt-eta", 0.1)),
        tau=float(config.get("fedopt-tau", 0.001)),
        beta_1=float(config.get("beta-1", 0.9)),
        beta_2=float(config.get("beta-2", 0.99)),
        **common_strategy_config(config),
    )
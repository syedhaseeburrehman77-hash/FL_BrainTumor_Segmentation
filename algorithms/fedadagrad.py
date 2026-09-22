from flwr.serverapp.strategy import FedAdagrad
from algorithms.base import common_strategy_config


def create(config: dict) -> FedAdagrad:
    return FedAdagrad(
        eta=float(config.get("fedopt-eta", 0.1)),
        tau=float(config.get("fedopt-tau", 0.001)),
        **common_strategy_config(config),
    )
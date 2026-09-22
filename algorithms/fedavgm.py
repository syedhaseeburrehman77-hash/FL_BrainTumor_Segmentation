from flwr.serverapp.strategy import FedAvgM
from algorithms.base import common_strategy_config


def create(config: dict) -> FedAvgM:
    return FedAvgM(
        server_learning_rate=float(config.get("server-learning-rate", 1.0)),
        server_momentum=float(config.get("server-momentum", 0.9)),
        **common_strategy_config(config),
    )
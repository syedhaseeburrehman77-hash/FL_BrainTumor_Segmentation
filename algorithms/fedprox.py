from flwr.serverapp.strategy import FedProx
from algorithms.base import common_strategy_config


def create(config: dict) -> FedProx:
    return FedProx(
        proximal_mu=float(config.get("proximal-mu", 0.01)),
        **common_strategy_config(config),
    )
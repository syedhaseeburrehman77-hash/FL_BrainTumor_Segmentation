from flwr.serverapp.strategy import FedTrimmedAvg
from algorithms.base import common_strategy_config


def create(config: dict) -> FedTrimmedAvg:
    return FedTrimmedAvg(
        beta=float(config.get("trim-beta", 0.2)),
        **common_strategy_config(config),
    )
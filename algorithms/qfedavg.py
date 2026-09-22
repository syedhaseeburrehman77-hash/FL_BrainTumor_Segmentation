from flwr.serverapp.strategy import QFedAvg
from algorithms.base import common_strategy_config


def create(config: dict) -> QFedAvg:
    return QFedAvg(
        q=float(config.get("qfedavg-q", 0.1)),
        **common_strategy_config(config),
    )

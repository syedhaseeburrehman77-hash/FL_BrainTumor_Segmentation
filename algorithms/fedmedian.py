from flwr.serverapp.strategy import FedMedian
from algorithms.base import common_strategy_config


def create(config: dict) -> FedMedian:
    return FedMedian(**common_strategy_config(config))
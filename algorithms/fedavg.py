from flwr.serverapp.strategy import FedAvg
from algorithms.base import common_strategy_config


def create(config: dict) -> FedAvg:
    return FedAvg(**common_strategy_config(config))
from algorithms import (
    fedadagrad, fedadam, fedavg, fedavgm, fedmedian,
    fedprox, fedtrimmedavg, fedyogi, qfedavg,
)
from algorithms.regsimagg import RegSimAgg
from algorithms.base import common_strategy_config


BUILDERS = {
    "fedavg": fedavg.create,
    "fedprox": fedprox.create,
    "fedavgm": fedavgm.create,
    "fedadagrad": fedadagrad.create,
    "fedadam": fedadam.create,
    "fedyogi": fedyogi.create,
    "qfedavg": qfedavg.create,
    "fedmedian": fedmedian.create,
    "fedtrimmedavg": fedtrimmedavg.create,
}


def create_strategy(config: dict):
    name = str(config.get("strategy", "fedavg")).lower()

    if name == "regsimagg":
        return RegSimAgg(
            regularization_round=int(
                config.get("regsimagg-regularization-round", 5)
            ),
            **common_strategy_config(config),
        )

    if name not in BUILDERS:
        raise ValueError(f"Unsupported strategy: {name}")

    return BUILDERS[name](config)
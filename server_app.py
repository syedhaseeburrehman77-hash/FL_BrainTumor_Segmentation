from pathlib import Path

import pandas as pd
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAdam, FedAvg, FedProx

from algorithms.regsimagg import RegSimAgg
from models.unet import build_model

app = ServerApp()


def build_strategy(name: str, config):
    common = {
        "fraction_train": 1.0,
        "fraction_evaluate": 1.0,
    }
    if name == "fedprox":
        return FedProx(proximal_mu=0.01, **common)
    if name == "fedadam":
        return FedAdam(**common)
    if name == "regsimagg":
        return RegSimAgg(
            regularization_round=int(config["regsimagg-regularization-round"]),
            **common,
        )
    return FedAvg(**common)


@app.main()
def main(grid: Grid, context: Context) -> None:
    config = context.run_config
    strategy_name = str(config["strategy"]).lower()
    result = build_strategy(strategy_name, config).start(
        grid=grid,
        initial_arrays=ArrayRecord(build_model().state_dict()),
        train_config=ConfigRecord({"lr": float(config["learning-rate"])}),
        num_rounds=int(config["num-server-rounds"]),
    )

    output = Path(config["output-dir"])
    output.mkdir(exist_ok=True)
    torch.save(result.arrays.to_torch_state_dict(), output / f"{strategy_name}_final.pt")

    rows = []
    for round_id, metrics in result.evaluate_metrics_clientapp.items():
        rows.append({"round": round_id, **dict(metrics)})
    if rows:
        pd.DataFrame(rows).to_csv(output / f"{strategy_name}_metrics.csv", index=False)
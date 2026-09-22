from pathlib import Path

from datasets_loaders.base import DatasetConfig
from datasets_loaders.fets2022_dataset import FeTS2022Dataset

DATASET_REGISTRY = {
    "fets2022": FeTS2022Dataset,
}

def get_partition_loaders(run_config: dict, partition_id: int):
    config = DatasetConfig(
        data_root=Path(run_config["data-root"]),
        partition_csv=Path(run_config["partition-csv"]),
        batch_size=int(run_config["batch-size"]),
        num_workers=int(run_config.get("num-workers", 0)),
        seed=int(run_config.get("seed", 42)),
    )
    return DATASET_REGISTRY["fets2022"](config).load_partition(partition_id)
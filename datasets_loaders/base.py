from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DatasetConfig:
    data_root: Path
    partition_csv: Path
    batch_size: int
    num_workers: int = 0
    seed: int = 42

class BaseFederatedDataset(ABC):
    @abstractmethod
    def load_partition(self, partition_id: int):
        """Return one client's train and validation DataLoaders."""
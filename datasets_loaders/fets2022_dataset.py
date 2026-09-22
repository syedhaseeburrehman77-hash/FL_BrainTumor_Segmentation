from pathlib import Path

import numpy as np
import pandas as pd
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,CropForegroundd,EnsureChannelFirstd,EnsureTyped,LoadImaged,MapLabelValued,
    NormalizeIntensityd,Orientationd,RandCropByPosNegLabeld,
    RandFlipd,RandRotate90d,Spacingd,
)

from datasets_loaders.base import BaseFederatedDataset, DatasetConfig

PATCH_SIZE = (96, 96, 96)
MODALITIES = ("t1ce", "t1", "t2", "flair")


class FeTS2022Dataset(BaseFederatedDataset):
    """Federated FeTS2022 dataset partitioned by institution."""

    def __init__(self, config: DatasetConfig):
        self.config = config

    @staticmethod
    def _case(subject_dir: Path) -> dict:
        files = list(subject_dir.rglob("*.nii*"))

        def first_match(name: str) -> str:
            matches = [
                file for file in files
                if name in file.name.lower() and "seg" not in file.name.lower()
            ]
            if not matches:
                raise FileNotFoundError(f"Missing {name} image in {subject_dir}")
            return str(sorted(matches, key=lambda file: len(file.name))[0])

        labels = [file for file in files if "seg" in file.name.lower()]
        if len(labels) != 1:
            raise FileNotFoundError(f"Expected one segmentation label in {subject_dir}")

        return {
            "image": [first_match(name) for name in MODALITIES],
            "label": str(labels[0]),
        }

    def _records(self, partition_id: int) -> list[dict]:
        table = pd.read_csv(self.config.partition_csv)
        groups = list(table.groupby("Partition_ID", sort=True))

        if not 0 <= partition_id < len(groups):
            raise IndexError(f"Invalid partition-id: {partition_id}")

        _, subjects = groups[partition_id]
        return [
            self._case(self.config.data_root / str(subject_id))
            for subject_id in subjects["Subject_ID"]
        ]

    @staticmethod
    def _transform(train: bool) -> Compose:
        transforms = [
            LoadImaged(keys=("image", "label")),
            EnsureChannelFirstd(keys=("image", "label")),
            Orientationd(keys=("image", "label"), axcodes="RAS"),
            Spacingd(
                keys=("image", "label"),
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            MapLabelValued(keys="label", orig_labels=[4], target_labels=[3]),
            CropForegroundd(keys=("image", "label"), source_key="image"),
        ]

        if train:
            transforms.extend(
                [
                    RandCropByPosNegLabeld(
                        keys=("image", "label"),
                        label_key="label",
                        spatial_size=PATCH_SIZE,
                        pos=1,
                        neg=1,
                        num_samples=2,
                    ),
                    RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=0),
                    RandRotate90d(keys=("image", "label"), prob=0.5, max_k=3),
                ]
            )

        return Compose(transforms + [EnsureTyped(keys=("image", "label"))])

    def load_partition(self, partition_id: int):
        records = self._records(partition_id)
        if len(records) < 2:
            raise ValueError(f"Partition {partition_id} needs at least two subjects")

        indices = np.random.default_rng(
            self.config.seed + partition_id
        ).permutation(len(records))

        validation_size = max(1, round(0.15 * len(records)))
        validation = [records[i] for i in indices[:validation_size]]
        training = [records[i] for i in indices[validation_size:]]

        return (
            DataLoader(
                Dataset(training, self._transform(train=True)),
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
            ),
            DataLoader(
                Dataset(validation, self._transform(train=False)),
                batch_size=1,
                num_workers=self.config.num_workers,
            ),
        )
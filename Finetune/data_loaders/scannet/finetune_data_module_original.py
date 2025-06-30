import numpy as np
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional

from data_loaders.utils.scannet import ScanNet
from data_loaders.utils.scannet_ngp import ScanNetNGP
from utils.loading import sanitize_split_file
from utils.paths import DATASET_PATH, REPO_ROOT


class FineTuneDataModule(pl.LightningDataModule):

    def __init__(
        self,
        exp: dict,
        scene: str,
        validation_equal_test: bool = False
    ):
        super().__init__()
        self.exp = exp
        self.cfg_loader = self.exp["data_module"]
        self.scene = scene
        self.validation_equal_test = validation_equal_test

    def setup(self, stage: Optional[str] = None) -> None:
        ## test adaption (last 20% of the new scenes)
        finetune_seqs = self.exp["scenes"]
        self.scannet_test_ada = ScanNetNGP(
            root=os.path.join(DATASET_PATH, 'scans'),
            mode="val",  # val
            scene=self.scene,
            val_ratio=self.exp['data_module']['data_preprocessing']['val_ratio']
        )
        ## test generation
        split_file = os.path.join(
            REPO_ROOT,
            'scannet',
            'split.npz'
        )
        img_list = sanitize_split_file(np.load(split_file))
        self.scannet_test_gen = ScanNet(
            root=os.path.join(DATASET_PATH, self.cfg_loader["root"]),
            img_list=img_list["test"],
            mode="test",
        )

        self.scannet_train = ScanNetNGP(
            root=os.path.join(DATASET_PATH, 'scans'),
            mode="train",  # val
            scene=self.scene,
            val_ratio=self.exp['data_module']['data_preprocessing']['val_ratio']
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.scannet_train,
            batch_size=self.cfg_loader["batch_size"],
            drop_last=True,
            shuffle=True,  # only true in train_dataloader
            collate_fn=self.scannet_train.collate
            if self.exp["cl"]["active"] else None,
            pin_memory=self.cfg_loader["pin_memory"],
            num_workers=self.cfg_loader["num_workers"],
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.scannet_test_ada if not self.validation_equal_test else self.scannet_train,
            batch_size=
            1,  ## set bs=1 to ensure a batch always has frames from the same scene
            drop_last=False,
            shuffle=False,  # false
            pin_memory=self.cfg_loader["pin_memory"],
            num_workers=self.cfg_loader["num_workers"],
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.scannet_test_gen,
            batch_size=self.cfg_loader["batch_size"],
            drop_last=False,
            shuffle=False,  # false
            pin_memory=self.cfg_loader["pin_memory"],
            num_workers=self.cfg_loader["num_workers"],
        )

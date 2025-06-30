import numpy as np
import os
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from typing import Optional

from data_loaders.utils.scannet import ScanNet
from utils.loading import sanitize_split_file
from utils.paths import DATASET_PATH, REPO_ROOT


class DataModule25K(pl.LightningDataModule):

    def __init__(self, cfg_dm: dict, consider_only_scene: str = ''):
        super().__init__()

        self.cfg_dm = cfg_dm
        self.consider_only_scene = consider_only_scene

    def setup(self, stage: Optional[str] = None) -> None:
        split_file = os.path.join(REPO_ROOT, 'scannet',
            self.cfg_dm["data_preprocessing"]["split_file"],
        )
        img_list = sanitize_split_file(np.load(split_file))
        self.scannet_test = ScanNet(root=os.path.join(DATASET_PATH, self.cfg_dm["root"]),
                                    img_list=[s for s in img_list["test"] if self.consider_only_scene in s],
                                    mode="test")
        self.scannet_train = ScanNet(root=os.path.join(DATASET_PATH, self.cfg_dm["root"]),
                                     img_list=[s for s in img_list["train"] if self.consider_only_scene in s],
                                     mode="train")
        self.scannet_val = ScanNet(root=os.path.join(DATASET_PATH, self.cfg_dm["root"]),
                                   img_list=[s for s in img_list["val"] if self.consider_only_scene in s],
                                   mode="val")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.scannet_train,
            batch_size=self.cfg_dm["batch_size"],
            drop_last=self.cfg_dm["drop_last"],
            shuffle=self.cfg_dm["shuffle"],
            pin_memory=self.cfg_dm["pin_memory"],
            num_workers=self.cfg_dm["num_workers"],
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.scannet_val,
            batch_size=self.cfg_dm["batch_size"],
            drop_last=self.cfg_dm["drop_last"],
            shuffle=False,
            pin_memory=self.cfg_dm["pin_memory"],
            num_workers=self.cfg_dm["num_workers"],
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.scannet_test,
            batch_size=self.cfg_dm["batch_size"],
            drop_last=self.cfg_dm["drop_last"],
            shuffle=False,
            pin_memory=self.cfg_dm["pin_memory"],
            num_workers=self.cfg_dm["num_workers"],
        )

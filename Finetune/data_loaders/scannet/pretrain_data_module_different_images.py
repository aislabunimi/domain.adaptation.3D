import numpy as np
import os
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from typing import Optional

from data_loaders.utils.scannet import ScanNet
from utils.loading import sanitize_split_file
from utils.paths import DATASET_PATH, REPO_ROOT


class DataModule25KTrainDifferent(pl.LightningDataModule):

    def __init__(self, cfg_dm: dict, scene_list: list, consider_only_scene: str = ''):
        super().__init__()

        self.cfg_dm = cfg_dm
        self.consider_only_scene = consider_only_scene
        self.scene_list = scene_list

    def setup(self, stage: Optional[str] = None) -> None:
        img_list = []
        for scene in self.scene_list:
            imgs = [os.path.join(DATASET_PATH, 'scans', scene, 'color', image)
                    for image in os.listdir(os.path.join(DATASET_PATH, 'scans', scene, 'color')) if '50.' in image]
            img_list += imgs

        self.scannet_test = ScanNet(root=os.path.join(DATASET_PATH, 'scans'), label_folder='label-filt',
                                    img_list=[s for s in img_list if self.consider_only_scene in s],
                                    mode="test")
        self.scannet_train = ScanNet(root=os.path.join(DATASET_PATH, 'scans'), label_folder='label-filt',
                                     img_list=[s for s in img_list if self.consider_only_scene in s],
                                     mode="train")
        self.scannet_val = ScanNet(root=os.path.join(DATASET_PATH, 'scans'), label_folder='label-filt',
                                   img_list=[s for s in img_list if self.consider_only_scene in s],
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

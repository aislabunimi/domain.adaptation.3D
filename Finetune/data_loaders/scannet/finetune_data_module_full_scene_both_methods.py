import numpy as np
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional

from data_loaders.utils.scannet_ngp_full_scene_both_methods import ScanNetNGPFullSceneBothMethods


class FineTuneDataModuleFullSceneBothMethods(pl.LightningDataModule):

    def __init__(
        self,
        exp: dict,
        dataset_path: str,
        scene: str,
        deeplab: bool,
        pseudo3d: bool,
        voxel: int,
        imsize_sam: str,
        validation_equal_test: bool = False
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.voxel = voxel
        self.imsize_sam = imsize_sam
        self.exp = exp
        self.deeplab = deeplab
        self.pseudo3d = pseudo3d
        self.cfg_loader = self.exp["data_module"]
        self.scene = scene
        self.validation_equal_test = validation_equal_test

    def setup(self, stage: Optional[str] = None) -> None:
        ## test adaption (last 20% of the new scenes)
        self.scannet_test_validation = ScanNetNGPFullSceneBothMethods(
            root=self.dataset_path,
            mode="val",  # val
            deeplab=self.deeplab,
            pseudo3d=self.pseudo3d,
            scene=self.scene,
            voxel=self.voxel,
            imsize_sam=self.imsize_sam,

        )
        self.scannet_test_gen = ScanNetNGPFullSceneBothMethods(
            root=self.dataset_path,
            mode="val",  # val
            deeplab=self.deeplab,
            pseudo3d=self.pseudo3d,
            scene=self.scene,
            voxel=self.voxel,
            imsize_sam=self.imsize_sam,
        )

        self.scannet_train = ScanNetNGPFullSceneBothMethods(
            root=self.dataset_path,
            mode="train",  # val
            data_augmentation=True,
            scene=self.scene,
            deeplab=self.deeplab,
            pseudo3d=self.pseudo3d,
            voxel=self.voxel,
            imsize_sam=self.imsize_sam,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.scannet_train,
            batch_size=self.cfg_loader["batch_size"],
            drop_last=True,
            shuffle=True,  # only true in train_dataloader
            pin_memory=self.cfg_loader["pin_memory"],
            num_workers=self.cfg_loader["num_workers"],
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.scannet_test_validation,
            batch_size=1,  ## set bs=1 to ensure a batch always has frames from the same scene
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

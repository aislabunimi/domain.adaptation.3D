import os
from pathlib import Path

import pandas as pd
import torch
from pytorch_lightning.strategies import DDPStrategy

from data_loaders.scannet.finetune_data_module import FineTuneDataModule
from data_loaders.scannet.pretrain_data_module import DataModule25K
from pytorch_lightning import seed_everything, Trainer

from models.semantic_segmentator import SemanticsLightningNet
from utils.loading import load_yaml, get_wandb_logger
from utils.paths import REPO_ROOT, DATASET_PATH

#parameters = load_yaml(os.path.join(REPO_ROOT, 'configs', 'pretrain_25k_test_10_scenes.yml'))

parameters = {
    'model': {
        'pretrained': False,
        'pretrained_backbone': True,
        'load_checkpoint': True,
        'checkpoint_path': 'pretrain_25k/best-epoch143-step175536.ckpt',
        'num_classes': 40
    },
    'trainer': {
        'max_epochs': 10,
        'accelerator': 'gpu',
        'num_sanity_val_steps': 0,
        'check_val_every_n_epoch': 1,
    },
    'optimizer': {
        'lr': 1.0e-5,
        'name': 'Adam'
    },
    'lr_scheduler': {
        'active': False,
    },
    'data_module': {
        'pin_memory': True,
        'batch_size': 4,
        'shuffle': True,
        'num_workers': 8,
        'drop_last': True,
        'val_ratio': 0.8
    },
    'visualizer': {
        'store': True,
        'store_n': {
            'train': 10,
            'val': 5000,
            'test': 5000
        }
    }
}

RESULTS_PATH = '/home/antonazzi/myfiles/fine_tune_deeplab'
DATASET_PATH = '/home/antonazzi/myfiles/scannet_signorelli'
TRAIN_MODELS_PATH = '/home/antonazzi/myfiles/fine_tune_deeplab'
seed_everything(123)

voxel = 3
method = 'C'
imsize_sam = 'b'
pseudo3d = False
deeplab = False

if pseudo3d:
    TRAIN_MODELS_PATH_GLOBAL = os.path.join(TRAIN_MODELS_PATH, 'fine_tune_3D', f'pseudo{voxel}')
    experiment_path_global = os.path.join(RESULTS_PATH, 'validation', 'fine_tune_3D', f'pseudo{voxel}')
else:
    TRAIN_MODELS_PATH_GLOBAL = os.path.join(TRAIN_MODELS_PATH, 'fine_tune_sam', f'{method}{imsize_sam}{voxel}')
    experiment_path_global = os.path.join(RESULTS_PATH, 'validation', 'fine_tune_sam', f'{method}{imsize_sam}{voxel}')

for scene in [f'scene000{i}_00' for i in range(0, 10)]:

    TRAIN_MODELS_PATH = os.path.join(TRAIN_MODELS_PATH_GLOBAL, scene, 'lightning_logs')
    last_version = sorted([d for d in os.listdir(TRAIN_MODELS_PATH) if 'version_' in d],
                          key=lambda d: (int(d.replace('version_', ''))))[-1]
    TRAIN_MODELS_PATH = os.path.join(TRAIN_MODELS_PATH, last_version, 'metrics.csv')
    dataframe = pd.read_csv(TRAIN_MODELS_PATH)
    print(f'{scene} -> {round(dataframe["val/mean_IoU_gg"].to_numpy()[-1]*100,1)}')




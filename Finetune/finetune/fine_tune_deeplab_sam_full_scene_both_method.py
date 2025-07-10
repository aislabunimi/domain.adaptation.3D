import os
from pathlib import Path
import torch
from pytorch_lightning.strategies import DDPStrategy

from data_loaders.scannet.finetune_data_module_full_scene_both_methods import FineTuneDataModuleFullSceneBothMethods
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
        'max_epochs': 5,
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
        'store': False,
        'store_n': {
            'train': 10,
            'val': 5000,
            'test': 5000
        }
    }
}

RESULTS_PATH = '/home/antonazzi/myfiles/fine_tune_deeplab'
DATASET_PATH = '/home/antonazzi/myfiles/scannet_signorelli'
seed_everything(123)

voxel = 5
imsize_sam = 's'
pseudo3d = False
deeplab = False

for scene in ['scene0000_01', 'scene0001_01', 'scene0002_01', 'scene0003_01', 'scene0005_01', 'scene0006_01', 'scene0009_01']:
#for scene in ['scene0005_01', 'scene0006_01', 'scene0009_01']:

    experiment_path = os.path.join(RESULTS_PATH, 'fine_tune_sam', f'CA{imsize_sam}{voxel}', scene)


    Path(experiment_path).mkdir(parents=True, exist_ok=True)

    ####################################
    # Load Model
    ###################################

    model = SemanticsLightningNet(parameters, {'results': 'experiments',
                                               'scannet': DATASET_PATH,
                                               'scannet_frames_25k': 'scannet_frames_25k'}, experiment_path=experiment_path)

    if parameters['model']['load_checkpoint']:
        checkpoint = torch.load(os.path.join(RESULTS_PATH, parameters["model"]["checkpoint_path"]))
    checkpoint = checkpoint["state_dict"]
    # remove any aux classifier stuff
    removekeys = [
        key for key in checkpoint.keys()
        if key.startswith("_model._model.aux_classifier")
    ]
    print(removekeys)
    for key in removekeys:
        del checkpoint[key]
    try:
        model.load_state_dict(checkpoint, strict=True)
    except RuntimeError as e:
        print(e)
    model.load_state_dict(checkpoint, strict=False)

    ###############################
    # Prepare datamodule
    ###############################

    datamodule = FineTuneDataModuleFullSceneBothMethods(parameters,
                                    dataset_path=DATASET_PATH,
                                    scene=scene,
                                    deeplab=deeplab,
                                    pseudo3d=pseudo3d,
                                    voxel=voxel,
                                    imsize_sam=imsize_sam)

    trainer = Trainer(**parameters["trainer"],
                      default_root_dir=experiment_path,
                      strategy=DDPStrategy(find_unused_parameters=False),
                      )
    #trainer.validate(model, datamodule=datamodule)

    trainer.validate(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)
    trainer.validate(model, datamodule=datamodule)




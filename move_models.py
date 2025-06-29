import os
import shutil
save_path = '/media/adaptation/New_volume/models_trained'
load_path = '/media/adaptation/D435-A0D8/fine_tune_deeplab'

if not os.path.exists(save_path):
    os.mkdir(save_path)

for folder in ['fine_tune_sam', 'fine_tune_3D']:
    for mod in os.listdir(os.path.join(load_path, folder)):
        for scene in os.listdir(os.path.join(load_path, folder, mod)):
            os.makedirs(os.path.join(save_path, folder, mod, scene), exist_ok=True)
            model_path = os.path.join(load_path, folder, mod, scene, 'lightning_logs')
            last_version = sorted([d for d in os.listdir(model_path) if 'version_' in d],
                          key=lambda d: (int(d.replace('version_', ''))))[-1]
            model_path = os.path.join(model_path, last_version, 'checkpoints', )
            print(model_path)
            checkpoint = os.listdir(model_path)[0]
            model_path = os.path.join(model_path, checkpoint)
            if not os.path.exists(os.path.join(save_path, folder, mod, scene, 'model.ckpt')):
                shutil.copyfile(model_path, os.path.join(save_path, folder, mod, scene, 'model.ckpt'))

    
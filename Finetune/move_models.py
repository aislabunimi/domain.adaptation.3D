import os
import shutil

path = '/media/adaptation/D435-A0D8/fine_tune_deeplab/fine_tune_sam/CAs3'

destination = '/media/adaptation/New_volume/models_trained/fine_tune_sam/CAs3'

for env in os.listdir(path):
    cpy_path = os.path.join(path, env, 'lightning_logs')

    v = sorted(os.listdir(cpy_path), key=lambda x: int(x.split('_')[-1]))[-1]

    cpy_path = os.path.join(cpy_path, v, 'checkpoints')
    cpy_path = os.path.join(cpy_path, os.listdir(cpy_path)[0])

    os.mkdir(os.path.join(destination, env))
    shutil.copy(cpy_path, os.path.join(destination, env, 'model.ckpt'))
import collections

import yaml
from pytorch_lightning.loggers import WandbLogger


def load_yaml(path):
    with open(path) as file:
        res = yaml.load(file, Loader=yaml.FullLoader)
    return res

def sanitize_split_file(split):
    split = dict(split)
    for key, value in split.items():
        split[key] = [v.replace('data/scannet_frames_25k/scannet_frames_25k/', '') for v in value]

    return split

def get_wandb_logger(exp, project_name, save_dir):
    #params = log_important_params(exp)
    name_full = exp["general"]["name"]
    name_short = "__".join(name_full.split("/")[-2:])
    return WandbLogger(
        name=name_short,
        project=project_name,
        save_dir=save_dir,
    )
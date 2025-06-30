import os

import pandas as pd

from utils.paths import RESULTS_PATH

scenes = [f'scene{i:04}' for i in range(588, 707)]

values = []

experiment_path = os.path.join(RESULTS_PATH, 'pretrain_25k_test_single_scene')

for count, scene in enumerate(scenes):
    df = pd.read_csv(os.path.join(experiment_path, f'test_{scene}', 'lightning_logs', 'version_0', 'metrics.csv'))

    mIoU = df.loc[0, ['test/mean_IoU']].iloc[0].item()*100
    values.append([588 + count, mIoU])

df = pd.DataFrame(values, columns=['env', 'mIoU'])
df.to_csv(os.path.join(experiment_path, 'aggregation.csv'))




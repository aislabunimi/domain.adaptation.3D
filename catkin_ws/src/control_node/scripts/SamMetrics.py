import numpy as np
import os
import json
from typing import Union

class SamMetrics:
    def __init__(self, log_path="sam_metrics_log.json", reset_log=False):
        self.log_path = log_path
        self.change_percentages = []
        self.global_conf_matrix = np.zeros((40, 40), dtype=np.int64)
        self.log_entries = []

        if reset_log or not os.path.exists(self.log_path):
            with open(self.log_path, 'w') as f:
                json.dump([], f)

    def update(self, frame_id: str, info: str, original: np.ndarray, gt: np.ndarray, refined: np.ndarray):
        assert original.shape == refined.shape == gt.shape
        assert np.issubdtype(original.dtype, np.integer)

        # Focus only on relevant ground truth areas (gt != -1)
        relevant_mask = gt != -1
        mask_changed = (original != refined) & relevant_mask
        num_changed = np.count_nonzero(mask_changed)
        num_relevant = np.count_nonzero(relevant_mask)

        percent_changed = (num_changed / num_relevant) * 100.0 if num_relevant > 0 else 0.0
        self.change_percentages.append(percent_changed)

        # Confusion matrix: only within relevant areas
        changed_from = original[mask_changed]
        changed_to = refined[mask_changed]
        conf_matrix = np.zeros((40, 40), dtype=np.int64)

        valid_mask = (changed_from >= 0) & (changed_from < 40) & (changed_to >= 0) & (changed_to < 40)
        np.add.at(conf_matrix, (changed_to[valid_mask], changed_from[valid_mask]), 1)
        np.fill_diagonal(conf_matrix, 0)
        np.add.at(self.global_conf_matrix, (changed_to[valid_mask], changed_from[valid_mask]), 1)

        self.log_entries.append({
            "frame_id": frame_id,
            "info": info,
            "percent_changed": percent_changed,
            "confusion_matrix": conf_matrix.tolist()
        })

    def measure(self):
        avg = float(np.mean(self.change_percentages)) if self.change_percentages else 0.0
        gcm = self.global_conf_matrix.copy()
        np.fill_diagonal(gcm, 0)
        return avg, gcm

    def save_log(self):
        with open(self.log_path, 'w') as f:
            json.dump(self.log_entries, f, indent=2)
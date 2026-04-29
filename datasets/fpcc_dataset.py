"""PyTorch dataset wrappers for original FPCC h5 files."""

import numpy as np
import torch
from torch.utils.data import Dataset

import provider


class FPCCDataset(Dataset):
    def __init__(self, file_list_txt, point_dim=6, num_groups=50):
        self.files = provider.getDataFiles(file_list_txt)
        self.point_dim = point_dim
        self.num_groups = num_groups

        self.data = []
        self.group = []
        self.center = []
        for f in self.files:
            cur_data, cur_group, _, _, cur_score = provider.loadDataFile_with_groupseglabel_stanfordindoor(f)
            self.data.append(cur_data)
            self.group.append(cur_group)
            self.center.append(cur_score)

        self.data = np.concatenate(self.data, axis=0).astype(np.float32)
        self.group = np.concatenate(self.group, axis=0).astype(np.int64)
        self.center = np.concatenate(self.center, axis=0).astype(np.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        points = self.data[idx, :, : self.point_dim]
        instance_labels = np.asarray(self.group[idx]).reshape(-1)
        center_labels = np.asarray(self.center[idx]).reshape(-1)
        one_hot = group_to_one_hot(instance_labels, self.num_groups)
        return {
            "points": torch.from_numpy(points),
            "instance_labels": torch.from_numpy(instance_labels),
            "center_labels": torch.from_numpy(center_labels),
            "group_one_hot": torch.from_numpy(one_hot),
        }


def group_to_one_hot(group_labels, num_groups=50):
    """group_labels: [N], output [N,num_groups]."""
    group_labels = np.asarray(group_labels)
    one_hot = np.zeros((group_labels.shape[0], num_groups), dtype=np.float32)
    uniq = [u for u in np.unique(group_labels) if u >= 0]
    mapping = {u: i for i, u in enumerate(uniq[:num_groups])}
    for i, gid in enumerate(group_labels):
        if gid in mapping:
            one_hot[i, mapping[gid]] = 1.0
    return one_hot

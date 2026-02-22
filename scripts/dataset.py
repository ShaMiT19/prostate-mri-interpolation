import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import random
import torchvision.transforms.functional as TF


class ProstateSliceDataset(Dataset):
    def __init__(self, data_dir, split_file, mode="train", augment=False):
        self.data_dir = data_dir
        self.mode = mode
        self.augment = augment

        with open(split_file, "rb") as f:
            splits = pickle.load(f)

        self.patient_ids = splits[mode]

        self.volumes = {
            pid: np.load(os.path.join(data_dir, pid + ".npy")).astype(np.float32)
            for pid in self.patient_ids
        }

        self.index_list = []
        for pid, vol in self.volumes.items():
            S = vol.shape[0]
            for i in range(1, S - 1):
                self.index_list.append((pid, i))

        print(f"{mode} samples:", len(self.index_list))

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        pid, i = self.index_list[idx]
        vol = self.volumes[pid]

        x1 = vol[i - 1]
        x2 = vol[i + 1]
        X = np.stack([x1, x2], axis=0)

        y = vol[i][None, :, :]

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        if self.augment and self.mode == "train":
            if random.random() < 0.5:
                X = TF.hflip(X)
                y = TF.hflip(y)
            if random.random() < 0.5:
                X = TF.vflip(X)
                y = TF.vflip(y)
            if random.random() < 0.5:
                X = X.rot90(1, dims=(1, 2))
                y = y.rot90(1, dims=(1, 2))

        return X, y

    

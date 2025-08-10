import torch
from torch.utils.data import Dataset
import numpy as np
import os

class MFCCFileDataset(Dataset):
    def __init__(self, data_dir):
        self.sample_paths = []
        self.labels = []

        for label_name in ['keyword', 'background']:
            label_dir = os.path.join(data_dir, label_name)
            label = 1 if label_name == 'keyword' else 0

            for file in os.listdir(label_dir):
                if file.endswith('.csv'):
                    self.sample_paths.append(os.path.join(label_dir, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        path = self.sample_paths[idx]
        mfcc = np.loadtxt(path, delimiter=',')  # shape: (399, 12)
        mfcc = mfcc.astype(np.float32) / 32768.0  # Normalize Q15 to [0, 1]
        mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # shape: (1, 399, 12)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return mfcc, label

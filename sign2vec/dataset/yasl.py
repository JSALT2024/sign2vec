import torch
from torch import nn


class YouTubeASLDatasetForPretraining(torch.utils.data.Dataset):
    def __init__(self, 
                 dataset, 
                 data_dir, 
                 max_length):
        
        self.dataset = dataset
        self.data_dir = data_dir
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
class ReviewsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx].astype(np.float32)), self.y[idx]
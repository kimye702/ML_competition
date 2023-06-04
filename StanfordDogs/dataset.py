import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F

def toTensor(x):
    return torch.tensor([x])

def toOne_hot(x):
    return torch.FloatTensor(np.array(F.one_hot(x, 120)))

class StanfordDataset(Dataset):
    def __init__(self, src, resol, transform=None):
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        
        self.transform = transform
        
        self.target_transfrom = transforms.Compose([
            transforms.Lambda(toTensor),
            transforms.Lambda(toOne_hot),
        ])

        self.dataset = ImageFolder(
            src,
            transform=self.transform,
            target_transform=self.target_transfrom
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][0], self.dataset[idx][1]

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
    def __init__(self, src, resol):
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        
        self.transform = transforms.Compose([
            transforms.Resize(int(resol/0.875)),
            transforms.RandomCrop(resol),             # 무작위로 이미지를 자르는 데이터 증대 기법
            transforms.RandomHorizontalFlip(),        # 무작위로 이미지를 좌우 반전하는 데이터 증대 기법
            transforms.ToTensor(),
            self.normalize
        ])
        self.target_transform = transforms.Compose([
            transforms.Lambda(toTensor),
            transforms.Lambda(toOne_hot),
        ])

        self.dataset = ImageFolder(
            src,
            transform=self.transform,
            target_transform=self.target_transform
        )

    def __len__(self): 
        return len(self.dataset)

    def __getitem__(self, idx): 
        return self.dataset[idx]
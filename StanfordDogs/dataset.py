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
            transforms.RandomCrop(resol),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),                  # 이미지를 무작위로 회전하는 데이터 증대 기법
            transforms.RandomAffine(0, shear=0.05, scale=(0.8, 1.2)),   # 이미지를 무작위로 이동, 변형하는 데이터 증대 기법
            transforms.ColorJitter(brightness=0.5, contrast=0.5),       # 이미지의 밝기와 대비를 무작위로 조정하는 데이터 증대 기법
            transforms.RandomPerspective(distortion_scale=0.2),         # 이미지의 원근 변환을 무작위로 적용하는 데이터 증대 기법
            transforms.RandomResizedCrop(resol, scale=(0.8, 1.2), ratio=(0.75, 1.33)),  # 이미지를 무작위로 자르고 크기를 조정하는 데이터 증대 기법
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
        self.dataset+=target_transform

    def __len__(self): 
        return len(self.dataset)

    def __getitem__(self, idx): 
        return self.dataset[idx]

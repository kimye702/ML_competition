import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from torch.utils.data import ConcatDataset

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
        
        self.transform = transforms.Compose(
            [transforms.Resize(int(resol/0.875)),
            transforms.CenterCrop(resol),
            transforms.ToTensor(),
            self.normalize
            ])

        self.transform2 = transforms.Compose([
            transforms.Resize(int(resol/0.875)),
            transforms.RandomCrop(resol),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomAffine(0, shear=0.05, scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.RandomPerspective(distortion_scale=0.2),
            transforms.RandomResizedCrop(resol, scale=(0.8, 1.2), ratio=(0.75, 1.33)),
            transforms.ToTensor(),
            self.normalize
        ])
        self.target_transform = transforms.Compose([
            transforms.Lambda(toTensor),
            transforms.Lambda(toOne_hot),
        ])

        self.original_dataset = ImageFolder(
            src,
            transform=self.transform,
            target_transform=self.target_transform
        )

        self.transformed_dataset = ImageFolder(
            src,
            transform=self.transform2,
            target_transform=self.target_transform
        )

        self.augmentation = ConcatDataset([self.original_dataset, self.transformed_dataset])

    def __len__(self): 
        return len(self.augmentation)

    def __getitem__(self, idx): 
        return self.augmentation[idx]


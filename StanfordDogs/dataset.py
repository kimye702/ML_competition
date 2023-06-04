from torchvision.datasets import ImageFolder

class StanfordDataset(ImageFolder):
    def __init__(self, src, resol, transform=None, target_transform=None):
        super().__init__(
            src,
            transform=transform,
            target_transform=target_transform
        )

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label

    def __len__(self):
        return len(self.dataset)


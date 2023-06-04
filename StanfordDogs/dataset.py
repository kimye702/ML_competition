class StanfordDataset(Dataset):
    def __init__(self, src, resol, transform=None):
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        
        self.transform = transform
        
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
        image = self.dataset[idx][0]
        label = self.dataset[idx][1]
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label

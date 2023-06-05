class StanfordDataset(Dataset): 
    def __init__(self, src, resol, transformed_src):
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        
        self.transform = transforms.Compose([
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

        self.dataset = ImageFolder(
            src,
            transform=self.transform,
            target_transform=self.target_transform
        )

        self.transformed_dataset = ImageFolder(
            transformed_src,
            transform=self.transform,
            target_transform=self.target_transform
        )

    def __len__(self): 
        return len(self.dataset) + len(self.transformed_dataset)

    def __getitem__(self, idx): 
        if idx < len(self.dataset):
            return self.dataset[idx]
        else:
            transformed_idx = idx - len(self.dataset)
            return self.transformed_dataset[transformed_idx]

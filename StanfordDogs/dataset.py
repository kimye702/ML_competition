class StanfordDataset(Dataset):
    def __init__(self, src, resol):
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

        self.target_transform = transforms.Compose([
            transforms.Lambda(toTensor),
            transforms.Lambda(toOne_hot),
        ])

        self.dataset = ImageFolder(
            src,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(resol),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                self.normalize
            ]),
            target_transform=self.target_transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        label = self.dataset[idx][1]

        return image, label

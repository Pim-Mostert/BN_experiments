import random

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MNISTSequenceDataset(Dataset):
    def __init__(self, root):
        self._mnist = datasets.MNIST(
            root=root,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.flatten()),
                ]
            ),
        )

        # Build index pools per digit
        self.digit_to_indices = {i: [] for i in range(10)}
        for idx, (_, label) in enumerate(iter(self._mnist)):
            self.digit_to_indices[label].append(idx)

        # Shuffle each digit pool once
        for d in range(10):
            random.shuffle(self.digit_to_indices[d])

        # Number of sequences limited by smallest digit pool
        self.num_sequences = min(len(v) for v in self.digit_to_indices.values())

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        images = []

        for digit in range(10):
            mnist_idx = self.digit_to_indices[digit][idx]
            img, _ = self._mnist[mnist_idx]
            images.append(img)

        # Shape: (10, 784)
        return torch.stack(images)

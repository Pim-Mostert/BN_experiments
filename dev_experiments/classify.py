# %% Imports

import logging


import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from bayesian_network.common.torch_settings import TorchSettings


logging.basicConfig(level=logging.INFO)

torch.set_printoptions(sci_mode=False)

# %%

TORCH_SETTINGS = TorchSettings(
    device="cpu",
    dtype="float64",
)

BATCH_SIZE = 64
LEARNING_RATE = 0.02
EPOCHS = 200

# %% Load data

height, width = 28, 28
num_classes = 10

transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.flatten()),
    ]
)

mnist_train = DataLoader(
    dataset=torchvision.datasets.MNIST(
        "./experiments/mnist", train=True, download=True, transform=transforms
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

mnist_test = DataLoader(
    dataset=torchvision.datasets.MNIST(
        "./experiments/mnist", train=False, download=True, transform=transforms
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# %% Define model


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


model = LogisticRegressionModel(height * width, num_classes).to(TORCH_SETTINGS.device)

# %% Prepare experiment

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# %% Training loop

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in mnist_train:
        images, labels = images.to(TORCH_SETTINGS.device), labels.to(TORCH_SETTINGS.device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {total_loss:.4f}")

# %% Evaluation

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in mnist_test:
        images, labels = images.to(TORCH_SETTINGS.device), labels.to(TORCH_SETTINGS.device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# %%

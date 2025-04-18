# %% Imports
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms


from bayesian_network.inference_machines.spa_v3.spa_inference_machine import (
    SpaInferenceMachine,
    SpaInferenceMachineSettings,
)


from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.optimizers.em_batch_optimizer import (
    EmBatchOptimizer,
    EmBatchOptimizerSettings,
)

import logging

from experiments.mnist.common import MLflowOptimizerLogger, MnistEvidenceBatches

logging.basicConfig(level=logging.INFO)

# %% tags=["parameters"]

TORCH_SETTINGS = TorchSettings(
    device="cpu",
    dtype="float64",
)

BATCH_SIZE = 200

EM_BATCH_OPTIMIZER_SETTINGS = EmBatchOptimizerSettings(
    num_iterations=20,
    learning_rate=0.05,
)

GAMMA = 0.001

TRUE_MEANS_NOISE = 0.2

# %% Load data

batches = MnistEvidenceBatches(
    TORCH_SETTINGS,
    GAMMA,
    BATCH_SIZE,
)

height, width = batches.mnist_dimensions

# %% True means

mnist = torchvision.datasets.MNIST(
    "./experiments/mnist",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)

data = (mnist.train_data / 255).to(torch.float64)

mu_true = [data[mnist.targets == i].mean(dim=0) for i in range(0, 10)]
mu_true = torch.stack(mu_true, dim=2)

plt.figure()
plt.title("True means")
for i in range(0, 10):
    plt.subplot(4, 3, i + 1)
    plt.imshow(mu_true[:, :, i])
    plt.colorbar()
    plt.clim(0, 1)


# %% Define network
num_classes = 10

# Create network
Q = Node(
    torch.ones(
        (num_classes),
        device=TORCH_SETTINGS.device,
        dtype=TORCH_SETTINGS.dtype,
    )
    / num_classes,
    name="Q",
)

noise = torch.rand(
    (height, width, num_classes),
    device=TORCH_SETTINGS.device,
    dtype=TORCH_SETTINGS.dtype,
)
mu = (1 - TRUE_MEANS_NOISE) * mu_true + TRUE_MEANS_NOISE * noise
mu = torch.stack([1 - mu, mu], dim=3)
Ys = [Node(mu[iy, ix], name=f"Y_{iy}x{ix}") for iy in range(height) for ix in range(width)]

nodes = [Q] + Ys
parents = {Y: [Q] for Y in Ys}
parents[Q] = []

network = BayesianNetwork(nodes, parents)


# %% Fit network
logger = MLflowOptimizerLogger()

em_optimizer = EmBatchOptimizer(
    bayesian_network=network,
    inference_machine_factory=lambda network: SpaInferenceMachine(
        settings=SpaInferenceMachineSettings(
            torch_settings=TORCH_SETTINGS,
            num_iterations=3,
            average_log_likelihood=True,
        ),
        bayesian_network=network,
        observed_nodes=Ys,
        num_observations=batches.batch_size,
    ),
    settings=EM_BATCH_OPTIMIZER_SETTINGS,
    logger=logger,
)
em_optimizer.optimize(batches)

# %% Plot

Ys = network.nodes[1:]
w = torch.stack([y.cpt.cpu() for y in Ys])

plt.figure()
plt.title("Fitted model means")
for i in range(0, 10):
    plt.subplot(4, 3, i + 1)
    plt.imshow(w[:, i, 1].reshape(28, 28))
    plt.colorbar()
    plt.clim(0, 1)

plt.figure()
plt.xlabel("Iteration")
plt.ylabel("Average log-likelihood")
plt.plot(logger.get_log_likelihood())

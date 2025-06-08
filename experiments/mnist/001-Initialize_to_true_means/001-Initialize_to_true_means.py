# %% Imports
import logging
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import mlflow
import torch
from matplotlib.figure import Figure
import torchvision
from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.inference_machines.spa_v3.spa_inference_machine import (
    SpaInferenceMachine,
    SpaInferenceMachineSettings,
)
from bayesian_network.optimizers.em_batch_optimizer import (
    EmBatchOptimizer,
    EmBatchOptimizerSettings,
)
from torchvision import transforms

from experiments.mnist.common import MLflowOptimizerLogger, MnistEvidenceBatches

logging.basicConfig(level=logging.INFO)

# %% tags=["parameters"]

TORCH_SETTINGS = TorchSettings(
    device="cpu",
    dtype="float64",
)

BATCH_SIZE = 100
LEARNING_RATE = 0.1

TRUE_MEANS_NOISE = 0

NUM_ITERATIONS = 200
GAMMA = 0.001

# %% True means

mnist = torchvision.datasets.MNIST(
    "./experiments/mnist",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)

data = (mnist.train_data / 255).to(torch.float64)

mu_true = torch.stack(
    [data[mnist.targets == i].mean(dim=0) for i in range(0, 10)],
    dim=2,
)

# Normalize according to GAMMA
mu_true = mu_true * (1 - GAMMA) + GAMMA / 2

# %% Plot means


def plot_means(mu: Iterable[torch.Tensor]) -> Figure:
    figure = plt.figure()
    figure.suptitle("True means")
    for i, m in enumerate(mu):
        plt.subplot(4, 3, i + 1)
        plt.imshow(m)
        plt.colorbar()
        plt.clim(0, 1)

    return figure


plot_means(mu_true.permute([2, 0, 1]))


# %% Load data
def create_batches(batch_size, gamma) -> MnistEvidenceBatches:
    batches = MnistEvidenceBatches(
        torch_settings=TORCH_SETTINGS,
        batch_size=batch_size,
        gamma=gamma,
    )

    return batches


# %% Define network
def create_network(true_means_noise) -> Tuple[BayesianNetwork, List[Node]]:
    num_classes = 10

    # Create root node
    Q = Node(
        torch.ones(
            (num_classes),
            device=TORCH_SETTINGS.device,
            dtype=TORCH_SETTINGS.dtype,
        )
        / num_classes,
        name="Q",
    )

    # Create leave nodes - mu
    height, width = mu_true.shape[0:2]

    noise = torch.rand(
        (height, width, num_classes),
        device=TORCH_SETTINGS.device,
        dtype=TORCH_SETTINGS.dtype,
    )

    mu = (1 - true_means_noise) * mu_true + true_means_noise * noise
    mu = torch.stack([1 - mu, mu], dim=3)

    # Create leave nodes
    Ys = [Node(mu[iy, ix], name=f"Y_{iy}x{ix}") for iy in range(height) for ix in range(width)]

    # Create network
    nodes = [Q, *Ys]
    parents = {Y: [Q] for Y in Ys}
    parents[Q] = []

    network = BayesianNetwork(nodes, parents)

    return network, Ys


# %% Fit network


def fit_network(
    network: BayesianNetwork,
    observed_nodes: List[Node],
    batches: MnistEvidenceBatches,
    em_batch_optimizer_settings: EmBatchOptimizerSettings,
) -> Tuple[BayesianNetwork, MLflowOptimizerLogger]:
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
            observed_nodes=observed_nodes,
            num_observations=batches.batch_size,
        ),
        settings=em_batch_optimizer_settings,
        logger=logger,
    )
    em_optimizer.optimize(batches)

    return network, logger


# %% Experiment


network, observed_nodes = create_network(TRUE_MEANS_NOISE)

batches = create_batches(BATCH_SIZE, GAMMA)
width, height = batches.mnist_dimensions

em_batch_optimizer_settings = EmBatchOptimizerSettings(
    num_iterations=NUM_ITERATIONS,
    learning_rate=LEARNING_RATE,
)

network, logger = fit_network(
    network,
    observed_nodes,
    batches,
    em_batch_optimizer_settings,
)

mlflow.log_metric("Log-likelihood", logger.get_log_likelihood()[-1])

# Plot means
w = (
    torch.stack([y.cpt.cpu()[:, 1] for y in observed_nodes])
    .reshape(width, height, 10)
    .permute([2, 0, 1])
)

figure = plot_means(w)
figure.suptitle("Means")
mlflow.log_figure(figure, "means.png")

# Plot log_likelihood
figure = plt.figure()
plt.xlabel("Iteration")
plt.ylabel("Average log-likelihood")
plt.plot(logger.get_log_likelihood())
mlflow.log_figure(figure, "avg_ll.png")

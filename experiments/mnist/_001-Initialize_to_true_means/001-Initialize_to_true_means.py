# %% Imports
import logging
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import mlflow
import torch
from matplotlib.figure import Figure
from torch.utils.data import DataLoader
import torchvision
from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.inference_machines.evidence import Evidence, EvidenceLoader
from bayesian_network.inference_machines.spa_v3.spa_inference_machine import (
    SpaInferenceMachine,
    SpaInferenceMachineSettings,
)
from bayesian_network.optimizers.abstractions import IEvaluator
from bayesian_network.optimizers.common import OptimizerLogger

from bayesian_network.optimizers.em_batch_optimizer import (
    EmBatchOptimizer,
    EmBatchOptimizerSettings,
)
from torchvision import transforms

from experiments.mnist.common import MLflowBatchEvaluator, MLflowOptimizerLogger

logging.basicConfig(level=logging.INFO)

# %% tags=["parameters"]

TORCH_SETTINGS = TorchSettings(
    device="cpu",
    dtype="float64",
)
BATCH_SIZE = 100
LEARNING_RATE = 0.1

TRUE_MEANS_NOISE = 0.1

REGULARIZATION = 0.01

NUM_EPOCHS = 5

# %% Load data

gamma = 0.001
mnist = torchvision.datasets.MNIST(
    "./experiments/mnist",
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.flatten()),
            transforms.Lambda(lambda x: x * (1 - gamma) + gamma / 2),
        ]
    ),
)

height, width = 28, 28
num_classes = 10

iterations_per_epoch = len(mnist) / BATCH_SIZE
assert int(iterations_per_epoch) == iterations_per_epoch, (
    "len(mnist) / BATCH_SIZE should be an integer"
)
iterations_per_epoch = int(iterations_per_epoch)

# %% True means

data_loader = DataLoader(
    dataset=mnist,
    batch_size=1000,
    shuffle=False,
)

# To store sums and counts per class
sums = torch.zeros(num_classes, height * width)
counts = torch.zeros(num_classes)

# Iterate over batches
for images, labels in data_loader:
    for i in range(num_classes):
        mask = labels == i
        sums[i] += images[mask].sum(dim=0)
        counts[i] += mask.sum()

# Compute means
mu_true = sums / counts[:, None]


def plot_means(mu: Iterable[torch.Tensor]) -> Figure:
    figure = plt.figure()
    figure.suptitle("True means")
    for i, m in enumerate(mu):
        plt.subplot(4, 3, i + 1)
        plt.imshow(m.reshape(height, width))
        plt.colorbar()
        plt.clim(0, 1)

    return figure


plot_means(mu_true)


# %% Load data
def transform(batch: torch.Tensor) -> Evidence:
    return Evidence(
        [
            torch.stack(
                [
                    1 - x,
                    x,
                ],
                dim=1,
            )
            for x in batch.T
        ],
        torch_settings=TORCH_SETTINGS,
    )


evidence_loader = EvidenceLoader(
    DataLoader(
        dataset=mnist,
        batch_size=BATCH_SIZE,
        shuffle=True,
    ),
    transform=transform,
)


# %% Define network
def create_network(true_means_noise) -> Tuple[BayesianNetwork, List[Node]]:
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
    noise = torch.rand(
        (num_classes, height * width),
        device=TORCH_SETTINGS.device,
        dtype=TORCH_SETTINGS.dtype,
    )

    mu = (1 - true_means_noise) * mu_true + true_means_noise * noise
    mu = torch.stack([1 - mu, mu], dim=2)

    # Create leave nodes
    Ys = [
        Node(mu[:, iy + ix * height], name=f"Y_{iy}x{ix}")
        for iy in range(height)
        for ix in range(width)
    ]

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
    evidence_loader: EvidenceLoader,
    logger: OptimizerLogger,
    evaluator: IEvaluator,
    em_batch_optimizer_settings: EmBatchOptimizerSettings,
) -> BayesianNetwork:
    em_optimizer = EmBatchOptimizer(
        bayesian_network=network,
        inference_machine_factory=lambda network: SpaInferenceMachine(
            settings=spa_inference_machine_settings,
            bayesian_network=network,
            observed_nodes=observed_nodes,
            num_observations=BATCH_SIZE,
        ),
        settings=em_batch_optimizer_settings,
        logger=logger,
        evaluator=evaluator,
    )
    em_optimizer.optimize(evidence_loader)

    return network


# %% Experiment

network, observed_nodes = create_network(TRUE_MEANS_NOISE)

em_batch_optimizer_settings = EmBatchOptimizerSettings(
    learning_rate=LEARNING_RATE,
    num_epochs=NUM_EPOCHS,
    regularization=REGULARIZATION,
)

logger = MLflowOptimizerLogger(iterations_per_epoch=iterations_per_epoch)

spa_inference_machine_settings = SpaInferenceMachineSettings(
    torch_settings=TORCH_SETTINGS,
    num_iterations=4,
    average_log_likelihood=True,
)

evaluator_batch_size = 1000
evaluator = MLflowBatchEvaluator(
    iterations_per_epoch=iterations_per_epoch,
    inference_machine_factory=lambda network: SpaInferenceMachine(
        settings=spa_inference_machine_settings,
        bayesian_network=network,
        observed_nodes=observed_nodes,
        num_observations=evaluator_batch_size,
    ),
    evidence_loader=EvidenceLoader(
        DataLoader(
            dataset=mnist,
            batch_size=evaluator_batch_size,
        ),
        transform=transform,
    ),
    should_evaluate=lambda epoch, iteration: (
        (iteration == 0)
        or (iteration == int(iterations_per_epoch / 2))
        or (epoch == (NUM_EPOCHS - 1) and (iteration == iterations_per_epoch - 1))
    ),
)

network = fit_network(
    network,
    observed_nodes,
    evidence_loader,
    logger,
    evaluator,
    em_batch_optimizer_settings,
)

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
train_iterations = [log.epoch * iterations_per_epoch + log.iteration for log in logger.logs]
train_values = [log.ll for log in logger.logs]
eval_iterations = [
    epoch * iterations_per_epoch + iteration
    for epoch, iteration in evaluator.log_likelihoods.keys()
]
eval_values = list(evaluator.log_likelihoods.values())

plt.figure()
plt.plot(train_iterations, train_values, label="Train")
plt.plot(eval_iterations, eval_values, label="Eval")
plt.xlabel("Iterations")
plt.ylabel("Average log-likelihood")
plt.legend()
mlflow.log_figure(figure, "avg_ll_iterations.png")

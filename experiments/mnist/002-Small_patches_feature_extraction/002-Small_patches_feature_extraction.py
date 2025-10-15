# %% Imports

import logging


import matplotlib.pyplot as plt
import mlflow
import torch
from torch.utils.data import DataLoader
import torchvision
from bayesian_network.bayesian_network import Node, BayesianNetworkBuilder
from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.inference_machines.evidence import Evidence, EvidenceLoader
from bayesian_network.inference_machines.spa_v3.spa_inference_machine import (
    SpaInferenceMachine,
    SpaInferenceMachineSettings,
)

from bayesian_network.optimizers.em_batch_optimizer import (
    EmBatchOptimizer,
    EmBatchOptimizerSettings,
)
from torchvision import transforms

from experiments.mnist.common import MLflowBatchEvaluator, MLflowOptimizerLogger

logging.basicConfig(level=logging.INFO)

# %% tags=["parameters"]

NUM_CLASSES = 10

# %% PARAMETERS

TORCH_SETTINGS = TorchSettings(
    device="cpu",
    dtype="float64",
)

BATCH_SIZE = 1000
LEARNING_RATE = 0.1
REGULARIZATION = 0.001
NUM_EPOCHS = 4

# %% Load data

height, width = 4, 4

x_slice = slice(int((28 - width) / 2), int((28 + width) / 2))
y_slice = slice(int((28 - height) / 2), int((28 + height) / 2))

mnist = torchvision.datasets.MNIST(
    "./experiments/mnist",
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:, y_slice, x_slice]),
            transforms.Lambda(lambda x: x.flatten()),
        ]
    ),
)


iterations_per_epoch = len(mnist) / BATCH_SIZE
assert int(iterations_per_epoch) == iterations_per_epoch, (
    "len(mnist) / BATCH_SIZE should be an integer"
)
iterations_per_epoch = int(iterations_per_epoch)


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

# Create root node
Q = Node.random((NUM_CLASSES), torch_settings=TORCH_SETTINGS, name="Q")

# Create leave nodes
Ys = [
    Node.random((NUM_CLASSES, 2), torch_settings=TORCH_SETTINGS, name=f"Y_{iy}x{ix}")
    for iy in range(height)
    for ix in range(width)
]

# Create network
network = BayesianNetworkBuilder().add_node(Q).add_nodes(Ys, parents=Q).build()


# %% Prepare experiment

em_batch_optimizer_settings = EmBatchOptimizerSettings(
    learning_rate=LEARNING_RATE,
    num_epochs=NUM_EPOCHS,
    regularization=REGULARIZATION,
)

logger = MLflowOptimizerLogger(iterations_per_epoch=iterations_per_epoch)


def create_inference_machine_factory(num_observations):
    def inference_machine_factory(network):
        return SpaInferenceMachine(
            settings=SpaInferenceMachineSettings(
                torch_settings=TORCH_SETTINGS,
                average_log_likelihood=True,
            ),
            bayesian_network=network,
            observed_nodes=Ys,
            num_observations=num_observations,
        )

    return inference_machine_factory


evaluator_batch_size = 1000
evaluator = MLflowBatchEvaluator(
    iterations_per_epoch=iterations_per_epoch,
    inference_machine_factory=create_inference_machine_factory(evaluator_batch_size),
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

# %% Run experiment

em_optimizer = EmBatchOptimizer(
    bayesian_network=network,
    inference_machine_factory=create_inference_machine_factory(BATCH_SIZE),
    settings=em_batch_optimizer_settings,
    logger=logger,
    evaluator=evaluator,
)
em_optimizer.optimize(evidence_loader)

# %% Plots

# Plot means

W = (
    torch.stack([y.cpt.cpu()[:, 1] for y in Ys])
    .reshape(width, height, NUM_CLASSES)
    .permute([2, 0, 1])
)

figure = plt.figure()
figure.suptitle("Weights")

for i, w in enumerate(W):
    plt.subplot(int(NUM_CLASSES**0.5) + 1, int(NUM_CLASSES**0.5) + 1, i + 1)
    plt.imshow(w.reshape(height, width))
    plt.colorbar()
    plt.clim(0, 1)

mlflow.log_figure(figure, "weights.png")

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

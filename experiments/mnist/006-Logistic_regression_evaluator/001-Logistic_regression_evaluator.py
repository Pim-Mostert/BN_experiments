# %% Imports

import logging


import matplotlib.pyplot as plt
import mlflow
import torch
from torch.utils.data import DataLoader, Subset
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

from experiments.mnist.logistic_regression_evaluator import (
    LogisticRegressionEvaluatorSettings,
    MLflowLogisticRegressionEvaluator,
)

from experiments.mnist.common import (
    MLflowOptimizerLogger,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s][%(asctime)s][%(module)s.%(funcName)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
torch.set_printoptions(sci_mode=False)

# %% tags=["parameters"]

# TODO: DAG ERBIJ MAKEN EN EXPERIMENT AFTRAPPEN
# Param(True, type="boolean")

NUM_CLASSES = 2
NUM_FEATURES = 3

Y_FEATURE_NODES = True
F_FEATURE_NODES = True
Q_FEATURE_NODES = False

# %% PARAMETERS

TORCH_SETTINGS = TorchSettings(
    device="cpu",
    dtype="float64",
)

BATCH_SIZE = 100
LEARNING_RATE = 0.1
REGULARIZATION = 0.001
NUM_EPOCHS = 1

mlflow.log_param("TORCH_SETTINGS", TORCH_SETTINGS)
mlflow.log_param("BATCH_SIZE", BATCH_SIZE)
mlflow.log_param("LEARNING_RATE", LEARNING_RATE)
mlflow.log_param("REGULARIZATION", REGULARIZATION)
mlflow.log_param("NUM_EPOCHS", NUM_EPOCHS)


# %% Load data

num_classes = 10
height, width = 28, 28
patch_height, patch_width = 4, 4

transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.flatten()),
    ],
)

mnist = torchvision.datasets.MNIST(
    "./experiments/mnist", train=True, download=True, transform=transforms
)

mnist_test = torchvision.datasets.MNIST(
    "./experiments/mnist", train=False, download=True, transform=transforms
)

iterations_per_epoch = len(mnist) / BATCH_SIZE
assert int(iterations_per_epoch) == iterations_per_epoch, (
    "len(mnist) / BATCH_SIZE should be an integer"
)
iterations_per_epoch = int(iterations_per_epoch)

# %% DELETE ME

mnist = Subset(mnist, range(1000))

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

builder = BayesianNetworkBuilder()

# Create root node
Q = Node.random((NUM_CLASSES), torch_settings=TORCH_SETTINGS, name="Q")
builder.add_node(Q)

Fs = {}
Ys = {}

for ify in range(int(height / patch_height)):
    for ifx in range(int(width / patch_width)):
        F = Node.random((NUM_CLASSES, NUM_FEATURES), TORCH_SETTINGS, name=f"F_{ify}x{ifx}")
        Fs[(ify, ifx)] = F
        builder.add_node(F, parents=Q)

        for iy in range(patch_height):
            for ix in range(patch_width):
                index_y = ify * patch_height + iy
                index_x = ifx * patch_width + ix

                Y = Node.random((NUM_FEATURES, 2), TORCH_SETTINGS, name=f"Y_{index_y}x{index_x}")
                Ys[(index_y, index_x)] = Y
                builder.add_node(Y, parents=F)

Ys = {i: Ys[i] for i in sorted(Ys)}

# Create network
network = builder.build()

mlflow.log_param("DoF", network.degrees_of_freedom)

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
            observed_nodes=list(Ys.values()),
            num_observations=num_observations,
        )

    return inference_machine_factory


feature_nodes = []
if Y_FEATURE_NODES:
    feature_nodes += list(Ys.values())
if F_FEATURE_NODES:
    feature_nodes += list(Fs.values())
if Q_FEATURE_NODES:
    feature_nodes += [Q]

logistic_regression_evaluator_settings = LogisticRegressionEvaluatorSettings(
    should_evaluate=lambda epoch, iteration: (
        iteration == 0
        or (iteration == int(iterations_per_epoch / 2))
        or (epoch == (NUM_EPOCHS - 1) and (iteration == iterations_per_epoch - 1))
    ),
    epochs=100,
    learning_rate=0.02,
    feature_nodes=feature_nodes,
    num_classes=num_classes,
    torch_settings=TORCH_SETTINGS,
    train_batch_size=64,
    test_batch_size=1000,
)

batch_size = 1000
evaluator = MLflowLogisticRegressionEvaluator(
    inference_machine_factory=create_inference_machine_factory(
        batch_size,
    ),
    transform=transform,
    mnist_train_loader=DataLoader(mnist, batch_size=batch_size, shuffle=True),
    mnist_test_loader=DataLoader(mnist_test, batch_size=batch_size, shuffle=False),
    settings=logistic_regression_evaluator_settings,
    iterations_per_epoch=iterations_per_epoch,
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

# %% Plot log_likelihood and accuracy

train_iterations = [log.epoch * iterations_per_epoch + log.iteration for log in logger.logs]
train_values = [log.ll for log in logger.logs]
eval_iterations = [
    epoch * iterations_per_epoch + iteration for epoch, iteration in evaluator.accuracies.keys()
]
eval_values = list(evaluator.accuracies.values())

figure, ax1 = plt.subplots()
ax1.set_xlabel("Iterations")
ax1.set_ylabel("Average log-likelihood", color="blue")
ax1.plot(train_iterations, train_values, color="blue")
ax1.tick_params(axis="y", labelcolor="blue")
ax2 = ax1.twinx()
ax2.set_ylabel("Accuracy", color="red")
ax2.plot(eval_iterations, eval_values, color="red")
ax2.tick_params(axis="y", labelcolor="red")
ax2.set_ylim(80, 100)

mlflow.log_figure(figure, "performance_iterations.png")

# %%

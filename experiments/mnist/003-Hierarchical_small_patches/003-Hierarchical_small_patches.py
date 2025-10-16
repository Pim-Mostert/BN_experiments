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

torch.set_printoptions(sci_mode=False)

# %% tags=["parameters"]

NUM_CLASSES = 10
NUM_FEATURES = 5

# %% PARAMETERS

TORCH_SETTINGS = TorchSettings(
    device="cpu",
    dtype="float64",
)

BATCH_SIZE = 1000
LEARNING_RATE = 0.1
REGULARIZATION = 0.001
NUM_EPOCHS = 2

# %% Load data

height, width = 8, 4

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

# Create feature nodes
F1 = Node.random((NUM_CLASSES, NUM_FEATURES), torch_settings=TORCH_SETTINGS, name="F1")
F2 = Node.random((NUM_CLASSES, NUM_FEATURES), torch_settings=TORCH_SETTINGS, name="F2")

# Create leave nodes
Y1s = [
    Node.random((NUM_FEATURES, 2), torch_settings=TORCH_SETTINGS, name=f"Y_{iy}x{ix}")
    for iy in range(int(height / 2))
    for ix in range(width)
]

Y2s = [
    Node.random((NUM_FEATURES, 2), torch_settings=TORCH_SETTINGS, name=f"Y_{iy}x{ix}")
    for iy in range(int(height / 2), height)
    for ix in range(width)
]

Ys = Y1s + Y2s

# Create network
network = (
    BayesianNetworkBuilder()
    .add_node(Q)
    .add_node(F1, parents=Q)
    .add_node(F2, parents=Q)
    .add_nodes(Y1s, parents=F1)
    .add_nodes(Y2s, parents=F2)
    .build()
)

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

W1 = (
    torch.stack([y.cpt.cpu()[:, 1] for y in Y1s])
    .reshape(width, int(height / 2), NUM_FEATURES)
    .permute([2, 0, 1])
)
W2 = (
    torch.stack([y.cpt.cpu()[:, 1] for y in Y2s])
    .reshape(width, int(height / 2), NUM_FEATURES)
    .permute([2, 0, 1])
)

figure = plt.figure()
figure.suptitle("W1")

for i, w in enumerate(W1):
    plt.subplot(int(NUM_FEATURES**0.5) + 1, int(NUM_FEATURES**0.5) + 1, i + 1)
    plt.imshow(w.reshape(int(height / 2), width))
    plt.colorbar()
    plt.clim(0, 1)

mlflow.log_figure(figure, "W1.png")

figure = plt.figure()
figure.suptitle("W2")

for i, w in enumerate(W2):
    plt.subplot(int(NUM_FEATURES**0.5) + 1, int(NUM_FEATURES**0.5) + 1, i + 1)
    plt.imshow(w.reshape(int(height / 2), width))
    plt.colorbar()
    plt.clim(0, 1)

mlflow.log_figure(figure, "W2.png")

# Plot log_likelihood

train_iterations = [log.epoch * iterations_per_epoch + log.iteration for log in logger.logs]
train_values = [log.ll for log in logger.logs]
eval_iterations = [
    epoch * iterations_per_epoch + iteration
    for epoch, iteration in evaluator.log_likelihoods.keys()
]
eval_values = list(evaluator.log_likelihoods.values())

figure = plt.figure()
plt.plot(train_iterations, train_values, label="Train")
plt.plot(eval_iterations, eval_values, label="Eval")
plt.xlabel("Iterations")
plt.ylabel("Average log-likelihood")
plt.legend()

mlflow.log_figure(figure, "avg_ll_iterations.png")

# %%


def plot(w):
    plt.imshow(w)
    plt.clim(0, 1)
    plt.axis("off")


# %%

figure = plt.figure()

plt.subplot(1, 2, 1)
plot(F1.cpt)
plt.title("F1.cpt")
plt.ylabel("Q")
plt.xlabel("Ys1")

plt.subplot(1, 2, 2)
plot(F2.cpt)
plt.title("F2.cpt")
plt.ylabel("Q")
plt.xlabel("Ys1")

mlflow.log_figure(figure, "F-cpts.png")

# %%

# Q Templates
p1 = F1.cpt @ torch.stack([y.cpt.cpu()[:, 1] for y in Y1s]).T
p2 = F2.cpt @ torch.stack([y.cpt.cpu()[:, 1] for y in Y2s]).T

p = torch.concat([p1, p2], dim=1)

figure = plt.figure()
plt.title("Q - Templates")
plt.axis("off")
for i in range(NUM_CLASSES):
    plt.subplot(int(NUM_CLASSES**0.5) + 1, int(NUM_CLASSES**0.5) + 1, i + 1)
    plot(p[i].reshape(height, width))

mlflow.log_figure(figure, "F-templates.png")

# F Templates

p1 = torch.stack([y.cpt.cpu()[:, 1] for y in Y1s]).T
p2 = torch.stack([y.cpt.cpu()[:, 1] for y in Y2s]).T

p = torch.concat([p1, p2], dim=1)

figure = plt.figure()
plt.title("F - Templates")
plt.axis("off")
for i in range(NUM_FEATURES):
    plt.subplot(int(NUM_FEATURES**0.5) + 1, int(NUM_FEATURES**0.5) + 1, i + 1)
    plot(p[i].reshape(height, width))

mlflow.log_figure(figure, "Q-templates.png")

# %%

X = torch.stack([mnist[i][0] for i in range(8, 16)])
evidence = transform(X)

inference_machine = create_inference_machine_factory(len(X))(network)
inference_machine.enter_evidence(evidence)
ll = inference_machine.log_likelihood()

[pQ, pF1, pF2] = inference_machine.infer_single_nodes([Q, F1, F2])

figure = plt.figure(figsize=(6, 10))
for i, (x, pq, pf1, pf2) in enumerate(zip(X, pQ, pF1, pF2)):
    # Sample
    ax1 = plt.subplot(8, 3, i * 3 + 1)
    plot(x.reshape(height, width))

    # From Q
    p1 = pq @ F1.cpt @ torch.stack([y.cpt.cpu()[:, 1] for y in Y1s]).T
    p2 = pq @ F2.cpt @ torch.stack([y.cpt.cpu()[:, 1] for y in Y2s]).T

    p = torch.concat([p1, p2], dim=0)
    ax2 = plt.subplot(8, 3, i * 3 + 2)
    plot(p.reshape(height, width))

    # From F
    p1 = pf1 @ torch.stack([y.cpt.cpu()[:, 1] for y in Y1s]).T
    p2 = pf2 @ torch.stack([y.cpt.cpu()[:, 1] for y in Y2s]).T

    p = torch.concat([p1, p2], dim=0)
    ax3 = plt.subplot(8, 3, i * 3 + 3)
    plot(p.reshape(height, width))

    if i == 0:
        ax1.set_title("Sample")
        ax2.set_title("From Q")
        ax3.set_title("From F")

mlflow.log_figure(figure, "Reconstructions.png")

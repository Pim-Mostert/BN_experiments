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

NUM_CLASSES = 50
NUM_FEATURES = 50

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

builder = BayesianNetworkBuilder()

# Create root node
Q = Node.random((NUM_CLASSES), torch_settings=TORCH_SETTINGS, name="Q")
builder.add_node(Q)

F1 = Node.random((NUM_CLASSES, NUM_FEATURES), torch_settings=TORCH_SETTINGS, name="F1")
F2 = Node.random((NUM_CLASSES, NUM_FEATURES), torch_settings=TORCH_SETTINGS, name="F2")
F3 = Node.random((NUM_CLASSES, NUM_FEATURES), torch_settings=TORCH_SETTINGS, name="F3")
builder.add_nodes([F1, F2, F3], parents=Q)

Ys = []

for iy in range(height):
    for ix in range(width):
        if iy >= 6:
            Y = Node.random((NUM_FEATURES, 2), TORCH_SETTINGS, name=f"Y_{iy}x{ix}")
            builder.add_node(Y, parents=[F3])
            Ys.append(Y)
            continue

        if iy >= 4:
            Y = Node.random((NUM_FEATURES, NUM_FEATURES, 2), TORCH_SETTINGS, name=f"Y_{iy}x{ix}")
            builder.add_node(Y, parents=[F2, F3])
            Ys.append(Y)
            continue

        if iy >= 2:
            Y = Node.random((NUM_FEATURES, NUM_FEATURES, 2), TORCH_SETTINGS, name=f"Y_{iy}x{ix}")
            builder.add_node(Y, parents=[F1, F2])
            Ys.append(Y)
            continue

        Y = Node.random((NUM_FEATURES, 2), TORCH_SETTINGS, name=f"Y_{iy}x{ix}")
        builder.add_node(Y, parents=[F1])
        Ys.append(Y)


# Create network
network = builder.build()

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
                allow_loops=True,
                num_iterations=10,
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

W1 = torch.stack([y.cpt.cpu()[:, 1] for y in Ys[0:8]])
W2 = torch.stack([y.cpt.cpu()[:, :, 1] for y in Ys[8:16]])
W3 = torch.stack([y.cpt.cpu()[:, :, 1] for y in Ys[16:24]])
W4 = torch.stack([y.cpt.cpu()[:, 1] for y in Ys[24:32]])


plot_width = NUM_FEATURES * 2
plot_height = NUM_FEATURES + 2


def index(iy, ix):
    return iy * plot_width + ix + 1


figure = plt.figure(figsize=(8, 3))

for iF1 in range(NUM_FEATURES):
    plt.subplot(plot_height, plot_width, index(0, iF1))
    plt.imshow(W1[:, iF1].reshape(2, 4))
    plt.axis("off")
    plt.clim(0, 1)

for iF1 in range(NUM_FEATURES):
    for iF2 in range(NUM_FEATURES):
        plt.subplot(plot_height, plot_width, index(iF2 + 1, iF1))
        plt.imshow(W2[:, iF1, iF2].reshape(2, 4))
        plt.axis("off")
        plt.clim(0, 1)

for iF2 in range(NUM_FEATURES):
    for iF3 in range(NUM_FEATURES):
        plt.subplot(plot_height, plot_width, index(iF2 + 1, iF3 + NUM_FEATURES))
        plt.imshow(W3[:, iF2, iF3].reshape(2, 4))
        plt.axis("off")
        plt.clim(0, 1)

for iF3 in range(NUM_FEATURES):
    plt.subplot(plot_height, plot_width, index(NUM_FEATURES + 1, NUM_FEATURES + iF3))
    plt.imshow(W4[:, iF3].reshape(2, 4))
    plt.axis("off")
    plt.clim(0, 1)

# %% Plot log_likelihood

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

plt.subplot(1, 3, 1)
plot(F1.cpt)
plt.ylabel("Q")
plt.xlabel("F1")

plt.subplot(1, 3, 2)
plot(F2.cpt)
plt.ylabel("Q")
plt.xlabel("F2")

plt.subplot(1, 3, 3)
plot(F3.cpt)
plt.ylabel("Q")
plt.xlabel("F3")


# %%

wY1 = torch.stack([y.cpt.cpu()[:, 1] for y in Ys[0:8]])
wY2 = torch.stack([y.cpt.cpu()[:, :, 1] for y in Ys[8:16]])
wY3 = torch.stack([y.cpt.cpu()[:, :, 1] for y in Ys[16:24]])
wY4 = torch.stack([y.cpt.cpu()[:, 1] for y in Ys[24:32]])

# Q Templates
p1 = F1.cpt @ wY1.T
p2 = torch.einsum("cu, cv, yuv->cy", F1.cpt, F2.cpt, wY2)
p3 = torch.einsum("cu, cv, yuv->cy", F2.cpt, F3.cpt, wY3)
p4 = F3.cpt @ wY4.T

p = torch.concat([p1, p2, p3, p4], dim=1)

figure = plt.figure()
plt.title("Q - Templates")
plt.axis("off")
for iQ in range(NUM_CLASSES):
    plt.subplot(int(NUM_CLASSES**0.5) + 1, int(NUM_CLASSES**0.5) + 1, iQ + 1)
    plot(p[iQ].reshape(height, width))


# %%

wY1 = torch.stack([y.cpt.cpu()[:, 1] for y in Ys[0:8]])
wY2 = torch.stack([y.cpt.cpu()[:, :, 1] for y in Ys[8:16]])
wY3 = torch.stack([y.cpt.cpu()[:, :, 1] for y in Ys[16:24]])
wY4 = torch.stack([y.cpt.cpu()[:, 1] for y in Ys[24:32]])

X = torch.stack([mnist[i][0] for i in range(8, 16)])
evidence = transform(X)

inference_machine = create_inference_machine_factory(len(X))(network)
inference_machine.enter_evidence(evidence)
ll = inference_machine.log_likelihood()

[pQ, pF1, pF2, pF3] = inference_machine.infer_single_nodes([Q, F1, F2, F3])

figure = plt.figure(figsize=(6, 10))
for i, (x, pq, pf1, pf2, pf3) in enumerate(zip(X, pQ, pF1, pF2, pF3)):
    # Sample
    ax1 = plt.subplot(8, 3, i * 3 + 1)
    plot(x.reshape(height, width))

    # From Q
    p1 = torch.einsum("c, cu, yu->y", pq, F1.cpt, wY1)
    p2 = torch.einsum("c, cu, cv, yuv->y", pq, F1.cpt, F2.cpt, wY2)
    p3 = torch.einsum("c, cu, cv, yuv->y", pq, F2.cpt, F3.cpt, wY3)
    p4 = torch.einsum("c, cv, yv->y", pq, F3.cpt, wY4)

    p = torch.concat([p1, p2, p3, p4], dim=0)

    ax2 = plt.subplot(8, 3, i * 3 + 2)
    plot(p.reshape(height, width))

    # From F
    p1 = torch.einsum("u, yu->y", pf1, wY1)
    p2 = torch.einsum("u, v, yuv->y", pf1, pf2, wY2)
    p3 = torch.einsum("u, v, yuv->y", pf2, pf3, wY3)
    p4 = torch.einsum("v, yv->y", pf3, wY4)

    p = torch.concat([p1, p2, p3, p4], dim=0)

    ax3 = plt.subplot(8, 3, i * 3 + 3)
    plot(p.reshape(height, width))

    if i == 0:
        ax1.set_title("Sample")
        ax2.set_title("From Q")
        ax3.set_title("From F")

# %%

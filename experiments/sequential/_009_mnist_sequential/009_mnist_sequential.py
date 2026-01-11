# %% Imports

from datetime import datetime
import logging

import matplotlib.pyplot as plt
import mlflow
import torch
from bayesian_network.inference_machines.spa_v3.spa_inference_machine import (
    SpaInferenceMachine,
    SpaInferenceMachineSettings,
)
from dynamic_bayesian_network.builder import DynamicBayesianNetworkBuilder
from dynamic_bayesian_network.dynamic_bayesian_network import DynamicBayesianNetwork, Node
from dynamic_bayesian_network.inference import InferenceMachine, SequentialEvidence
from dynamic_bayesian_network.optimizer import EmOptimizer
from pim_common.extensions import extension
from pim_common.torch_settings import TorchSettings
from torch.utils.data import DataLoader, Subset
from experiments.sequential._009_mnist_sequential.data import MNISTSequenceDataset

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s][%(asctime)s][%(module)s.%(funcName)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
torch.set_printoptions(sci_mode=False)

# %% tags=["parameters"]

BATCH_SIZE = 1000
LEARNING_RATE = 0.1
REGULARIZATION = 0.001
NUM_EPOCHS = 10
NUM_SUBSET = 5000


@extension(to=torch.Tensor)
def to(x: torch.Tensor, torch_settings: TorchSettings):
    return x.to(dtype=torch_settings.dtype, device=torch_settings.device)


# %% PARAMETERS

TORCH_SETTINGS = TorchSettings(
    device="cpu",
    dtype="float64",
)

mlflow.log_param("TORCH_SETTINGS", TORCH_SETTINGS)
mlflow.log_param("BATCH_SIZE", BATCH_SIZE)
mlflow.log_param("LEARNING_RATE", LEARNING_RATE)
mlflow.log_param("REGULARIZATION", REGULARIZATION)
mlflow.log_param("NUM_EPOCHS", NUM_EPOCHS)


# %%

dataset = MNISTSequenceDataset(root="./experiments/sequential")
subset = Subset(dataset, indices=range(NUM_SUBSET))
data_loader = DataLoader(subset, BATCH_SIZE, shuffle=True)

height, width = 28, 28

iterations_per_epoch = len(subset) / BATCH_SIZE
assert int(iterations_per_epoch) == iterations_per_epoch, (
    "len(subset) / BATCH_SIZE should be an integer"
)
iterations_per_epoch = int(iterations_per_epoch)

# %% Plot a sequence


plt.figure()
for e, x in enumerate(iter(dataset[0])):
    plt.subplot(4, 3, e + 1)
    plt.imshow(x.reshape(28, 28))
    plt.clim(0, 1)

plt.show()

# %% Define network


num_classes = 10
num_features = 2

Q = Node.random(
    (num_classes, num_classes),
    is_sequential=True,
    is_observed=False,
    device=TORCH_SETTINGS.device,
    dtype=TORCH_SETTINGS.dtype,
    prior_size=(num_classes),
    name="Q",
)
builder = DynamicBayesianNetworkBuilder().add_node(Q, sequential_parents=[Q])

for iy in range(height):
    for ix in range(width):
        node = Node.random(
            (num_classes, 2),
            is_sequential=False,
            is_observed=True,
            device=TORCH_SETTINGS.device,
            dtype=TORCH_SETTINGS.dtype,
            name=f"Y_{iy}x{ix}",
        )
        builder.add_node(node, parents=Q)

dbn = builder.build()

# %% Inference machine


def inference_machine_factory(
    dbn: DynamicBayesianNetwork,
    sev: SequentialEvidence,
):
    return InferenceMachine(
        dbn,
        lambda network, observed_nodes: SpaInferenceMachine(
            SpaInferenceMachineSettings(
                torch_settings=TORCH_SETTINGS,
                average_log_likelihood=True,
                allow_loops=False,
                callback=lambda event: print(
                    f"Finished iteration {event.iteration}/{event.num_iterations}. It took {event.duration}."
                ),
            ),
            bayesian_network=network,
            observed_nodes=observed_nodes,
            num_observations=sev.num_observations,
        ),
    )


# %% Sequential evidence


@extension(to=torch.Tensor)
def to_sequential_evidence(batch: torch.Tensor, nodes: list[Node]) -> SequentialEvidence:
    if not len(batch.shape) == 3:
        raise ValueError(
            f"Batch's shape should have dimensions [observations, sequence_length, features], but is: {batch.shape}"
        )

    if not len(nodes) == batch.shape[2]:
        raise ValueError(
            f"Number of nodes {len(nodes)} does not match batch's last dimension: {batch.shape[-1]}"
        )

    return SequentialEvidence(
        {
            node: torch.stack(
                [
                    1 - x,
                    x,
                ],
                dim=2,
            )
            for node, x in zip(nodes, batch.permute([2, 0, 1]))
        }
    )


# %% Fit network

ll = []

optimizer = EmOptimizer(
    dbn,
    lr=LEARNING_RATE,
    regularization=REGULARIZATION,
)

for e in range(NUM_EPOCHS):
    for i, batch in enumerate(data_loader):
        t_start = datetime.now()

        sev = batch | to(TORCH_SETTINGS) | to_sequential_evidence(dbn.observed_nodes)

        # Construct inference machine and enter evidence
        im = inference_machine_factory(dbn, sev)
        im.enter_evidence(sev)
        log_likelihood = im.log_likelihood()

        mlflow.log_metric("ll", log_likelihood, step=iterations_per_epoch * NUM_EPOCHS + i)
        ll.append(log_likelihood)

        # Optimize
        optimizer.e_step(im)
        optimizer.m_step()

        # Feedback
        duration = datetime.now() - t_start
        print(
            f"Finished epoch {e + 1}/{NUM_EPOCHS}, batch {i + 1}/{len(data_loader)}. It took: {duration}."
        )


# %% Plot log_likelihood
train_iterations = range(len(ll))
train_values = ll

plt.figure()
plt.plot(train_iterations, train_values, label="Train")
plt.xlabel("Iterations")
plt.ylabel("Average log-likelihood")
plt.legend()

# %% Plot

w = torch.stack([y.cpt.cpu() for y in dbn.observed_nodes])

fig = plt.figure()
for i in range(0, w.shape[1]):
    plt.subplot(3, 4, i + 1)
    plt.imshow(w[:, i, 1].reshape(28, 28))
    plt.colorbar()
    plt.clim(0, 1)

mlflow.log_figure(fig, "weights.png")

Q_prior = dbn.get_node("Q").prior.cpu()
fig = plt.figure()
plt.imshow(Q_prior.unsqueeze(dim=1).T)
plt.colorbar()
plt.clim(0, 1)
plt.gca().xaxis.tick_top()
plt.gca().xaxis.set_label_position("top")
plt.xlabel("$Q_{t0}$")

mlflow.log_figure(fig, "Q_cpt.png")

Q_cpt = dbn.get_node("Q").cpt.cpu()
fig = plt.figure()
plt.imshow(Q_cpt)
plt.colorbar()
plt.clim(0, 1)
plt.gca().xaxis.tick_top()
plt.gca().xaxis.set_label_position("top")
plt.xlabel("$Q_t$")
plt.ylabel("$Q_{t-1}$")

mlflow.log_figure(fig, "Q_cpt.png")

# %%

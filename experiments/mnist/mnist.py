# %% Imports
import matplotlib.pyplot as plt
import torch


from bayesian_network.inference_machines.spa_v3.spa_inference_machine import (
    SpaInferenceMachine,
    SpaInferenceMachineSettings,
)


from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.optimizers.common import (
    OptimizerLogger,
)
from bayesian_network.optimizers.em_batch_optimizer import (
    EmBatchOptimizer,
    EmBatchOptimizerSettings,
)

import logging

from experiments.mnist.common import MnistEvidenceBatches

logging.basicConfig(level=logging.INFO)

# %% tags=["parameters"]

TORCH_SETTINGS = TorchSettings(
    device="cpu",
    dtype="float64",
)

BATCH_SIZE = 50

EM_BATCH_OPTIMIZER_SETTINGS = EmBatchOptimizerSettings(
    num_iterations=400,
    learning_rate=0.01,
)

GAMMA = 0.001

# %% Load data

batches = MnistEvidenceBatches(
    torch_settings=TORCH_SETTINGS,
    gamma=GAMMA,
    batch_size=BATCH_SIZE,
)

height, width = batches.mnist_dimensions

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
mu = (
    torch.rand(
        (height, width, num_classes),
        device=TORCH_SETTINGS.device,
        dtype=TORCH_SETTINGS.dtype,
    )
    * 0.2
    + 0.4
)
mu = torch.stack([1 - mu, mu], dim=3)
Ys = [Node(mu[iy, ix], name=f"Y_{iy}x{ix}") for iy in range(height) for ix in range(width)]

nodes = [Q] + Ys
parents = {Y: [Q] for Y in Ys}
parents[Q] = []

network = BayesianNetwork(nodes, parents)


# %% Fit network
logger = OptimizerLogger()

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
for i in range(0, 10):
    plt.subplot(4, 3, i + 1)
    plt.imshow(w[:, i, 1].reshape(28, 28))
    plt.colorbar()
    plt.clim(0, 1)

plt.figure()
plt.plot(logger.get_log_likelihood())

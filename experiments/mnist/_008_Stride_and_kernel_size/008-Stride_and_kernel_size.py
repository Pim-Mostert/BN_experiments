# %% Imports

import logging
from typing import List

import matplotlib.pyplot as plt
import mlflow
import torch
import torchvision
from bayesian_network.samplers.torch_sampler import TorchBayesianNetworkSampler
from bayesian_network.bayesian_network import Node
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
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from experiments.mnist._008_Stride_and_kernel_size.network import Network3LBuilder
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

# NUM_F0 = 2
# NUM_F1 = 5
# NUM_F2 = 5


# %% PARAMETERS

TORCH_SETTINGS = TorchSettings(
    device="cpu",
    dtype="float64",
)

# TORCH_SETTINGS = TorchSettings(
#     device="mps",
#     dtype="float32",
# )

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

height, width = 22, 22

mnist_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:, 3:25, 3:25]),
        transforms.Lambda(lambda x: x.flatten()),
    ],
)

mnist = torchvision.datasets.MNIST(
    "./experiments/mnist", train=True, download=True, transform=mnist_transform
)
mnist = Subset(mnist, range(5000))

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
numF0 = 50
numF1 = 3
numF2 = 5

network, F0, F1s, F2s, Ys = Network3LBuilder(TORCH_SETTINGS).build(
    numF0=numF0,
    numF1=numF1,
    numF2=numF2,
)

logging.info("Network degrees of freedom: %s", network.degrees_of_freedom)
mlflow.log_metric("DoF", network.degrees_of_freedom)

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


# %% Run experiment

em_optimizer = EmBatchOptimizer(
    bayesian_network=network,
    inference_machine_factory=create_inference_machine_factory(BATCH_SIZE),
    settings=em_batch_optimizer_settings,
    logger=logger,
)
em_optimizer.optimize(evidence_loader)

# %% Plot log_likelihood

train_iterations = [log.epoch * iterations_per_epoch + log.iteration for log in logger.logs]
train_values = [log.ll for log in logger.logs]

figure = plt.figure()
plt.plot(train_iterations, train_values, label="Train")
plt.xlabel("Iterations")
plt.ylabel("Average log-likelihood")
plt.legend()

# %% Plots


def plot(w):
    plt.imshow(w)
    plt.clim(0, 1)
    plt.axis("off")


def get_reconstruction(observed_nodes: List[Node], queries: Evidence):
    im = SpaInferenceMachine(
        settings=SpaInferenceMachineSettings(
            TORCH_SETTINGS,
            average_log_likelihood=True,
            allow_loops=True,
            num_iterations=10,
        ),
        bayesian_network=network,
        num_observations=queries.num_observations,
        observed_nodes=observed_nodes,
    )

    im.enter_evidence(queries)
    y_hat = torch.stack(im.infer_single_nodes(Ys))

    return y_hat[:, :, 1].T


# %% Root template

queries = Evidence([F0.cpt[None, :]], torch_settings=TORCH_SETTINGS)
template = get_reconstruction([F0], queries).squeeze()

figure = plt.figure()
figure.suptitle("Grand mean")
plot(template.reshape(height, width))


mlflow.log_figure(figure, "Templates_root.png")

# %% F0 templates

queries = Evidence([torch.eye(numF0)], torch_settings=TORCH_SETTINGS)
templates = get_reconstruction([F0], queries)

plt.figure()
plt.suptitle("F0 templates")
for i in range(numF0):
    plt.subplot(int(numF0**0.5) + 1, int(numF0**0.5) + 1, i + 1)
    plot(templates[i].reshape(height, width))

mlflow.log_figure(figure, "Templates_F0.png")

# %% F1 templates

iF1 = 5

queries = Evidence([torch.eye(numF1)], torch_settings=TORCH_SETTINGS)
templates = get_reconstruction([F1s[iF1]], queries)

for i in range(0, numF1):
    plt.subplot(1, 5, i + 1)
    plot(templates[i].reshape(height, width))

mlflow.log_figure(figure, "Templates_F1.png")

# %% F2 templates

iF2 = 55

queries = Evidence([torch.eye(numF2)], torch_settings=TORCH_SETTINGS)
templates = get_reconstruction([F2s[iF2]], queries)

for i in range(0, numF2):
    plt.subplot(1, 5, i + 1)
    plot(templates[i].reshape(height, width))

mlflow.log_figure(figure, "Templates_F2.png")


# %% Sample reconstruction

### P(Q | Y) and P(Fs | Y)
index_samples = range(0, 32)

X = torch.stack([mnist[i][0] for i in index_samples])
evidence = transform(X)

inference_machine = create_inference_machine_factory(len(X))(network)
inference_machine.enter_evidence(evidence)

pF0 = inference_machine.infer_single_nodes([F0])
pF1s = inference_machine.infer_single_nodes(F1s)
pF2s = inference_machine.infer_single_nodes(F2s)

### P(Y | F0)
y_hat_F0 = get_reconstruction([F0], Evidence(pF0, TORCH_SETTINGS))

### P(Y | F1)
y_hat_F1 = get_reconstruction(F1s, Evidence(pF1s, TORCH_SETTINGS))

### P(Y | F1)
y_hat_F2 = get_reconstruction(F2s, Evidence(pF2s, TORCH_SETTINGS))

### Plot
figure = plt.figure(figsize=(6, 50))
for i, x in enumerate(X):
    pfs = [e[i][None, :] for e in pF1s]

    # Sample
    ax1 = plt.subplot(len(X), 4, i * 4 + 1)
    plot(x.reshape(height, width))

    # From F0
    ax2 = plt.subplot(len(X), 4, i * 4 + 2)
    plot(y_hat_F0[i].reshape(height, width))

    # From F1
    ax3 = plt.subplot(len(X), 4, i * 4 + 3)
    plot(y_hat_F1[i].reshape(height, width))

    # From F2
    ax3 = plt.subplot(len(X), 4, i * 4 + 4)
    plot(y_hat_F2[i].reshape(height, width))

    if i == 0:
        ax1.set_title("Sample")
        ax2.set_title("From F0")
        ax3.set_title("From F1")
        ax3.set_title("From F2")

mlflow.log_figure(figure, "Reconstructions.png")

# %% Generate samples

sampler = TorchBayesianNetworkSampler(network, TORCH_SETTINGS)

data = sampler.sample(25, Ys)

figure = plt.figure()
for i, x in enumerate(data):
    plt.subplot(5, 5, i + 1)
    plot(x.reshape(height, width))

mlflow.log_figure(figure, "Generated_samples.png")

# %%

# %%

# %% Imports

import logging
from typing import List


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
from bayesian_network.samplers.torch_sampler import TorchBayesianNetworkSampler
from bayesian_network.optimizers.em_batch_optimizer import (
    EmBatchOptimizer,
    EmBatchOptimizerSettings,
)
from torchvision import transforms

from experiments.mnist.logistic_regression_evaluator import (
    LogisticRegressionEvaluator,
    LogisticRegressionEvaluatorSettings,
)
from experiments.mnist.common import (
    MLflowOptimizerLogger,
)

IT SEEMS IM STILL GETTING EXCESSIVE READS. THE CHUNK CACHING DOESNT WORK?
ALSO, TRAIN LOSS APPEARED TO INCREASE WITH EVERY EPOCH. THIS SHOULD BE FIXED NOW
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s][%(asctime)s][%(module)s.%(funcName)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
torch.set_printoptions(sci_mode=False)

# %% tags=["parameters"]

NUM_CLASSES = 5
NUM_FEATURES = 10


# %% PARAMETERS

TORCH_SETTINGS = TorchSettings(
    device="cpu",
    dtype="float64",
)

# TORCH_SETTINGS = TorchSettings(
#     device="mps",
#     dtype="float32",
# )

BATCH_SIZE = 200
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
mnist = Subset(mnist, indices=range(10000))

mnist_test = torchvision.datasets.MNIST(
    "./experiments/mnist", train=False, download=True, transform=transforms
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
            ),
            bayesian_network=network,
            observed_nodes=list(Ys.values()),
            num_observations=num_observations,
        )

    return inference_machine_factory


# evaluator_batch_size = 2000
# batch_evaluator = MLflowBatchEvaluator(
#     iterations_per_epoch=iterations_per_epoch,
#     inference_machine_factory=create_inference_machine_factory(evaluator_batch_size),
#     evidence_loader=EvidenceLoader(
#         DataLoader(
#             dataset=mnist,
#             batch_size=evaluator_batch_size,
#         ),
#         transform=transform,
#     ),
#     should_evaluate=lambda epoch, iteration: (
#         (iteration == 0)
#         or (iteration == int(iterations_per_epoch / 2))
#         or (epoch == (NUM_EPOCHS - 1) and (iteration == iterations_per_epoch - 1))
#     ),
# )


logistic_regression_evaluator_settings = LogisticRegressionEvaluatorSettings(
    should_evaluate=lambda epoch, iteration: (
        # (iteration == 0)
        # or (iteration == int(iterations_per_epoch / 2))
        epoch == (NUM_EPOCHS - 1) and (iteration == iterations_per_epoch - 1)
    ),
    epochs=10,
    learning_rate=0.01,
    feature_nodes=list(Fs.values()),
    num_classes=num_classes,
    torch_settings=TORCH_SETTINGS,
    train_batch_size=64,
    test_batch_size=1000,
)

batch_size = 1000
logistic_regression_evaluator = LogisticRegressionEvaluator(
    inference_machine_factory=create_inference_machine_factory(
        batch_size,
    ),
    evidence_loader_factory=lambda dataset: EvidenceLoader(
        data_loader=DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
        ),
        transform=transform,
    ),
    mnist_train=mnist,
    mnist_test=mnist_test,
    settings=logistic_regression_evaluator_settings,
)

# %% Run experiment

em_optimizer = EmBatchOptimizer(
    bayesian_network=network,
    inference_machine_factory=create_inference_machine_factory(BATCH_SIZE),
    settings=em_batch_optimizer_settings,
    logger=logger,
    evaluator=logistic_regression_evaluator,
)
em_optimizer.optimize(evidence_loader)

# %% Plot log_likelihood

train_iterations = [log.epoch * iterations_per_epoch + log.iteration for log in logger.logs]
train_values = [log.ll for log in logger.logs]
# eval_iterations = [
#     epoch * iterations_per_epoch + iteration
#     for epoch, iteration in evaluator.log_likelihoods.keys()
# ]
# eval_values = list(evaluator.log_likelihoods.values())

figure = plt.figure()
plt.plot(train_iterations, train_values, label="Train")
# plt.plot(eval_iterations, eval_values, label="Eval")
plt.xlabel("Iterations")
plt.ylabel("Average log-likelihood")
plt.legend()

mlflow.log_figure(figure, "avg_ll_iterations.png")

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
        ),
        bayesian_network=network,
        num_observations=queries.num_observations,
        observed_nodes=observed_nodes,
    )

    im.enter_evidence(queries)
    y_hat = torch.stack(im.infer_single_nodes(list(Ys.values())))

    return y_hat[:, :, 1].T


# %% Plot F->Y weights

wY = torch.stack([Ys[y].cpt.cpu()[:, 1] for y in Ys])


figure = plt.figure()
figure.suptitle("F->Y weights")
for iF in range(NUM_FEATURES):
    plt.subplot(int(NUM_FEATURES**0.5) + 1, int(NUM_FEATURES**0.5) + 1, iF + 1)
    plot(wY[:, iF].reshape(height, width))


mlflow.log_figure(figure, "F-Y_weights.png")

# %% Root template

queries = Evidence([Q.cpt], torch_settings=TORCH_SETTINGS)
template = get_reconstruction([Q], queries)[0]

figure = plt.figure()
figure.suptitle("Grand mean")
plot(template.reshape(height, width))


mlflow.log_figure(figure, "Templates_root.png")

# %% Q templates

queries = Evidence([torch.eye(NUM_CLASSES)], torch_settings=TORCH_SETTINGS)
templates = get_reconstruction([Q], queries)

plt.figure()
plt.suptitle("Q templates")
for i in range(NUM_CLASSES):
    plt.subplot(int(NUM_CLASSES**0.5) + 1, int(NUM_CLASSES**0.5) + 1, i + 1)
    plot(templates[i].reshape(height, width))

mlflow.log_figure(figure, "Templates_Q.png")

# %% Sample reconstruction

### P(Q | Y) and P(Fs | Y)
index_samples = range(0, 32)

X = torch.stack([mnist[i][0] for i in index_samples])
evidence = transform(X)

inference_machine = create_inference_machine_factory(len(X))(network)
inference_machine.enter_evidence(evidence)

inferred = inference_machine.infer_single_nodes([Q, *Fs.values()])
pQ = inferred[0:1]
pFs = inferred[1:]

### P(Y | Q)
y_hat_Q = get_reconstruction([Q], Evidence(pQ, TORCH_SETTINGS))

### P(Y | F)
y_hat_F = get_reconstruction(list(Fs.values()), Evidence(pFs, TORCH_SETTINGS))

### Plot
figure = plt.figure(figsize=(6, 50))
for i, x in enumerate(X):
    pfs = [e[i][None, :] for e in pFs]

    # Sample
    ax1 = plt.subplot(len(X), 3, i * 3 + 1)
    plot(x.reshape(height, width))

    # From Q
    ax2 = plt.subplot(len(X), 3, i * 3 + 2)
    plot(y_hat_Q[i].reshape(height, width))

    # From F
    ax3 = plt.subplot(len(X), 3, i * 3 + 3)
    plot(y_hat_F[i].reshape(height, width))

    if i == 0:
        ax1.set_title("Sample")
        ax2.set_title("From Q")
        ax3.set_title("From F")

mlflow.log_figure(figure, "Reconstructions.png")

# %% Generate samples

sampler = TorchBayesianNetworkSampler(network, TORCH_SETTINGS)

data = sampler.sample(25, list(Ys.values()))

figure = plt.figure()
for i, x in enumerate(data):
    plt.subplot(5, 5, i + 1)
    plot(x.reshape(height, width))

mlflow.log_figure(figure, "Generated_samples.png")

# %%

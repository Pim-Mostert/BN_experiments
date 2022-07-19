import time

import matplotlib.pyplot as plt
import torch
import torchvision as torchvision
from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.statistics import generate_random_probability_matrix
from bayesian_network.inference_machines.torch_sum_product_algorithm_inference_machine_2 import \
    TorchSumProductAlgorithmInferenceMachine
from bayesian_network.interfaces import IInferenceMachine
from bayesian_network.optimizers.em_optimizer import EmOptimizer
from torchvision.transforms import transforms

num_observations = 200
device = torch.device('cpu')

# Prepare training data set
selected_labels = [0, 1]
num_classes = len(selected_labels)

mnist = torchvision.datasets.MNIST('./mnist', train=True, transform=transforms.ToTensor(), download=True)
selection = [(data, label) for data, label in zip(mnist.train_data, mnist.train_labels) if label in selected_labels] \
    [0:num_observations]
training_data = torch.stack([data for data, label in selection]) \
    .ge(128).double()
true_labels = [int(label) for data, label in selection]

height, width = training_data.shape[1:3]
num_features = height * width

# Morph into evidence structure
evidence = training_data.reshape([num_observations, num_features]).double()

# Train model
num_iterations = 7
num_iterations_sp = 8

pi = torch.zeros(num_iterations, num_classes, dtype=torch.double)
pi[0] = 1/num_classes

mu = torch.zeros((num_iterations, num_features, num_classes, 2), dtype=torch.double)
mu[0, :, :, 0] = torch.rand((num_features, num_classes))
mu[0, :, :, 1] = 1-mu[0, :, :, 0]

# [num_observations, num_features, 2]
e = torch.stack([1-evidence, evidence], dim=2)
for i in range(0, num_iterations-1):
    t_start = time.time()

    # E-step
    a = torch.ones((num_observations, num_features, 2), dtype=torch.double) * 0.5
    b = torch.ones((num_observations, num_features, 2), dtype=torch.double) * 0.5
    c = torch.ones((num_observations, num_features, num_classes), dtype=torch.double) / num_classes
    d = torch.ones((num_observations, num_features, num_classes), dtype=torch.double) / num_classes
    v = pi[i]
    m = mu[i]

    for j in range(num_iterations_sp):
        gamma = (b * e).sum(dim=2, keepdim=True)
        a = e / gamma

        for k in range(num_features):
            dims = list(range(num_features))
            del dims[k]

            d[:, k, :] = c[:, dims, :].prod(axis=1)

        d *= v
        d /= d.sum(dim=2, keepdim=True)

        c = (a[:, :, None, :] * m[None, :, :, :]).sum(dim=3)
        b = (d[:, :, :, None] * m[None, :, :, :]).sum(dim=2)

        phi = (c.prod(dim=1) * v).sum(dim=1, keepdim=True)
        u = c.prod(dim=1)/phi

    # M-step
    pi[i+1] = (u*v).mean(dim=0)
    mu[i+1] = (a[:, :, None, :] * d[:, :, :, None] * m[None, :, :, :]).mean(dim=0)
    mu[i+1] /= mu[i+1].sum(dim=2, keepdim=True)

    duration = time.time() - t_start
    print(f'Finished iteration {i}/{num_iterations} - it took {duration} s')

plt.figure()
for i in range(num_classes):
    plt.subplot(3, 4, i+1); plt.imshow(mu[-1, :, i, 1].reshape(28, 28))

plt.figure()
for i in range(num_classes):
    plt.subplot(3, 4, i+1); plt.plot(range(num_iterations), mu[:, :, i, 1].reshape(num_iterations, -1))

plt.figure()
plt.plot(range(num_iterations), pi.reshape(num_iterations, -1))

pass


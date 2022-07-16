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

num_observations = 1000
device = torch.device('cpu')

# Prepare training data set
selected_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
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
num_iterations = 10

pi = torch.zeros(num_iterations, num_classes, dtype=torch.double)
pi[0] = 0.5

mu = torch.zeros(num_iterations, num_classes, num_features, dtype=torch.double)
mu[0] = torch.rand(num_classes, num_features)*0.2 + 0.4

for i in range(0, num_iterations-1):
    c = torch.log(1-mu[i])
    a = torch.log(mu[i]) - c
    b = c.sum(dim=1)

    # E-step
    likelihood = evidence @ a.T + b
    p = torch.exp(likelihood + torch.exp(pi[i][None, :]))
    p /= p.sum(dim=1, keepdim=True)

    # M-step
    pi[i+1] = p.mean(dim=0)
    mu[i+1] = (p.T @ evidence) / p.sum(dim=0)[:, None]
    mu[i+1] = 0.999999 * mu[i+1] + 0.0000005

plt.figure()
for i in range(num_classes):
    plt.subplot(3, 4, i+1); plt.imshow(mu[-1, i, :].reshape(28, 28))

plt.figure()
for i in range(num_classes):
    plt.subplot(3, 4, i+1); plt.plot(range(num_iterations), mu[:, i, :].reshape(num_iterations, -1))

plt.figure()
plt.plot(range(num_iterations), pi.reshape(num_iterations, -1))

pass


import matplotlib.pyplot as plt
import torch
import torchvision as torchvision
from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.statistics import generate_random_probability_matrix
from bayesian_network.inference_machines.torch_sum_product_algorithm_inference_machine import \
    TorchSumProductAlgorithmInferenceMachine
from bayesian_network.interfaces import IInferenceMachine
from bayesian_network.optimizers.em_optimizer import EmOptimizer
from bayesian_network.common.torch_settings import TorchSettings
from torchvision.transforms import transforms
from torch.nn.functional import one_hot

num_observations = 2000
torch_settings = TorchSettings(torch.device('cpu'), torch.float64)

# Prepare training data set
selected_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
num_classes = len(selected_labels)

mnist = torchvision.datasets.MNIST('./mnist', train=True, transform=transforms.ToTensor(), download=True)
selection = [(data, label) for data, label in zip(mnist.train_data, mnist.train_labels) if label in selected_labels] \
    [0:num_observations]
training_data = torch.stack([data for data, label in selection]) \
    .ge(128).long()
true_labels = [int(label) for data, label in selection]

height, width = training_data.shape[1:3]
num_features = height * width

# Morph into evidence structure
training_data_reshaped = training_data.reshape([num_observations, num_features])

# evidence: List[num_observed_nodes x torch.Tensor[num_observations x num_states]], one-hot encoded
gamma = 0.000001
evidence = [
    node_evidence * (1-gamma) + gamma/2
    for node_evidence 
    in one_hot(training_data_reshaped.T, 2).to(torch_settings.dtype)
]

# Create network
Q = Node(torch.ones((num_classes), device=torch_settings.device, dtype=torch_settings.dtype)/num_classes, name='Q')
mu = torch.rand((height, width, num_classes), device=torch_settings.device, dtype=torch_settings.dtype)*0.2 + 0.4
mu = torch.stack([1-mu, mu], dim=3)
Ys = [
    Node(mu[iy, ix], name=f'Y_{iy}x{ix}')
    for iy in range(height)
    for ix in range(width)
]

nodes = [Q] + Ys
parents = {
    Y: [Q] for Y in Ys
}
parents[Q] = []

network = BayesianNetwork(nodes, parents)

# Train network
num_iterations = 5

def inference_machine_factory(bayesian_network: BayesianNetwork) -> IInferenceMachine:
    return TorchSumProductAlgorithmInferenceMachine(
        bayesian_network=bayesian_network,
        observed_nodes=Ys,
        torch_settings=torch_settings,
        num_iterations=8,
        num_observations=num_observations,
        callback=lambda x, y: None)

em_optimizer = EmOptimizer(network, inference_machine_factory)
em_optimizer.optimize(evidence, num_iterations, lambda ll, iteration, duration:
    print(f'Finished iteration {iteration}/{num_iterations} - ll: {ll} - it took: {duration} s'))

w = torch.stack([y.cpt.cpu() for y in Ys])

plt.figure()
for i in range(len(selected_labels)):
    plt.subplot(4, 3, i+1)
    plt.imshow(w[:, i, 1].reshape(height, width))
    plt.colorbar()
    plt.clim(0, 1)


pass
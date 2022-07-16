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

mnist = torchvision.datasets.MNIST('./mnist', train=True, transform=transforms.ToTensor(), download=True)
selection = [(data, label) for data, label in zip(mnist.train_data, mnist.train_labels) if label in selected_labels] \
    [0:num_observations]
training_data = torch.stack([data for data, label in selection]) \
    .ge(128).int()
true_labels = [int(label) for data, label in selection]

height, width = training_data.shape[1:3]
num_features = height * width

# Morph into evidence structure
evidence = training_data.reshape([num_observations, num_features])

# Create network
Q = Node(generate_random_probability_matrix([len(selected_labels)], device), name='Q')
Y_map = {
    (iy, ix): Node(generate_random_probability_matrix([len(selected_labels), 2], device), name=f'Y_{iy}x{ix}')
    for iy in range(height)
    for ix in range(width)
}
Ys = list(Y_map.values())
nodes = [Q] + Ys
parents = {node: [Q] for node in Ys}
parents[Q] = []

network = BayesianNetwork(nodes, parents)

# Train network
num_iterations = 5

def inference_machine_factory(bayesian_network: BayesianNetwork) -> IInferenceMachine:
    return TorchSumProductAlgorithmInferenceMachine(
        bayesian_network=bayesian_network,
        observed_nodes=Ys,
        device=device,
        num_iterations=8,
        num_observations=num_observations,
        callback=lambda x, y: None)

em_optimizer = EmOptimizer(network, inference_machine_factory)
em_optimizer.optimize(evidence, num_iterations, lambda ll, iteration, duration:
    print(f'Finished iteration {iteration}/{num_iterations} - ll: {ll} - it took: {duration} s'))

w = torch.stack([y.cpt for y in Ys])

plt.figure()
for i in range(len(selected_labels)):
    plt.subplot(1, len(selected_labels), i+1)
    plt.imshow(w[:, i, 1].reshape(height, width))
    plt.colorbar()


pass
from typing import Callable

import torch
import torchvision as torchvision
from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.statistics import generate_random_probability_matrix
from bayesian_network.inference_machines.factor_graph.factor_graph_2 import FactorGraph
from bayesian_network.inference_machines.torch_sum_product_algorithm_inference_machine_2 import \
    TorchSumProductAlgorithmInferenceMachine
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

num_observations = 10
device = torch.device('cpu')

# Prepare training data set
selected_labels = [0, 1]

mnist = torchvision.datasets.MNIST('./mnist', train=True, transform=transforms.ToTensor(), download=True)
data = torch.stack([data for data, label in zip(mnist.train_data, mnist.train_labels) if label in selected_labels]) \
    [0:num_observations].ge(128).int()

height, width = data.shape[1:3]
num_features = height * width

# Morph into evidence structure
evidence = data.reshape([num_observations, num_features])

# Create network
Q = Node(generate_random_probability_matrix([len(selected_labels)], device), name='Q')
Y_map = {
    (iy, ix): Node(generate_random_probability_matrix([2, 2], device), name=f'Y_{iy}x{ix}')
    for iy in range(height)
    for ix in range(width)
}
Ys = list(Y_map.values())
nodes = [Q] + Ys
parents = {node: [Q] for node in Ys}
parents[Q] = []

network = BayesianNetwork(nodes, parents)

# Train network
num_iterations=8

d = torch.zeros((num_iterations, num_features, num_observations, 2), dtype=torch.double)
a1 = torch.zeros((num_iterations, num_features, num_observations, 2), dtype=torch.double)

def callback(factor_graph: FactorGraph, iteration: int):
    d[iteration] = torch.stack([
        variable_node.input_from_observation
        for variable_node
        in [factor_graph.variable_nodes[node] for node in Ys]
    ])
    a1[iteration] = torch.stack([
        factor_node.input_from_local_variable_node
        for factor_node
        in [factor_graph.factor_nodes[node] for node in Ys]
    ])
    print(f'Finished iteration {iteration}/{num_iterations}')

sp_inference_machine = TorchSumProductAlgorithmInferenceMachine(
    bayesian_network=network,
    observed_nodes=Ys,
    device=device,
    num_iterations=num_iterations,
    num_observations=num_observations,
    callback=callback)

sp_inference_machine.enter_evidence(evidence)
ll = sp_inference_machine.log_likelihood()
p = sp_inference_machine.infer_single_nodes([Q])
p_family = sp_inference_machine.infer_nodes_with_parents(Ys)

plt.figure()
plt.plot(range(num_iterations), a1.reshape([num_iterations, -1]))


pass
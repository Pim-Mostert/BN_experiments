import torch
from bayesian_network.bayesian_network import Node, BayesianNetwork
from bayesian_network.common.statistics import generate_random_probability_matrix
from bayesian_network.inference_machines.factor_graph.factor_graph_2 import FactorGraph
from bayesian_network.inference_machines.torch_naive_inference_machine import TorchNaiveInferenceMachine
from bayesian_network.inference_machines.torch_sum_product_algorithm_inference_machine_2 import \
    TorchSumProductAlgorithmInferenceMachine
from bayesian_network.interfaces import IInferenceMachine
from bayesian_network.optimizers.em_optimizer import EmOptimizer
from bayesian_network.samplers.torch_sampler import TorchBayesianNetworkSampler
from matplotlib import pyplot as plt

device = torch.device('cpu')

# True network
Q_true = Node(torch.tensor([0.8, 0.2], device=device, dtype=torch.double), name='Q_true')
Y1_true = Node(torch.tensor([[0, 1], [1, 0]], device=device, dtype=torch.double), name='Y1_true')
Y2_true = Node(torch.tensor([[1, 0], [0, 1]], device=device, dtype=torch.double), name='Y2_true')
nodes_true = [Q_true, Y1_true, Y2_true]
parents_true = {
    Q_true: [],
    Y1_true: [Q_true],
    Y2_true: [Q_true],
}
true_network = BayesianNetwork(nodes_true, parents_true)

# Prepare training data set
num_observations = 1000
sampler = TorchBayesianNetworkSampler(true_network, device=device)
training_data = sampler.sample(num_observations, [Y1_true, Y2_true])

# Create new network
Q = Node(generate_random_probability_matrix((2), device=device), name='Q')
Y1 = Node(generate_random_probability_matrix((2, 2), device=device), name='Y1')
Y2 = Node(generate_random_probability_matrix((2, 2), device=device), name='Y2')
nodes = [Q, Y1, Y2]
parents = {
    Q: [],
    Y1: [Q],
    Y2: [Q]
}
network = BayesianNetwork(nodes, parents)

# Train network
num_iterations = 10

def inference_machine_factory(bayesian_network: BayesianNetwork) -> IInferenceMachine:
    return TorchSumProductAlgorithmInferenceMachine(
        bayesian_network=bayesian_network,
        observed_nodes=[Y1, Y2],
        device=device,
        num_iterations=5,
        num_observations=num_observations,
        callback=lambda a, b: None)

em_optimizer = EmOptimizer(network, inference_machine_factory)
em_optimizer.optimize(training_data, num_iterations, lambda ll, iteration: print(f'Finished iteration {iteration}/{num_iterations} - ll: {ll}'))

pass
# num_iterations = 10
#
# d = torch.zeros((num_iterations, num_observations, 2), dtype=torch.double)
# a1 = torch.zeros((num_iterations, num_observations, 2), dtype=torch.double)
# b1 = torch.zeros((num_iterations, num_observations, 2), dtype=torch.double)
# a2 = torch.zeros((num_iterations, num_observations, 2), dtype=torch.double)
# b2 = torch.zeros((num_iterations, num_observations, 2), dtype=torch.double)
# a3 = torch.zeros((num_iterations, num_observations, 2), dtype=torch.double)
# b3 = torch.zeros((num_iterations, num_observations, 2), dtype=torch.double)
# def callback(factor_graph: FactorGraph, iteration: int):
#     d[iteration] = factor_graph.variable_nodes[Y].input_from_observation
#     a1[iteration] = factor_graph.factor_nodes[Y].input_from_local_variable_node
#     b1[iteration] = factor_graph.variable_nodes[Y].input_from_local_factor_node
#     a2[iteration] = factor_graph.factor_nodes[Y].inputs_from_remote_variable_nodes[0]
#     b2[iteration] = factor_graph.variable_nodes[Q].input_from_remote_factor_nodes[0]
#     a3[iteration] = factor_graph.factor_nodes[Q].input_from_local_variable_node
#     b3[iteration] = factor_graph.variable_nodes[Q].input_from_local_factor_node
#     print(f'Finished iteration {iteration}/{num_iterations}')
#
# sp_inference_machine = TorchSumProductAlgorithmInferenceMachine(
#     bayesian_network=network,
#     observed_nodes=[Y],
#     device=device,
#     num_iterations=num_iterations,
#     num_observations=num_observations,
#     callback=callback)
#
# sp_inference_machine.enter_evidence(training_data)
# ll = sp_inference_machine.log_likelihood()
#
# plt.figure()
# plt.subplot(7, 1, 1); plt.plot(range(num_iterations), d.reshape([num_iterations, -1])); plt.title('d')
# plt.subplot(7, 1, 2); plt.plot(range(num_iterations), a1.reshape([num_iterations, -1])); plt.title('a1')
# plt.subplot(7, 1, 3); plt.plot(range(num_iterations), b1.reshape([num_iterations, -1])); plt.title('b1')
# plt.subplot(7, 1, 4); plt.plot(range(num_iterations), a2.reshape([num_iterations, -1])); plt.title('a2')
# plt.subplot(7, 1, 5); plt.plot(range(num_iterations), b2.reshape([num_iterations, -1])); plt.title('b2')
# plt.subplot(7, 1, 6); plt.plot(range(num_iterations), a3.reshape([num_iterations, -1])); plt.title('a3')
# plt.subplot(7, 1, 7); plt.plot(range(num_iterations), b3.reshape([num_iterations, -1])); plt.title('b3')

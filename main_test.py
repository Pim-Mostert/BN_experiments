import torch
from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.statistics import generate_random_probability_matrix
from bayesian_network.inference_machines.torch_sum_product_algorithm_inference_machine_2 import TorchSumProductAlgorithmInferenceMachine as InferenceMachine2
from bayesian_network.inference_machines.torch_sum_product_algorithm_inference_machine_3 import TorchSumProductAlgorithmInferenceMachine as InferenceMachine3


# Create network
Q1 = Node(
    generate_random_probability_matrix((2), device="cpu"),
    name='Q1')
Q2 = Node(
    generate_random_probability_matrix((2, 2), device="cpu"),
    name='Q2')
Y = Node(
    generate_random_probability_matrix((2, 2), device="cpu"),
    name='Y')

nodes = [Q1, Q2, Y]
parents = {
    Q1: [],
    Q2: [Q1],
    Y: [Q2],
}
observed_nodes = [Y]

bayesian_network = BayesianNetwork(nodes, parents)

# Inference machines
inference_machine2 = InferenceMachine2(
            bayesian_network=bayesian_network,
            observed_nodes=observed_nodes,
            device="cpu",
            num_iterations=20,
            num_observations=0,
            callback=lambda factor_graph, iteration: None)

inference_machine3 = InferenceMachine3(
            bayesian_network=bayesian_network,
            observed_nodes=observed_nodes,
            device="cpu",
            num_iterations=20,
            num_observations=0,
            callback=lambda factor_graph, iteration: None)

# Iterate
for _ in range(20):
    inference_machine2.factor_graph.iterate()
    inference_machine3.factor_graph.iterate()

# 1. var[Y] -> fact[Y]
inference_machine2.factor_graph.factor_nodes[Y].input_from_local_variable_node
inference_machine3.factor_graph.get_factor_node_group(Y).get_input_tensor(Y, Y)

# 2. fact[Y] -> var[Y]
inference_machine2.factor_graph.variable_nodes[Y].input_from_local_factor_node
inference_machine3.factor_graph.get_variable_node_group(Y).get_input_tensor(Y, Y)

# 3. fact[Y] -> var[Q2]
inference_machine2.factor_graph.variable_nodes[Q2].input_from_remote_factor_nodes[0]
inference_machine3.factor_graph.get_variable_node_group(Q2).get_input_tensor(Q2, Y)

# 4. var[Q2] -> fact[Y]
inference_machine2.factor_graph.factor_nodes[Y].inputs_from_remote_variable_nodes[0]
inference_machine3.factor_graph.get_factor_node_group(Y).get_input_tensor(Y, Q2)

# 5. var[Q2] -> fact[Q2]
inference_machine2.factor_graph.factor_nodes[Q2].input_from_local_variable_node
inference_machine3.factor_graph.get_factor_node_group(Q2).get_input_tensor(Q2, Q2)

# 6. fact[Q2] -> var[Q2]
inference_machine2.factor_graph.variable_nodes[Q2].input_from_local_factor_node
inference_machine3.factor_graph.get_variable_node_group(Q2).get_input_tensor(Q2, Q2)

# 7. fact[Q2] -> var[Q1]
inference_machine2.factor_graph.variable_nodes[Q1].input_from_remote_factor_nodes[0]
inference_machine3.factor_graph.get_variable_node_group(Q1).get_input_tensor(Q1, Q2)

# 8. var[Q1] -> fact[Q2]
inference_machine2.factor_graph.factor_nodes[Q2].inputs_from_remote_variable_nodes[0]
inference_machine3.factor_graph.get_factor_node_group(Q2).get_input_tensor(Q2, Q1)

# 9. var[Q1] -> fact[Q1]
inference_machine2.factor_graph.factor_nodes[Q1].input_from_local_variable_node
inference_machine3.factor_graph.get_factor_node_group(Q1).get_input_tensor(Q1, Q1)

# 10. fact[Q1] -> var[Q1]
inference_machine2.factor_graph.variable_nodes[Q1].input_from_local_factor_node
inference_machine3.factor_graph.get_variable_node_group(Q1).get_input_tensor(Q1, Q1)

# Check connections
# 1 - klopt
inference_machine3.factor_graph.get_variable_node_group(Y)._output_tensors[0][0]
inference_machine3.factor_graph.get_factor_node_group(Y).get_input_tensor(Y, Y)

# 2 - klopt
inference_machine3.factor_graph.get_factor_node_group(Y)._output_tensors[1][1]
inference_machine3.factor_graph.get_variable_node_group(Y).get_input_tensor(Y, Y)

# 3 - klopt
inference_machine3.factor_graph.get_factor_node_group(Y)._output_tensors[0][1]
inference_machine3.factor_graph.get_variable_node_group(Q2).get_input_tensor(Q2, Y)

# 4 - klopt
inference_machine3.factor_graph.get_variable_node_group(Q2)._output_tensors[0][1]
inference_machine3.factor_graph.get_factor_node_group(Y).get_input_tensor(Y, Q2)

# 5 - klopt
inference_machine3.factor_graph.get_variable_node_group(Q2)._output_tensors[1][1]
inference_machine3.factor_graph.get_factor_node_group(Q2).get_input_tensor(Q2, Q2)

# 6 - klopt
inference_machine3.factor_graph.get_factor_node_group(Q2)._output_tensors[1][0]
inference_machine3.factor_graph.get_variable_node_group(Q2).get_input_tensor(Q2, Q2)

# 7 - klopt
inference_machine3.factor_graph.get_factor_node_group(Q2)._output_tensors[0][0]
inference_machine3.factor_graph.get_variable_node_group(Q1).get_input_tensor(Q1, Q2)

# 8 - klopt
inference_machine3.factor_graph.get_variable_node_group(Q1)._output_tensors[0][0]
inference_machine3.factor_graph.get_factor_node_group(Q2).get_input_tensor(Q2, Q1)

# 9 - klopt
inference_machine3.factor_graph.get_variable_node_group(Q1)._output_tensors[1][0]
inference_machine3.factor_graph.get_factor_node_group(Q1).get_input_tensor(Q1, Q1)

# 10 - klopt
inference_machine3.factor_graph.get_factor_node_group(Q1)._output_tensors[0][0]
inference_machine3.factor_graph.get_variable_node_group(Q1).get_input_tensor(Q1, Q1)

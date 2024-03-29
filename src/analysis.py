from preprocess import preprocess
from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.torch_settings import TorchSettings
import torch
import torchvision as torchvision
from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.inference_machines.torch_sum_product_algorithm_inference_machine import \
    TorchSumProductAlgorithmInferenceMachine
from bayesian_network.interfaces import IInferenceMachine
from bayesian_network.optimizers.em_optimizer import EmOptimizer
from bayesian_network.common.torch_settings import TorchSettings

def analysis(
    torch_device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64
):
    torch_settings = TorchSettings(torch_device, dtype)

    # evidence: List[num_observed_nodes x torch.Tensor[num_observations x num_states]], one-hot encoded
    evidence = preprocess(
        gamma=0.000001,
        torch_device=torch_settings.device,
        dtype=torch_settings.dtype,
    )

    num_observations = evidence[0].shape[0]
    height = 28
    width = 28
    num_classes = 10

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
    num_iterations = 10
    num_sp_iterations = 8

    def inference_machine_factory(bayesian_network: BayesianNetwork) -> IInferenceMachine:
        return TorchSumProductAlgorithmInferenceMachine(
            bayesian_network=bayesian_network,
            observed_nodes=Ys,
            torch_settings=torch_settings,
            num_iterations=num_sp_iterations,
            num_observations=num_observations,
            callback=lambda factor_graph, iteration: print(f'Finished SP iteration {iteration}/{num_sp_iterations}'))

    em_optimizer = EmOptimizer(network, inference_machine_factory)
    em_optimizer.optimize(evidence, num_iterations, lambda ll, iteration, duration:
        print(f'Finished iteration {iteration}/{num_iterations} - ll: {ll} - it took: {duration} s'))
        
    return network
    
import torchvision
from torchvision.transforms import transforms
from torch.nn.functional import one_hot
import torch
from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.torch_settings import TorchSettings
import torch
import torch
import torchvision as torchvision
from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.inference_machines.torch_sum_product_algorithm_inference_machine import \
    TorchSumProductAlgorithmInferenceMachine
from bayesian_network.interfaces import IInferenceMachine
from bayesian_network.optimizers.em_optimizer import EmOptimizer
from bayesian_network.common.torch_settings import TorchSettings

def analysis():
    mnist = torchvision.datasets.MNIST('./mnist', train=True, transform=transforms.ToTensor(), download=True)
    data = mnist.train_data.ge(128).long()

    height, width = data.shape[1:3]
    num_features = height * width
    num_observations = data.shape[0]
    num_classes = 10

    # Morph into evidence structure
    training_data_reshaped = data.reshape([num_observations, num_features])

    # evidence: List[num_observed_nodes x torch.Tensor[num_observations x num_states]], one-hot encoded
    gamma = 0.000001
    evidence = [
        node_evidence * (1-gamma) + gamma/2
        for node_evidence 
        in one_hot(training_data_reshaped.T, 2).to(torch.float64)
    ]
            
    # Torch settings
    torch_settings = TorchSettings(torch.device('cpu'), torch.float64)
    
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
    
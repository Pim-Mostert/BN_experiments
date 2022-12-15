import torchvision
from torchvision.transforms import transforms
from torch.nn.functional import one_hot
import torch

def preprocess_mnist(
    gamma: float
):
    mnist = torchvision.datasets.MNIST('./mnist', train=True, transform=transforms.ToTensor(), download=True)
    data = mnist.train_data.ge(128).long()

    height, width = data.shape[1:3]
    num_features = height * width
    num_samples = data.shape[0]

    # Morph into evidence structure
    training_data_reshaped = data.reshape([num_samples, num_features])

    # evidence: List[num_observed_nodes x torch.Tensor[num_observations x num_states]], one-hot encoded
    evidence = [
        node_evidence * (1-gamma) + gamma/2
        for node_evidence 
        in one_hot(training_data_reshaped.T, 2).to(torch.float64)
    ]
    
    return evidence

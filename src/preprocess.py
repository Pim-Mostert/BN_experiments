import torchvision
from torchvision.transforms import transforms
from torch.nn.functional import one_hot
import torch
import torch
import torch
import torchvision as torchvision

def preprocess():
    mnist = torchvision.datasets.MNIST('./mnist', train=True, transform=transforms.ToTensor(), download=True)
    data = mnist.train_data.ge(128).long()

    height, width = data.shape[1:3]
    num_features = height * width
    num_observations = data.shape[0]

    # Morph into evidence structure
    training_data_reshaped = data.reshape([num_observations, num_features])
    
    # Make smaller selection
    training_data_reshaped = training_data_reshaped[:10000]
    num_observations = training_data_reshaped.shape[0]

    # evidence: List[num_observed_nodes x torch.Tensor[num_observations x num_states]], one-hot encoded
    gamma = 0.000001
    evidence = [
        node_evidence * (1-gamma) + gamma/2
        for node_evidence 
        in one_hot(training_data_reshaped.T, 2).to(torch.float64)
    ]
            
    return evidence    
from mldesigner import command_component, Input, Output
import torchvision
from torchvision.transforms import transforms
from torch.nn.functional import one_hot
import pickle

env = dict(
    # note that mldesigner package must be included.
    # conda_file=Path(__file__).parent / "conda.yaml",
    image="mcr.microsoft.com/azureml/curated/acpt-pytorch-1.12-py39-cuda11.6-gpu:3",
)
@command_component(
    environment=env
)
def preprocess_mnist_component(
    gamma: Input(type="number"),    
    output_file: Output(type="uri_file"),
    data_dtype: Input(type="string", default="float64"),
):
    mnist = torchvision.datasets.MNIST('./mnist', train=True, transform=transforms.ToTensor(), download=True)
    # selection = [(data, label) for data, label in zip(mnist.train_data, mnist.train_labels) if label in selected_labels] \
    #     [0:num_observations]
    data = mnist.train_data.ge(128).long()
    labels = [int(label) for label in mnist.train_labels]

    height, width = data.shape[1:3]
    num_features = height * width
    num_samples = data.shape[0]

    # Morph into evidence structure
    training_data_reshaped = data.reshape([num_samples, num_features])

    # evidence: List[num_observed_nodes x torch.Tensor[num_observations x num_states]], one-hot encoded
    evidence = [
        node_evidence * (1-gamma) + gamma/2
        for node_evidence 
        in one_hot(training_data_reshaped.T, 2).to(data_dtype)
    ]
    
    pickle.dump(evidence, output_file)
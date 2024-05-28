import argparse
from collections import OrderedDict

import flwr as fl
from flwr.server import criterion
from flwr_datasets import FederatedDataset
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

# #####################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and dataloader
# #####################################################################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the neural network
class Net(nn.Module):
    """
    Simple neural network
    """

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train(net, trainloader, epochs):
    """
    Train the model on the training set.
    :param trainloader:
    :param epochs:
    :return:
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for _ in range(epochs):
        for batch in tqdm(trainloader, "Training"):
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()


def test(net, testloader):
    """
    Validate the model on the test set.
    :param testloader:
    :return:
    """
    criterion = nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in tqdm(testloader, "Testing"):
            images = batch["img"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def load_data(partition_id):
    """
    Load CIFAR-10 dataset.
    :param partition_id:
    :return:
    """
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": 3})
    partition = fds.load_partition(partition_id)
    # Divide data on each client: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2)
    # Transform data
    pytorch_transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def apply_transforms(batch):
        """
        Apply transforms to the partition from FederatedDataset.
        :param batch:
        :return:
        """
        batch["img"] = [pytorch_transform(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)

    return trainloader, testloader


# #####################################################################
# 2. Federation of the pipeline with Flower
# #####################################################################

# Get partition_id from command line
parser = argparse.ArgumentParser(description="Flower")

parser.add_argument(
    "--partition-id",
    choices=[0, 1, 2],
    required=True,
    type=int,
    help="Partition of the dataset divided into 3 iid partitions created artificially.",
)
partition_id = parser.parse_args().partition_id

# Load model and data (Simple CNN, CIFAR-10)
net = Net().to(DEVICE)
trainloader, testloader = load_data(partition_id)


# Define the Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        """
        Set the parameters of the model.
        :param parameters: a list of numpy arrays
        :return:
        """
        paras_dict = zip(net.state_dict().keys(), parameters)
        # k: key in dict
        # v: value, Numpy array to torch Tensor
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in paras_dict})
        # Then load the state_dict into the model
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print("\n Received request for fit")
        # server send the parameters to the client and the client will train the model
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        # use get_parameters to get the updated parameters, and return training dataset size and empty dictionary
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        print("\n Received request for evaluate")
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}

# Start Flower client
fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient().to_client(),
)

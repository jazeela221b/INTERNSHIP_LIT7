import random
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
# Set the seed for reproducibility
seed = 123
torch.manual_seed(seed)
random.seed(seed)


def load_MNIST():

    # MNIST dataset
    """ Load MNIST dataset
    Returns:
    trainloader (object): training dataloader
    testloader (object): test dataloader

    """
    transformation = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307, ), (0.3081, ))
    ])
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=transformation, download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=transformation
    )

    # Split trainset into train_loader and val_loader
    size_split = int(len(train_dataset) * 0.8)
    trainset, valset = torch.utils.data.random_split(
        train_dataset, [size_split, len(train_dataset) - size_split]
    )

    testset = test_dataset

    return trainset, valset, testset


class LeNet(nn.Module):
    """MNIST model"""

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

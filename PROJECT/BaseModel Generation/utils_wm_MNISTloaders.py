import torch
import torchvision
import torchvision.transforms as transforms


def get_data():
    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=transforms.ToTensor()
    )

    # Split trainset into train_loader and val_loader
    size_split = int(len(train_dataset) * 0.8)
    trainset, valset = torch.utils.data.random_split(
        train_dataset, [size_split, len(train_dataset) - size_split]
    )

    testset = test_dataset

    return trainset, valset, testset

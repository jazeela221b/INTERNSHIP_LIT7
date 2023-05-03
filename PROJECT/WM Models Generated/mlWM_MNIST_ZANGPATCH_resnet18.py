import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
from utils_wm_MNISTloaders import get_data
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from Watermarking_zang.ml_model_watermarking_main.mlmodelwatermarking.marktorch import Trainer
from Watermarking_zang.ml_model_watermarking_main.mlmodelwatermarking import TrainingWMArgs

# Set the seed for reproducibility
seed = 123
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


# Hyper-parameters
batch_size = 64

# Define modified ResNet18 model


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=False)
        # modify the first convolutional layer to accept 1 channel input
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x


# Replace NeuralNet class with modified ResNet18 model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = ResNet18()

    def forward(self, x):
        x = self.resnet(x)
        return x


def MNIST_patch():
    """Testing of watermarking for MNIST model."""

    # WATERMARKED
    model = Net()
    trainset, valset, testset = get_data()
    path_args = {"msg": "ID42", "target": 5}

    args = TrainingWMArgs(
        trigger_technique="patch",
        optimizer="SGD",
        lr=0.1,
        gpu=True,
        epochs=10,
        nbr_classes=10,
        batch_size=64,
        trigger_patch_args=path_args,
        watermark=True,
    )

    trainer = Trainer(
        model=model, args=args, trainset=trainset, valset=valset, testset=testset
    )

    ownership = trainer.train()
    accuracy_wm_regular = trainer.test()
    verification = trainer.verify(ownership)
    print(
        f'Do we detect the watermark in the watermark_model ? {verification["is_stolen"]}\n'
    )
    assert verification["is_stolen"] is True
    # CLEAN
    model = Net()
    args.watermark = False
    trainer_clean = Trainer(
        model=model, args=args, trainset=trainset, valset=valset, testset=testset
    )
    trainer_clean.train()
    accuracy_clean_regular = trainer_clean.test()
    accuracy_loss = round(accuracy_clean_regular - accuracy_wm_regular, 4)
    print(f"Accuracy loss: {accuracy_loss}")
    clean_model = trainer_clean.get_model()

    verification = trainer.verify(ownership, suspect=clean_model)
    assert verification["is_stolen"] is False
    print(
        f'Do we detect the watermark in the watermark_model ? {verification["is_stolen"]}\n'
    )


MNIST_patch()

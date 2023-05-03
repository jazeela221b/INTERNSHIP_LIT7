import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from utils_wm_MNISTloaders import get_data
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score


from Watermarking_zang.ml_model_watermarking_main.mlmodelwatermarking.marktorch import (
    Trainer,
)
from Watermarking_zang.ml_model_watermarking_main.mlmodelwatermarking import (
    TrainingWMArgs,
)

# Set the seed for reproducibility
seed = 123
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


# Hyper-parameters
input_size = 28 * 28  # 28x28   # 3 size of image channels (3 for RGB)
hidden_size = 10000
num_classes = 10  # number of classes in your dataset
batch_size = 64
#num_epochs = 2
#learning_rate = 0.001


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Add batch normalization layer
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.l1(x)
        x = self.bn1(x)  # Apply batch normalization
        x = self.relu(x)
        x = self.l2(x)
        return x


def MNIST_patch():
    """Testing of watermarking for MNIST model."""

    # WATERMARKED
    model = NeuralNet(input_size, hidden_size, num_classes)
    trainset, valset, testset = get_data()
    path_args = {"msg": "ID42", "target": 5}

    args = TrainingWMArgs(
        trigger_technique="patch",
        optimizer="SGD",
        lr=0.0001,
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
    model = NeuralNet(input_size, hidden_size, num_classes)
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

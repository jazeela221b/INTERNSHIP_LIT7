import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_wm_MNISTloaders import get_data
from mlmodelwatermarking import TrainingWMArgs
from mlmodelwatermarking.verification import verify
from mlmodelwatermarking.marktorch import Trainer

# Set the seed for reproducibility
seed = 123
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


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


def MNIST_patch():
    """Testing of watermarking for MNIST model."""

    # WATERMARKED
    model = LeNet()
    trainset, valset, testset = get_data()
    path_args = {"msg": "ID42", "target": 5}

    args = TrainingWMArgs(
        trigger_technique="patch",
        optimizer="SGD",
        lr=0.01,
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
    model = LeNet()
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

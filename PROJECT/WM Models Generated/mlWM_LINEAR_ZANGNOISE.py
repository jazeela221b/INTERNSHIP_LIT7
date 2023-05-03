import random
import numpy as np
from mlmodelwatermarking import TrainingWMArgs
from mlmodelwatermarking.verification import verify
from mlmodelwatermarking.marktorch import Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from INTERNSHIP_LIT7.PROJECT.BaseModel_Generation import util
# Set the seed for reproducibility
seed = 123
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

X, y = util.get_linear(1000)
# Convert the data to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)


# Reshape the target labels to the expected shape
y = y.reshape(-1, 1)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


def MNIST_noise():
    """Testing of watermarking for MNIST model."""

    # Split the data into train, validation, and test sets
    trainset = TensorDataset(X[:60].float(), y[:60].float())
    valset = TensorDataset(X[60:80].float(), y[60:80].float())
    testset = TensorDataset(X[80:].float(), y[80:].float())

    args = TrainingWMArgs(
        trigger_technique="noise",
        optimizer="SGD",
        lr=0.01,
        gpu=True,
        epochs=10,
        nbr_classes=2,
        batch_size=50,
        watermark=True,
    )
    # WATERMARKED
    model = Net()
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

    print(
        f'Do we detect the watermark in the watermark_model ? {verification["is_stolen"]}\n'
    )
    assert verification["is_stolen"] is False


if __name__ == "__main__":
    MNIST_noise()

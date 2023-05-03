import random
import numpy as np
import torch
import torch.nn.functional as F
from mlmodelwatermarking import TrainingWMArgs
from mlmodelwatermarking.verification import verify
from mlmodelwatermarking.marktorch import Trainer
from utils_MNIST_CNN import LeNet, load_MNIST

# Set the seed for reproducibility
seed = 123
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


def MNIST_patch():
    """Testing of watermarking for MNIST model."""

    # WATERMARKED
    model = LeNet()
    trainset, valset, testset = load_MNIST()

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


################################################################################################

    # Build a new model and use the noisy_train_loader to train it : adding noise to the trainsetand verify its ownership
    stolen_model = LeNet()
    args.watermark = True

    # Define a function that adds Gaussian noise to an image

    def add_gaussian_noise(image, mean=0, std=1):
        noise = torch.randn(image.size()) * std + mean
        noisy_image = image + noise
        return noisy_image

    # Apply the noise function to each sample of the train_loader data : add noise to the trainset
    noisy_train_loader = []
    for data in trainset:
        images, labels = data
        noisy_images = add_gaussian_noise(images)
        noisy_train_loader.append((noisy_images, labels))
    noisy_trainset = torch.utils.data.Subset(
        trainset.dataset, range(len(noisy_train_loader)))

    trainer_stolen = Trainer(
        model=stolen_model, args=args, trainset=noisy_trainset, valset=valset, testset=testset
    )

    trainer_stolen.train()
    accuracy_stolen = trainer_stolen.test()
    print(f"Accuracy of stolen model: {accuracy_stolen}")
    accuracy_loss = round(accuracy_stolen - accuracy_wm_regular, 4)
    print(f"Accuracy loss: {accuracy_loss}")
    stolen_model = trainer_stolen.get_model()
    verification = trainer.verify(
        ownership, suspect=stolen_model)
    print(
        f'Do we detect the watermark in the stolen_model ? {verification["is_stolen"]}\n'
    )
    assert verification["is_stolen"] is True


MNIST_patch()

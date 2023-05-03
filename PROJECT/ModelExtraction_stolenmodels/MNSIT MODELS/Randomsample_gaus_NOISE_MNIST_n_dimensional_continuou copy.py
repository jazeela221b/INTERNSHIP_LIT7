import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from utils1 import get_data
import random
import torchvision.transforms as transforms

# Hyper-parameters
input_size = 784  # 28x28   # 3 size of image channels (3 for RGB)
hidden_size = 100
num_classes = 10  # number of classes in your dataset
num_epochs = 2
batch_size = 100
learning_rate = 0.001


# Load the data
train_loader, test_loader = get_data(batch_size)


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        # cross-entropy loss implemented further will apply it automatically
        return out


# Build the model ie Load the original model
model = NeuralNet(input_size, hidden_size, num_classes)
if torch.cuda.is_available():
    model.cuda()

# Loss and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Training the model
n_total_steps = len(train_loader)  # length of training loader
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        # origin shape: [100, 1, 28, 28]
        # resized: [100, 784]
        images = images.reshape(-1, input_size)
        labels = labels

        # forward pass
        outputs = model(images)
        loss = loss_function(outputs, labels)

        # backwards pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (i + 1) % 100 == 0:
        #    print(
        #        f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}"
        #   )

# Test the model
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, input_size)  # reshape the input tensor
        labels = labels
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f"Accuracy of the model on the test images: {acc}%")


##########################################################################################################################################


# add noise to the train_loader here and put it in the stolen_model
# Define a function that adds Gaussian noise to an image
def add_gaussian_noise(image, mean=0, std=1):
    noise = torch.randn(image.size()) * std + mean
    noisy_image = image + noise
    return noisy_image


# Apply the noise function to each sample of the train_loader data
noisy_train_loader = []
for data in train_loader:
    images, labels = data
    noisy_images = add_gaussian_noise(images)
    noisy_train_loader.append((noisy_images, labels))


# Build a new model and use the noisy_train_loader to train it
stolen_model = NeuralNet(input_size, hidden_size, num_classes)
if torch.cuda.is_available():
    stolen_model.cuda()

# Loss and optimizer for the new model
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(stolen_model.parameters(), lr=learning_rate)

# Training the new model with the noisy data
n_total_steps = len(noisy_train_loader)  # length of training loader
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(noisy_train_loader):
        images = images.reshape(-1, input_size)
        labels = labels
        outputs = stolen_model(images)
        loss = loss_function(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # if (i + 1) % 100 == 0:
    #    print(
    #        f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}"
    #   )

# Test the new model

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, input_size)  # reshape the input tensor
        labels = labels
        outputs = stolen_model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(
        f"case 1_MNIST_NOISE: Accuracy of the Stolen model on the test images: {acc}%"
    )


#####################################################################

import numpy as np
from sklearn.model_selection import train_test_split
import util
import torch
import torch.nn as nn
import torch.nn.functional as F

# load the data
X, y = util.get_linear(1000)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=33
)

# Convert the data to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Reshape the target labels to the expected shape
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Define the model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


model = Net()

# Set the optimizer and loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Make predictions on the test set
with torch.no_grad():
    predicted = model(X_test)
    predicted = predicted.round()

# Calculate the accuracy of the model on the test set
accuracy = (predicted == y_test).sum().item() / y_test.size(0)
print("Accuracy:", accuracy)

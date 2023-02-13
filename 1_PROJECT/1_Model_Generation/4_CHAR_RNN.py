import torch
import torch.nn as nn

# import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

from utils_char import ALL_LETTERS, N_LETTERS
from utils_char import (
    load_data,
    letter_to_tensor,
    line_to_tensor,
    random_training_example,
)


class RNN(nn.Module):
    # implement RNN from scratch rather than using nn.RNN
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)

        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


###############################
# Load the data
category_lines, all_categories = load_data()
n_categories = len(all_categories)
# print(n_categories)


######################
# Define the model
n_hidden = 128
rnn = RNN(N_LETTERS, n_hidden, n_categories)

# one step
# input_tensor = letter_to_tensor("A")
# hidden_tensor = rnn.init_hidden()

# output, next_hidden = rnn(input_tensor, hidden_tensor)
# print(output.size())
# print(next_hidden.size())

# whole sequence/name
input_tensor = line_to_tensor("Albert")
hidden_tensor = rnn.init_hidden()

output, next_hidden = rnn(input_tensor[0], hidden_tensor)
# print(output.size())
# print(next_hidden.size())


def category_from_output(output):
    category_idx = torch.argmax(output).item()
    return all_categories[category_idx]


# Print("this is for checking first cat frm outpt", category_from_output(output))

criterion = nn.NLLLoss()
learning_rate = 0.005
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

# variables to keep track of accuracy
correct_count = 0
total_count = 0


def train(line_tensor, category_tensor):
    hidden = rnn.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()


current_loss = 0
all_losses = []
plot_steps, print_steps = 1000, 5000
n_iters = 100000
for i in range(n_iters):
    category, line, category_tensor, line_tensor = random_training_example(
        category_lines, all_categories
    )

    output, loss = train(line_tensor, category_tensor)
    current_loss += loss

    # check accuracy
    guess = category_from_output(output)
    if guess == category:
        correct_count += 1
    total_count += 1

    # if (i + 1) % plot_steps == 0:
    #    all_losses.append(current_loss / plot_steps)
    #    current_loss = 0

    # if (i + 1) % print_steps == 0:
    # accuracy = correct_count / total_count
    # print(f"Iteration: {i+1} Loss: {loss} Accuracy: {accuracy}")

# Calculate final accuracy after training
final_accuracy = correct_count / total_count
print(f"Final Accuracy: {final_accuracy}")

import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torch
from torch import nn
import torch.optim as optim

# Ensure only CPU is used
torch.cuda.is_available = lambda: False

# Using CPU for computations
device = torch.device('cpu')

# Function to apply labeling rules
def apply_rules(sequence):
    tokens = sequence.split()
    rule_1 = all(token[0] == tokens[0][0] for token in tokens)
    rule_2 = tokens[0][-1] == tokens[-1][-1]
    rule_3 = all(tokens[i][-1] != tokens[i + 1][-1] for i in range(len(tokens) - 1))
    return 1 if rule_1 or rule_2 or rule_3 else 0

# Synthetic data generation
def generate_sequences(num_sequences, length):
    symbols = ['▲', '■', '●', '◆']
    colors = ['r', 'g', 'b', 'y']
    data = []
    for _ in range(num_sequences):
        sequence = ' '.join(random.choice(symbols) + random.choice(colors) for _ in range(length))
        label = apply_rules(sequence)
        data.append((sequence, label))
    return data

# Data preparation
num_sequences = 300
sequence_length = 5
data = generate_sequences(num_sequences, sequence_length)
sequences, labels = zip(*data)
syn_df = pd.DataFrame({'sequence': sequences, 'label': labels})

# Split the data
train_df = syn_df.iloc[:int(0.8 * len(syn_df))]
dev_df = syn_df.iloc[int(0.8 * len(syn_df)):int(0.9 * len(syn_df))]
test_df = syn_df.iloc[int(0.9 * len(syn_df)):]

# Prepare PyTorch tensors
def prepare_tensors(df):
    sequences = list(df['sequence'])
    labels = list(df['label'])
    sequence_tensors = []
    
    for seq in sequences:
        # Convert sequence to a tensor of ASCII values
        tensor = torch.tensor(
            [ord(char) for token in seq.split() for char in token] + [0] * (10 * sequence_length - len(seq.split()) * 2),
            dtype=torch.float
        )
        sequence_tensors.append(tensor)
        
    label_tensors = torch.tensor(labels, dtype=torch.long)
    return sequence_tensors, label_tensors

train_sequences, train_labels = prepare_tensors(train_df)
dev_sequences, dev_labels = prepare_tensors(dev_df)
test_sequences, test_labels = prepare_tensors(test_df)

# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Input size adjustment
input_dim = 10 * sequence_length  # Adjust input dimension for the network

# Instantiate the model, define optimizer and loss
model = SimpleNN(input_size=input_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training function
def train():
    model.train()
    for epoch in range(10):
        total_loss = 0
        for sequence, label in zip(train_sequences, train_labels):
            sequence = sequence.to(device).unsqueeze(0)  # Add batch dimension
            label = label.unsqueeze(0).to(device)
            optimizer.zero_grad()
            output = model(sequence)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_sequences):.4f}")

    print("Training completed.")

# Evaluation function
def evaluate(sequences, labels, name='Dev'):
    model.eval()
    predictions, true_labels = [], []
    correct = 0

    with torch.no_grad():
        for sequence, label in zip(sequences, labels):
            sequence = sequence.to(device).unsqueeze(0)  # Add batch dimension
            label = label.to(device)
            output = model(sequence)
            predicted = output.argmax(dim=1)
            predictions.append(predicted.item())
            true_labels.append(label.item())
            correct += (predicted == label).sum().item()

    accuracy = correct / len(true_labels)
    print(f"{name} Accuracy: {accuracy:.2f}")
    return true_labels, predictions

# Plotting accuracy results
def plot_results(dev_acc, test_acc):
    plt.figure(figsize=(10, 5))
    plt.axhline(y=dev_acc, color='r', linestyle='--', label='Dev Accuracy')
    plt.axhline(y=test_acc, color='g', linestyle='--', label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Development and Test Set Accuracies')
    plt.savefig('Figure_1.png')
    plt.show()

# Execute training and evaluation
train()

# Evaluate development and testing set
_, dev_predictions = evaluate(dev_sequences, dev_labels, 'Dev')
_, test_predictions = evaluate(test_sequences, test_labels, 'Test')

# Plot results
dev_accuracy = accuracy_score(dev_labels.numpy(), dev_predictions)
test_accuracy = accuracy_score(test_labels.numpy(), test_predictions)
plot_results(dev_accuracy, test_accuracy)
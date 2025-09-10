import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Model implementation
class NeuralGrammarSymbolicModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads):
        super(NeuralGrammarSymbolicModel, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        x = x.permute(1, 0, 2)  # Expecting input in (seq_len, batch, features)
        x, _ = self.attention(x, x, x)
        x = x.permute(1, 0, 2)
        x = self.feed_forward(x)
        return x.mean(dim=1)  # Aggregate over sequence length

# Define synthetic dataset creation
def create_synthetic_dataset(num_samples, sequence_length, num_features):
    data = np.random.rand(num_samples, sequence_length, num_features)
    labels = np.random.randint(0, 2, num_samples)  # Binary classification
    return data, labels

# Evaluation functions
def evaluate_model(model, dataloader):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for data, labels in dataloader:
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.numpy())
            actuals.extend(labels.numpy())
    return accuracy_score(actuals, predictions)

# Dataset Class
class SyntheticDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Setting model parameters
input_dim = 32
hidden_dim = 64
output_dim = 2
num_heads = 4

# Creating datasets
num_samples = 1000
sequence_length = 10
num_features = 32
batch_size = 32

train_data, train_labels = create_synthetic_dataset(num_samples, sequence_length, num_features)
dev_data, dev_labels = create_synthetic_dataset(num_samples, sequence_length, num_features)

train_dataset = SyntheticDataset(train_data, train_labels)
dev_dataset = SyntheticDataset(dev_data, dev_labels)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size)

# Instantiate model
model = NeuralGrammarSymbolicModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_heads=num_heads)

# Model training placeholder
print("Training the neural grammar symbolic model... [This would be detailed with epochs and optimizer]")

# Baseline Models Comparison
dt_model = DecisionTreeClassifier()
rf_model = RandomForestClassifier()
svm_model = SVC()

# Reshape data for baselines
train_data_flat = train_data.reshape(train_data.shape[0], -1)
dev_data_flat = dev_data.reshape(dev_data.shape[0], -1)

# Training decision tree model
dt_model.fit(train_data_flat, train_labels)
dt_predictions = dt_model.predict(dev_data_flat)
dt_accuracy = accuracy_score(dev_labels, dt_predictions)
print(f"Decision Tree Accuracy: {dt_accuracy}")

# Training random forest model
rf_model.fit(train_data_flat, train_labels)
rf_predictions = rf_model.predict(dev_data_flat)
rf_accuracy = accuracy_score(dev_labels, rf_predictions)
print(f"Random Forest Accuracy: {rf_accuracy}")

# Training SVM model
svm_model.fit(train_data_flat, train_labels)
svm_predictions = svm_model.predict(dev_data_flat)
svm_accuracy = accuracy_score(dev_labels, svm_predictions)
print(f"SVM Accuracy: {svm_accuracy}")

# Evaluate the neural-grammar-symbolic model
model_accuracy = evaluate_model(model, dev_dataloader)
print(f"Neural Grammar Symbolic Model Accuracy: {model_accuracy}")

# Generating figures
plt.figure()
plt.plot([dt_accuracy, rf_accuracy, svm_accuracy, model_accuracy], marker='o')
plt.xticks(range(4), ['Decision Tree', 'Random Forest', 'SVM', 'Neural Grammar'])
plt.ylabel('Accuracy')
plt.title('Figure_1: Model Accuracies Comparison')
plt.savefig('Figure_1.png')
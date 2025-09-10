import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Set PyTorch to use CPU only
torch.cuda.is_available = lambda : False

# Custom Dataset class to convert HuggingFace Dataset to PyTorch Dataset
class CustomDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        sequence = torch.tensor(item['sequence'], dtype=torch.long)
        label = torch.tensor(item['label'], dtype=torch.long)
        return sequence, label

# Function to preprocess sequences
def preprocess(sequence):
    tokens = sequence.split()
    token_to_index = {token: idx for idx, token in enumerate(set(tokens))}
    return [token_to_index[token] for token in tokens]

# Padding sequences to ensure uniform length
def pad_sequence(sequence, max_length):
    return sequence + [0] * (max_length - len(sequence))

# Preprocess and pad datasets
tokenized_train = train_dataset.map(lambda x: {'sequence': preprocess(x['sequence'])})
tokenized_dev = dev_dataset.map(lambda x: {'sequence': preprocess(x['sequence'])})
tokenized_test = test_dataset.map(lambda x: {'sequence': preprocess(x['sequence'])})

max_seq_length = max(max(len(ex['sequence']) for ex in tokenized_train),
                     max(len(ex['sequence']) for ex in tokenized_dev),
                     max(len(ex['sequence']) for ex in tokenized_test))

padded_train = tokenized_train.map(lambda x: {'sequence': pad_sequence(x['sequence'], max_seq_length)})
padded_dev = tokenized_dev.map(lambda x: {'sequence': pad_sequence(x['sequence'], max_seq_length)})
padded_test = tokenized_test.map(lambda x: {'sequence': pad_sequence(x['sequence'], max_seq_length)})

# Convert datasets to DataLoader for batch processing
train_loader = DataLoader(CustomDataset(padded_train), batch_size=32, shuffle=True)
dev_loader = DataLoader(CustomDataset(padded_dev), batch_size=32, shuffle=False)
test_loader = DataLoader(CustomDataset(padded_test), batch_size=32, shuffle=False)

# Define Dynamic Graph CNN Model
class DGCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DGCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1).float()
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Set device to CPU
device = torch.device('cpu')

# Instantiate and train model
model = DGCNN(input_dim=max_seq_length, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model on CPU
print("Training the model on CPU...")
for epoch in range(5):
    model.train()
    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(sequences.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluating the model
def evaluate_model(loader, name):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for sequences, batch_labels in loader:
            sequences = sequences.to(device)
            outputs = model(sequences.float())
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.tolist())
            labels.extend(batch_labels.tolist())

    accuracy = accuracy_score(labels, preds)
    print(f"{name} Accuracy: {accuracy:.2f}")
    return accuracy

evaluate_model(dev_loader, "Development")
evaluate_model(test_loader, "Test")

# Visualization of results
print("Generating visualizations for test samples...")
for i in range(2):
    G = nx.Graph()
    sequence = padded_test[i]['sequence']
    for j in range(len(sequence)):
        G.add_node(j, label=sequence[j])
    plt.figure()
    nx.draw(G, with_labels=True, node_size=700)
    plt.title(f'Test Sample Graph {i+1}')
    plt.savefig(f'Figure_{i+1}.png')
    plt.close()
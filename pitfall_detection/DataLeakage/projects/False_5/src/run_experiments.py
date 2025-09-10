import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from datasets import load_dataset

# Ensure torch does not use CUDA
torch.cuda.is_available = lambda: False

# Load the dataset
dataset = load_dataset('csv', data_files={
    'train': 'SPR_BENCH/train.csv', 
    'dev': 'SPR_BENCH/dev.csv', 
    'test': 'SPR_BENCH/test.csv'
})

# Create a symbol to integer mapping
def create_symbol_mapping(dataset):
    symbol_set = set()
    for split in dataset:
        for seq in dataset[split]['sequence']:
            symbol_set.update(seq.split())
    return {symbol: idx for idx, symbol in enumerate(sorted(symbol_set))}

# Create the symbol-to-number mapping
symbol_to_num = create_symbol_mapping(dataset)
print("\nSymbol-to-Number Mapping:")
print(symbol_to_num)

# Encode sequences with the symbol-to-number mapping
def encode_and_update_dataset(dataset, symbol_to_num):
    def encode(row):
        return {'encoded_sequence': [symbol_to_num[symbol] for symbol in row['sequence'].split()]}
    return dataset.map(encode, remove_columns=['sequence'])

# Encode the dataset
encoded_dataset = encode_and_update_dataset(dataset, symbol_to_num)

# Pad sequences to fixed length
def pad_sequences(sequences, maxlen):
    padded_sequences = np.zeros((len(sequences), maxlen), dtype=np.float32)
    for i, seq in enumerate(sequences):
        truncated_seq = seq[:maxlen]
        padded_sequences[i, :len(truncated_seq)] = truncated_seq
    return padded_sequences

# Define a simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Set device to CPU only
device = torch.device('cpu')

# Set model parameters
maxlen = max(len(seq) for seq in encoded_dataset['train']['encoded_sequence'])
input_dim = maxlen
hidden_dim = 128
output_dim = len(set(encoded_dataset['train']['label']))

# Initialize the model, optimizer, and loss function
model = SimpleNN(input_dim, hidden_dim, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train model for one epoch
def train_epoch(model, dataloader, optimizer, criterion, maxlen):
    model.train()
    total_loss = 0
    for batch in dataloader:
        sequences = pad_sequences(batch['encoded_sequence'], maxlen)
        sequences = torch.tensor(sequences, dtype=torch.float32).to(device)
        labels = torch.tensor(batch['label'], dtype=torch.long).to(device)
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Evaluate model
def evaluate(model, dataloader, maxlen):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            sequences = pad_sequences(batch['encoded_sequence'], maxlen)
            sequences = torch.tensor(sequences, dtype=torch.float32).to(device)
            labels = torch.tensor(batch['label'], dtype=torch.long).to(device)
            outputs = model(sequences)
            _, preds = outputs.max(1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return accuracy_score(true_labels, predictions)

# Create data loaders
train_loader = DataLoader(encoded_dataset['train'], batch_size=32, shuffle=True, num_workers=0, drop_last=True)
dev_loader = DataLoader(encoded_dataset['dev'], batch_size=32, num_workers=0, drop_last=True)
test_loader = DataLoader(encoded_dataset['test'], batch_size=32, num_workers=0, drop_last=True)

# Training procedure
epochs = 5
val_accs = []

print("Experiment 1: Training and evaluating on the dev set:")
for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, maxlen)
    val_acc = evaluate(model, dev_loader, maxlen)
    val_accs.append(val_acc)
    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

# Plot validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), val_accs, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy over Epochs')
plt.legend()
plt.savefig('Figure_1.png')

# Evaluate the model on the test set
print("Experiment 2: Evaluating on the test set:")
test_acc = evaluate(model, test_loader, maxlen)
print(f"Test Accuracy: {test_acc:.4f}")

# Plot test results
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 2), val_accs + [test_acc], marker='o', label='Test Accuracy after Training')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Test and Validation Accuracies')
plt.legend()
plt.savefig('Figure_2.png')
plt.show()
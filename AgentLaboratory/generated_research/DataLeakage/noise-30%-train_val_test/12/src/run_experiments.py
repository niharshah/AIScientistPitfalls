import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Ensure we're using CPU
device = torch.device('cpu')

# Load and preprocess the data
def load_data():
    train_data = pd.read_csv('SPR_BENCH/train.csv')
    dev_data = pd.read_csv('SPR_BENCH/dev.csv')
    test_data = pd.read_csv('SPR_BENCH/test.csv')

    # Tokenizing the sequences
    def tokenize(sequence):
        return sequence.split()

    for data in [train_data, dev_data, test_data]:
        data['tokens'] = data['sequence'].map(tokenize)
    
    return train_data, dev_data, test_data

train_data, dev_data, test_data = load_data()

# Create a mapping of tokens to indices
unique_tokens = {token for tokens in train_data['tokens'] for token in tokens}
token_to_index = {token: idx + 1 for idx, token in enumerate(unique_tokens)}
token_to_index['<PAD>'] = 0  # Add padding token to the dictionary

# Encode sequences with padding
def encode_sequence(tokens, token_to_index, max_len=30):
    encoded = [token_to_index.get(token, 0) for token in tokens]
    return encoded[:max_len] + [token_to_index['<PAD>']] * (max_len - len(encoded))

for data in [train_data, dev_data, test_data]:
    data['encoded'] = data['tokens'].map(lambda x: encode_sequence(x, token_to_index))

# Define the Custom Dataset
class SPRDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# Create datasets
train_dataset = SPRDataset(train_data['encoded'].tolist(), train_data['label'].tolist())
dev_dataset = SPRDataset(dev_data['encoded'].tolist(), dev_data['label'].tolist())
test_dataset = SPRDataset(test_data['encoded'].tolist(), test_data['label'].tolist())

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the Model
class GCNWithAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_classes, max_len):
        super(GCNWithAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gcn_layer = nn.Linear(embedding_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1)
        self.fc = nn.Linear(hidden_dim * max_len, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(output_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = F.relu(self.gcn_layer(x))
        x = x.permute(1, 0, 2)  # Rearrange to match attention input requirements
        x, _ = self.attn(x, x, x)
        x = x.permute(1, 0, 2)  # Rearrange back to the original format
        x = x.contiguous().view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc(x)))
        x = self.out(x)
        return x

# Model parameters
vocab_size = len(token_to_index)
embedding_dim = 64
hidden_dim = 128
output_dim = 64
num_classes = 2
max_len = 30

# Model creation
model = GCNWithAttention(vocab_size, embedding_dim, hidden_dim, output_dim, num_classes, max_len).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training function
def train_model(model, train_loader, dev_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs} Loss: {loss.item():.4f}')

    return model

# Train the model
model = train_model(model, train_loader, dev_loader, criterion, optimizer)

# Evaluate the model
def evaluate_model(model, dataloader):
    model.eval()
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            outputs = model(sequences)
            _, preds = torch.max(outputs, 1)
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return accuracy_score(all_labels, all_predictions)

# Calculate accuracy
test_accuracy = evaluate_model(model, test_loader)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Visual results
_, ax = plt.subplots(figsize=(5, 5))
ax.set_title('Model Test Accuracy Comparison')
x = ['Our Model']
y = [test_accuracy]
ax.bar(x, y, color='blue')
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.savefig('Figure_1.png')
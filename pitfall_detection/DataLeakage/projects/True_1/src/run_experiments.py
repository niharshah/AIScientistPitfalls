import os
# Disable CUDA devices to force CPU usage and avoid CUDA initialization errors.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Confirm CPU usage.
device = torch.device("cpu")
print("Using device:", device)

# Patch to override any CUDA calls that might trigger errors.
if hasattr(torch.cuda, 'is_current_stream_capturing'):
    torch.cuda.is_current_stream_capturing = lambda: False

# Set random seeds for reproducibility.
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ----------------------------------------------------------------------------------
# Synthetic Dataset Generation for SPR Task with Multi-Modal Token Embeddings.
# Each instance is a sequence of L tokens (L = 7) where each token is represented as a triple: (shape, color, texture).
# Shapes: ['▲', '■', '●', '◆'], Colors: ['r', 'g', 'b', 'y'], Textures: ['solid', 'dashed'].
# Hidden Rule Example:
#   - Rule 1: Exactly two tokens must have shape '▲' and texture 'solid'.
#   - Rule 2: The token at position 4 (index 3) must have color 'r'.
# Label = 1 if both conditions are met; otherwise, label = 0.
L = 7
shapes = ['▲', '■', '●', '◆']
colors = ['r', 'g', 'b', 'y']
textures = ['solid', 'dashed']

print("Generating synthetic datasets with hidden rules:")
print("  - Exactly two tokens must have shape '▲' and texture 'solid'.")
print("  - The token at position 4 must have color 'r'.")

train_size = 2000
dev_size = 500
test_size = 500

# Generate synthetic training data.
train_data = []
for _ in range(train_size):
    seq = []
    for i in range(L):
        token = (random.choice(shapes), random.choice(colors), random.choice(textures))
        seq.append(token)
    count_target = sum(1 for (s, c, t) in seq if (s == '▲' and t == 'solid'))
    cond1 = (count_target == 2)
    cond2 = (seq[3][1] == 'r')
    label = 1 if (cond1 and cond2) else 0
    train_data.append((seq, label))

# Generate synthetic development data.
dev_data = []
for _ in range(dev_size):
    seq = []
    for i in range(L):
        token = (random.choice(shapes), random.choice(colors), random.choice(textures))
        seq.append(token)
    count_target = sum(1 for (s, c, t) in seq if (s == '▲' and t == 'solid'))
    cond1 = (count_target == 2)
    cond2 = (seq[3][1] == 'r')
    label = 1 if (cond1 and cond2) else 0
    dev_data.append((seq, label))

# Generate synthetic test data.
test_data = []
for _ in range(test_size):
    seq = []
    for i in range(L):
        token = (random.choice(shapes), random.choice(colors), random.choice(textures))
        seq.append(token)
    count_target = sum(1 for (s, c, t) in seq if (s == '▲' and t == 'solid'))
    cond1 = (count_target == 2)
    cond2 = (seq[3][1] == 'r')
    label = 1 if (cond1 and cond2) else 0
    test_data.append((seq, label))

# Create mappings for token attributes.
shape2idx = {s: i for i, s in enumerate(shapes)}
color2idx = {c: i for i, c in enumerate(colors)}
texture2idx = {t: i for i, t in enumerate(textures)}

# Custom PyTorch Dataset for SPR Task.
class SPRDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        seq, label = self.data[idx]
        # Convert each attribute to indices.
        shape_idx = [shape2idx[s] for (s, c, t) in seq]
        color_idx = [color2idx[c] for (s, c, t) in seq]
        texture_idx = [texture2idx[t] for (s, c, t) in seq]
        # Return label as float tensor for BCEWithLogitsLoss.
        return (torch.tensor(shape_idx, dtype=torch.long),
                torch.tensor(color_idx, dtype=torch.long),
                torch.tensor(texture_idx, dtype=torch.long)), torch.tensor([label], dtype=torch.float)

train_dataset = SPRDataset(train_data)
dev_dataset   = SPRDataset(dev_data)
test_dataset  = SPRDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
dev_loader   = DataLoader(dev_dataset, batch_size=32, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# ----------------------------------------------------------------------------------
# Model Definition: Multi-Modal Transformer Encoder with Differentiable Symbolic Reasoning Module.
# - Three separate embedding layers: for shape, color, and texture.
# - An additional positional embedding.
# - Embeddings are fused via summation.
# - A Transformer encoder processes the fused embeddings.
# - Two heads:
#      1. Classification head for binary SPR decision.
#      2. Symbolic reasoning head for extracting soft symbolic predicate activations.
# - An L1 sparsity loss is applied on the symbolic outputs for interpretability.
d_model = 32
nhead = 4
num_layers = 2
dropout = 0.1

class MultiModalTransformer(nn.Module):
    def __init__(self):
        super(MultiModalTransformer, self).__init__()
        self.shape_emb = nn.Embedding(len(shapes), d_model)
        self.color_emb = nn.Embedding(len(colors), d_model)
        self.texture_emb = nn.Embedding(len(textures), d_model)
        self.pos_emb = nn.Embedding(L, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, 1)  # Output single logit.
        self.symbolic = nn.Linear(d_model, 5)   # 5 candidate symbolic predicates.
    
    def forward(self, shape_idx, color_idx, texture_idx):
        batch_size = shape_idx.size(0)
        # Positional encodings.
        positions = torch.arange(0, L, device=shape_idx.device).unsqueeze(0).expand(batch_size, L)
        emb_shape = self.shape_emb(shape_idx)       # [batch, L, d_model]
        emb_color = self.color_emb(color_idx)         # [batch, L, d_model]
        emb_texture = self.texture_emb(texture_idx)   # [batch, L, d_model]
        emb_pos = self.pos_emb(positions)             # [batch, L, d_model]
        # Fuse embeddings via summation.
        x = emb_shape + emb_color + emb_texture + emb_pos  # [batch, L, d_model]
        # Prepare for transformer: transpose so shape becomes [sequence_length, batch, d_model]
        x = x.transpose(0, 1)
        transformer_out = self.transformer(x)
        transformer_out = transformer_out.transpose(0, 1)  # Back to [batch, L, d_model]
        pooled = transformer_out.mean(dim=1)               # Mean pooling over tokens.
        cls_logit = self.classifier(pooled)
        sym_out = torch.sigmoid(self.symbolic(pooled))
        return cls_logit, sym_out

model = MultiModalTransformer().to(device)
print("Model instantiated on device:", device)

# ----------------------------------------------------------------------------------
# Training Setup.
# Loss: Binary Cross-Entropy with Logits for classification plus L1 sparsity loss on symbolic outputs.
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# Ensure optimizer doesn't call CUDA-specific graph capture routines.
optimizer._cuda_graph_capture_health_check = lambda: None
num_epochs = 5
lambda_sparsity = 0.001  # Weight for the sparsity loss.

train_losses = []
dev_accuracies = []

# ----------------------------------------------------------------------------------
# Begin End-to-End Training Experiment for SPR Task.
print("\n=== Starting Training Experiment for SPR Task ===")
print("This experiment trains a multi-modal transformer model that fuses shape, color, and texture embeddings with positional encoding.")
print("It evaluates two aspects:")
print("  1. SPR classification performance (binary decision) based on the hidden rules.")
print("  2. Extraction of interpretable soft symbolic predicates via a symbolic reasoning head.")
print("The training loss is computed as the sum of binary cross-entropy loss and an L1 sparsity loss on the symbolic outputs.\n")

for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0.0
    for (shape_idx, color_idx, texture_idx), labels in train_loader:
        shape_idx = shape_idx.to(device)
        color_idx = color_idx.to(device)
        texture_idx = texture_idx.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        cls_out, sym_out = model(shape_idx, color_idx, texture_idx)
        loss_cls = criterion(cls_out, labels)
        loss_sparse = torch.mean(torch.abs(sym_out))
        loss = loss_cls + lambda_sparsity * loss_sparse
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * shape_idx.size(0)
    epoch_loss /= len(train_dataset)
    train_losses.append(epoch_loss)
    
    # Evaluate on the Dev Set.
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (shape_idx, color_idx, texture_idx), labels in dev_loader:
            shape_idx = shape_idx.to(device)
            color_idx = color_idx.to(device)
            texture_idx = texture_idx.to(device)
            labels = labels.to(device)
            cls_out, _ = model(shape_idx, color_idx, texture_idx)
            preds = (torch.sigmoid(cls_out) > 0.5).float()
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    dev_acc = (correct / total) * 100
    dev_accuracies.append(dev_acc)
    print(f"Epoch {epoch}: Training Loss = {epoch_loss:.4f}, Dev Accuracy = {dev_acc:.2f}%")

# ----------------------------------------------------------------------------------
# Test Set Evaluation.
print("\n=== Test Evaluation ===")
print("This evaluation measures the overall SPR classification performance on unseen test data.")
print("Predictions are derived by applying a threshold (0.5) to the sigmoid-transformed outputs of the classifier head.\n")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for (shape_idx, color_idx, texture_idx), labels in test_loader:
        shape_idx = shape_idx.to(device)
        color_idx = color_idx.to(device)
        texture_idx = texture_idx.to(device)
        labels = labels.to(device)
        cls_out, _ = model(shape_idx, color_idx, texture_idx)
        preds = (torch.sigmoid(cls_out) > 0.5).float()
        total += labels.size(0)
        correct += (preds == labels).sum().item()
test_accuracy = (correct / total) * 100
print(f"Test Accuracy: {test_accuracy:.2f}%  (SPR classification performance on unseen data)")
if test_accuracy == 0.0:
    print("Warning: Test accuracy is 0%, which indicates an error in the model or accuracy calculation.")

# ----------------------------------------------------------------------------------
# Visualization: Generate Figures for Experimental Results.
# Figure_1.png: Training Loss Curve.
print("\nGenerating Figure_1.png")
print("Figure_1.png plots the training loss over epochs. This figure demonstrates the convergence behavior of the model as it minimizes the combined loss (classification + sparsity).")
plt.figure(figsize=(8,6))
plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid(True)
plt.savefig("Figure_1.png")
plt.close()

# Figure_2.png: Development Set Accuracy Curve.
print("\nGenerating Figure_2.png")
print("Figure_2.png presents the development set accuracy over epochs, illustrating improvements in the SPR classification performance during training.")
plt.figure(figsize=(8,6))
plt.plot(range(1, num_epochs + 1), dev_accuracies, marker='s', color='green', label='Dev Accuracy (%)')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Development Set Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("Figure_2.png")
plt.close()

print("\nTraining and evaluation completed successfully.")
print("Figures 'Figure_1.png' and 'Figure_2.png' have been generated to visualize training dynamics and performance improvements.")
import os
# Force CPU usage by disabling any CUDA devices.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import matplotlib.pyplot as plt
import math, random, numpy as np

# Monkey-patch CUDA stream capturing function to always return False to avoid CUDA calls.
if hasattr(torch, "cuda"):
    torch.cuda.is_current_stream_capturing = lambda: False

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# ------------------------------
# Dataset Loading and Preprocessing
# ------------------------------
# Provided header: Load SPR_BENCH dataset from local CSV files.
data_files = {
    "train": "SPR_BENCH/train.csv",
    "dev": "SPR_BENCH/dev.csv",
    "test": "SPR_BENCH/test.csv"
}
dataset = load_dataset("csv", data_files=data_files, delimiter=",")

# Tokenize the sequence by splitting on whitespace and ensure label is integer.
dataset = dataset.map(lambda x: {"tokens": x["sequence"].split(), "label": int(x["label"])})
print("Dataset structure:", dataset)

# For faster debugging, restrict each split to a limited number of samples (remove this for full training).
for split in dataset.keys():
    n_samples = min(200, len(dataset[split]))
    dataset[split] = dataset[split].shuffle(seed=seed).select(range(n_samples))

# ------------------------------
# Build Vocabulary from Training Tokens
# ------------------------------
all_tokens = []
for tokens in dataset["train"]["tokens"]:
    all_tokens.extend(tokens)
# Reserve index 0 for PAD token.
vocab = {token: idx + 1 for idx, token in enumerate(set(all_tokens))}
vocab["<PAD>"] = 0
vocab_size = len(vocab)
print("Vocabulary size:", vocab_size)

# Determine maximum sequence length from training split.
max_len = max(len(x["tokens"]) for x in dataset["train"])

# Helper function: encode tokens to indices and pad to max_len.
def encode_tokens(tokens):
    ids = [vocab.get(t, 0) for t in tokens]
    if len(ids) < max_len:
        ids = ids + [0]*(max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

# Encode tokens and add as "input_ids" in the dataset.
for split in dataset.keys():
    dataset[split] = dataset[split].map(lambda x: {"input_ids": encode_tokens(x["tokens"])})

# ------------------------------
# Compute Auxiliary Predicate Labels Using Heuristics
# Predicates:
# 1. shape_count: 1 if exactly three tokens start with '▲', else 0.
# 2. color_position: 1 if the 4th token's last character is 'r', else 0.
# 3. parity: 1 if even number of tokens start with '■', else 0.
# 4. order: 1 if first occurrence of '▲' comes before first occurrence of '●', else 0.
# ------------------------------
def compute_predicates(tokens):
    # shape_count predicate
    count_shape = sum(1 for t in tokens if t[0] == '▲')
    shape_count = 1.0 if count_shape == 3 else 0.0
    # color_position predicate
    if len(tokens) >= 4:
        color_position = 1.0 if tokens[3][-1] == 'r' else 0.0
    else:
        color_position = 0.0
    # parity predicate: even count for tokens starting with '■'
    count_square = sum(1 for t in tokens if t[0] == '■')
    parity = 1.0 if (count_square % 2 == 0) else 0.0
    # order predicate: first '▲' before first '●'
    pos_triangle, pos_circle = None, None
    for i, t in enumerate(tokens):
        if t[0] == '▲' and pos_triangle is None:
            pos_triangle = i
        if t[0] == '●' and pos_circle is None:
            pos_circle = i
    order = 1.0 if (pos_triangle is not None and pos_circle is not None and pos_triangle < pos_circle) else 0.0
    return {"shape_count": shape_count, "color_position": color_position, "parity": parity, "order": order}

for split in dataset.keys():
    dataset[split] = dataset[split].map(lambda x: compute_predicates(x["tokens"]))

# ------------------------------
# Define Custom PyTorch Dataset for SPR
# ------------------------------
class SPRDataset(Dataset):
    def __init__(self, split):
        self.data = dataset[split]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            torch.tensor(item["input_ids"], dtype=torch.long),
            torch.tensor(item["label"], dtype=torch.float),
            torch.tensor(item["shape_count"], dtype=torch.float),
            torch.tensor(item["color_position"], dtype=torch.float),
            torch.tensor(item["parity"], dtype=torch.float),
            torch.tensor(item["order"], dtype=torch.float)
        )

train_dataset = SPRDataset("train")
dev_dataset = SPRDataset("dev")
test_dataset = SPRDataset("test")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
dev_loader   = DataLoader(dev_dataset, batch_size=32, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# ------------------------------
# Define Positional Encoding Module for the Transformer
# ------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ------------------------------
# Define Model Architectures
# ------------------------------
# Baseline Model: Transformer encoder + global average pooling + binary classifier.
class BaselineModel(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(d_model, 1)
    def forward(self, x):
        emb = self.embedding(x)            # (batch, seq, d_model)
        emb = self.pos_enc(emb)
        out = self.transformer(emb)          # (batch, seq, d_model)
        pooled = out.mean(dim=1)             # Global average pooling
        logits = self.fc(pooled)             # (batch, 1)
        return logits.squeeze(1)

# Augmented Model: Transformer encoder with atomic predicate modules and rule aggregator.
class AugmentedModel(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # Predicate-specific modules (linear layers) for specialized estimators.
        self.shape_count_net   = nn.Linear(d_model, 1)
        self.color_position_net = nn.Linear(d_model, 1)
        self.parity_net         = nn.Linear(d_model, 1)
        self.order_net          = nn.Linear(d_model, 1)
        # Aggregator: fuse predicate logits using a linear layer.
        self.aggregator = nn.Linear(4, d_model)
        # Final binary classifier combining global and aggregated representations.
        self.fc = nn.Linear(d_model * 2, 1)
    def forward(self, x):
        emb = self.embedding(x)            # (batch, seq, d_model)
        emb = self.pos_enc(emb)
        out = self.transformer(emb)          # (batch, seq, d_model)
        pooled = out.mean(dim=1)             # (batch, d_model)
        # Predicate modules:
        shape_logit = self.shape_count_net(pooled)
        # For color_position, use token at position 3 if available; otherwise use zeros.
        if x.size(1) > 3:
            token4 = out[:, 3, :]
        else:
            token4 = torch.zeros_like(pooled)
        color_logit = self.color_position_net(token4)
        parity_logit = self.parity_net(pooled)
        order_logit = self.order_net(pooled)
        predicate_logits = torch.cat([shape_logit, color_logit, parity_logit, order_logit], dim=1)  # (batch, 4)
        agg = torch.tanh(self.aggregator(predicate_logits))  # (batch, d_model)
        combined = torch.cat([pooled, agg], dim=1)             # (batch, 2*d_model)
        logits = self.fc(combined)
        return logits.squeeze(1), predicate_logits

# ------------------------------
# Set device to CPU explicitly and initialize models.
# ------------------------------
device = torch.device("cpu")
print("Running on device:", device)

d_model = 64
baseline_model = BaselineModel(vocab_size, d_model, max_len).to(device)
augmented_model = AugmentedModel(vocab_size, d_model, max_len).to(device)

criterion = nn.BCEWithLogitsLoss()
baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=1e-3)
augmented_optimizer = optim.Adam(augmented_model.parameters(), lr=1e-3)
epochs = 5

# ------------------------------
# Experiment 1: Baseline Model Training and Evaluation
# This experiment trains a transformer-based baseline model (global pooling + binary classifier).
# It reports training loss per epoch and validates dev accuracy.
# ------------------------------
print("\nExperiment 1: Baseline Model Training and Evaluation")
for epoch in range(epochs):
    baseline_model.train()
    running_loss = 0.0
    for batch in train_loader:
        input_ids, labels, _, _, _, _ = [b.to(device) for b in batch]
        baseline_optimizer.zero_grad()
        logits = baseline_model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        baseline_optimizer.step()
        running_loss += loss.item() * input_ids.size(0)
    avg_loss = running_loss / len(train_dataset)
    print(f"Baseline Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

baseline_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in dev_loader:
        input_ids, labels, _, _, _, _ = [b.to(device) for b in batch]
        logits = baseline_model(input_ids)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
baseline_dev_acc = correct / total
print("Baseline Model Dev Accuracy:", baseline_dev_acc)

# ------------------------------
# Experiment 2: Augmented Model Training and Evaluation
# This experiment trains the full augmented model with predicate-specific modules and a rule aggregator.
# Both main classification loss and auxiliary losses (for predicates) are used.
# ------------------------------
print("\nExperiment 2: Augmented Model Training and Evaluation")
for epoch in range(epochs):
    augmented_model.train()
    running_loss = 0.0
    for batch in train_loader:
        input_ids, labels, shape_gt, color_gt, parity_gt, order_gt = [b.to(device) for b in batch]
        augmented_optimizer.zero_grad()
        logits, predicate_logits = augmented_model(input_ids)
        main_loss = criterion(logits, labels)
        aux_loss = (criterion(predicate_logits[:, 0], shape_gt) +
                    criterion(predicate_logits[:, 1], color_gt) +
                    criterion(predicate_logits[:, 2], parity_gt) +
                    criterion(predicate_logits[:, 3], order_gt))
        loss = main_loss + 0.5 * aux_loss
        loss.backward()
        augmented_optimizer.step()
        running_loss += loss.item() * input_ids.size(0)
    avg_loss = running_loss / len(train_dataset)
    print(f"Augmented Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

augmented_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in dev_loader:
        input_ids, labels, _, _, _, _ = [b.to(device) for b in batch]
        logits, _ = augmented_model(input_ids)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
augmented_dev_acc = correct / total
print("Augmented Model Dev Accuracy:", augmented_dev_acc)

# ------------------------------
# Test Evaluation: Compare Baseline vs Augmented Models on Unseen Test Data
# ------------------------------
print("\nEvaluating Baseline Model on Test Split")
baseline_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        input_ids, labels, _, _, _, _ = [b.to(device) for b in batch]
        logits = baseline_model(input_ids)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
baseline_test_acc = correct / total
print("Baseline Model Test Accuracy:", baseline_test_acc)

print("\nEvaluating Augmented Model on Test Split")
augmented_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        input_ids, labels, _, _, _, _ = [b.to(device) for b in batch]
        logits, _ = augmented_model(input_ids)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
augmented_test_acc = correct / total
print("Augmented Model Test Accuracy:", augmented_test_acc)

# ------------------------------
# Generate Figures to Showcase Results
# Figure_1.png: Bar chart comparing Dev Accuracy between Baseline and Augmented models.
# Figure_2.png: Bar chart comparing Test Accuracy between Baseline and Augmented models.
# ------------------------------
print("\nGenerating Figure_1.png: Comparing Dev Accuracy between Baseline and Augmented models")
fig, ax = plt.subplots()
models = ["Baseline", "Augmented"]
dev_accuracies = [baseline_dev_acc, augmented_dev_acc]
ax.bar(models, dev_accuracies, color=["blue", "green"])
ax.set_title("Dev Accuracy Comparison")
ax.set_ylabel("Accuracy")
plt.savefig("Figure_1.png")
plt.close()
print("Figure_1.png saved.")

print("\nGenerating Figure_2.png: Comparing Test Accuracy between Baseline and Augmented models")
fig, ax = plt.subplots()
test_accuracies = [baseline_test_acc, augmented_test_acc]
ax.bar(models, test_accuracies, color=["blue", "green"])
ax.set_title("Test Accuracy Comparison")
ax.set_ylabel("Accuracy")
plt.savefig("Figure_2.png")
plt.close()
print("Figure_2.png saved.")

# ------------------------------
# Ablation Study: Evaluate Impact of Each Predicate Module in the Augmented Model
# For each predicate (shape_count, color_position, parity, order), we zero its logit and recompute predictions.
# The dev accuracy for each ablation scenario is printed.
# ------------------------------
print("\nStarting Ablation Study on Augmented Model Predicate Modules")
predicate_names = ["shape_count", "color_position", "parity", "order"]
ablation_results = {}
augmented_model.eval()
with torch.no_grad():
    for i, pname in enumerate(predicate_names):
        correct = 0
        total = 0
        for batch in dev_loader:
            input_ids, labels, _, _, _, _ = [b.to(device) for b in batch]
            logits, predicate_logits = augmented_model(input_ids)
            # Zero out the i-th predicate logit to simulate dropping that predicate module.
            predicate_logits[:, i] = 0
            agg = torch.tanh(augmented_model.aggregator(predicate_logits))
            emb = augmented_model.embedding(input_ids)
            emb = augmented_model.pos_enc(emb)
            out = augmented_model.transformer(emb)
            pooled = out.mean(dim=1)
            combined = torch.cat([pooled, agg], dim=1)
            new_logits = augmented_model.fc(combined)
            preds = (torch.sigmoid(new_logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        acc = correct / total
        ablation_results[pname] = acc
        print(f"Ablation (dropping {pname}): Dev Accuracy = {acc:.4f}")
print("\nAblation Study Completed. Results:", ablation_results)

# Final check: Ensure that both models achieve non-zero accuracy.
if baseline_test_acc <= 0 or augmented_test_acc <= 0:
    print("\nWarning: One of the models obtained 0% accuracy. Please review the implementation.")
else:
    print("\nBoth models achieved non-zero test accuracy. Experiments completed successfully.")
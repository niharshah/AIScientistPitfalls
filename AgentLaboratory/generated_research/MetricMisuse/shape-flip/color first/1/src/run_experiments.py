import os
# Force CPU-only computation.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# Disable CUDA functions to avoid inadvertent CUDA calls.
import torch
torch.cuda.is_current_stream_capturing = lambda: False
torch.cuda.is_available = lambda: False
torch.cuda.current_device = lambda: 0

import pathlib
from datasets import load_dataset, DatasetDict
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility.
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ----------------------------
# Data Loading and Processing
# ----------------------------
data_folder = pathlib.Path("./SPR_BENCH/")
data_files = {
    "train": str(data_folder / "train.csv"),
    "dev": str(data_folder / "dev.csv"),
    "test": str(data_folder / "test.csv")
}

spr_dataset = DatasetDict({
    split: load_dataset("csv", data_files=path, split="train", cache_dir=".cache_dsets")
    for split, path in data_files.items()
})

# Lambda functions to compute unique color and shape counts.
color_count = lambda seq: len({token[1] for token in seq.strip().split() if len(token) > 1})
shape_count = lambda seq: len({token[0] for token in seq.strip().split() if token})

# Add computed features.
for split in spr_dataset.keys():
    spr_dataset[split] = spr_dataset[split].map(lambda ex: {
        "color_variety": color_count(ex["sequence"]),
        "shape_variety": shape_count(ex["sequence"])
    })

print("Training sample with computed features:")
print(spr_dataset["train"][0])

# Use device=CPU.
device = torch.device("cpu")

# Token dictionaries.
shape_to_idx = {"▲": 0, "■": 1, "●": 2, "◆": 3}
color_to_idx = {"r": 0, "g": 1, "b": 2, "y": 3}

# Hyperparameters.
embed_dim = 16               # Embedding dimension for shape & color.
total_embed_dim = embed_dim    # Final token embedding dimension.
max_seq_length = 30          # Max sequence length for positional embeddings.
num_heads = 2                # Number of transformer attention heads.
num_transformer_layers = 1   # Single transformer encoder layer.
batch_size = 32
num_epochs = 1               # One epoch for demonstration.
lr = 1e-3

# Process dataset splits into lists.
def process_split(split_dataset, limit=None):
    shapes, colors, labels, lengths, color_vars, shape_vars = [], [], [], [], [], []
    count = 0
    for ex in split_dataset:
        if limit is not None and count >= limit:
            break
        tokens = ex["sequence"].strip().split()
        s_ids = []
        c_ids = []
        for token in tokens:
            s_ids.append(shape_to_idx.get(token[0], 0))
            # Default to 'r' if color missing.
            c_ids.append(color_to_idx.get(token[1] if len(token)>1 else 'r', 0))
        shapes.append(torch.tensor(s_ids, dtype=torch.long))
        colors.append(torch.tensor(c_ids, dtype=torch.long))
        labels.append(torch.tensor(float(ex["label"]), dtype=torch.float))
        lengths.append(len(tokens))
        color_vars.append(torch.tensor(ex["color_variety"], dtype=torch.float))
        shape_vars.append(torch.tensor(ex["shape_variety"], dtype=torch.float))
        count += 1
    return shapes, colors, labels, lengths, color_vars, shape_vars

# Use a subset for training for fast demonstration; full dev and test.
train_shapes, train_colors, train_labels, train_lengths, _, _ = process_split(spr_dataset["train"], limit=500)
dev_shapes, dev_colors, dev_labels, dev_lengths, dev_colorvars, dev_shapevars = process_split(spr_dataset["dev"])
test_shapes, test_colors, test_labels, test_lengths, test_colorvars, test_shapevars = process_split(spr_dataset["test"])

# ----------------------------
# R-NSR Model: Neural-Symbolic Transformer with Sparse Rule Extraction
# ----------------------------
class RNSRModel(nn.Module):
    def __init__(self, vocab_shape_size, vocab_color_size, embed_dim, total_embed_dim, max_seq_length, num_heads, num_layers):
        super(RNSRModel, self).__init__()
        self.shape_emb = nn.Embedding(vocab_shape_size, embed_dim)
        self.color_emb = nn.Embedding(vocab_color_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_length, total_embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=total_embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Sparse concept extraction layer with 4 candidate predicates.
        self.sparse_layer = nn.Linear(total_embed_dim, 4)
        # Learnable bias for differentiable symbolic reasoning.
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, shape_indices, color_indices, seq_lengths):
        # shape_indices, color_indices: [B, L]
        batch_size, seq_len = shape_indices.shape
        shape_embeds = self.shape_emb(shape_indices)   # [B, L, embed_dim]
        color_embeds = self.color_emb(color_indices)     # [B, L, embed_dim]
        token_embeds = shape_embeds + color_embeds         # [B, L, embed_dim]
        # Add positional embeddings.
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        pos_embeds = self.pos_embedding(positions)
        token_embeds = token_embeds + pos_embeds          # [B, L, embed_dim]
        # Transformer expects (L, B, D)
        transformer_input = token_embeds.transpose(0, 1)   # [L, B, D]
        transformer_output = self.transformer_encoder(transformer_input)  # [L, B, D]
        transformer_output = transformer_output.transpose(0, 1)           # [B, L, D]
        # Mean pooling over valid tokens.
        pooled = []
        for i in range(batch_size):
            valid_len = seq_lengths[i]
            pooled.append(transformer_output[i, :valid_len, :].mean(dim=0))
        pooled = torch.stack(pooled, dim=0)  # [B, D]
        # Sparse concept extraction.
        sparse_logits = self.sparse_layer(pooled)  # [B, 4]
        sparse_activations = torch.sigmoid(sparse_logits)  # Candidate predicate activations.
        # Differentiable symbolic reasoning: soft AND via product plus bias.
        eps = 1e-6
        sparse_activations = torch.clamp(sparse_activations, min=eps, max=1.0)
        soft_and = torch.prod(sparse_activations, dim=1) + self.bias  # [B]
        final_prob = torch.sigmoid(soft_and).unsqueeze(1)  # [B, 1]
        return final_prob, sparse_activations

model = RNSRModel(
    vocab_shape_size=len(shape_to_idx),
    vocab_color_size=len(color_to_idx),
    embed_dim=embed_dim,
    total_embed_dim=total_embed_dim,
    max_seq_length=max_seq_length,
    num_heads=num_heads,
    num_layers=num_transformer_layers
).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

print("\nStarting training of R-NSR model on SPR_BENCH dataset (CPU-only).")
print("This experiment aims to demonstrate joint learning of dense representations with a transformer encoder integrated with a sparse concept extraction layer and symbolic reasoning. The model is trained on 500 examples for 1 epoch.")

# Training loop for R-NSR.
train_losses = []
num_train = len(train_shapes)
for epoch in range(num_epochs):
    epoch_loss = 0.0
    batch_count = 0
    indices = list(range(num_train))
    random.shuffle(indices)
    for start in range(0, num_train, batch_size):
        batch_idx = indices[start:start+batch_size]
        # Prepare batch.
        batch_shape_list = [train_shapes[i] for i in batch_idx]
        batch_color_list = [train_colors[i] for i in batch_idx]
        batch_label_tensor = torch.stack([train_labels[i] for i in batch_idx]).to(device).unsqueeze(1)
        batch_seq_lengths = [train_lengths[i] for i in batch_idx]
        max_len = max(batch_seq_lengths)
        padded_shapes = torch.stack([F.pad(seq, (0, max_len - seq.size(0)), "constant", 0) for seq in batch_shape_list]).to(device)
        padded_colors = torch.stack([F.pad(seq, (0, max_len - seq.size(0)), "constant", 0) for seq in batch_color_list]).to(device)
        
        model.train()
        optimizer.zero_grad()
        outputs, sparse_acts = model(padded_shapes, padded_colors, batch_seq_lengths)
        loss = criterion(outputs, batch_label_tensor)
        # L1 penalty on sparse activations.
        l1_penalty = 0.001 * torch.norm(sparse_acts, 1)
        total_loss = loss + l1_penalty
        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss.item()
        batch_count += 1
    avg_loss = epoch_loss / batch_count
    train_losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Average Loss: {avg_loss:.4f}")

# ----------------------------
# Evaluation of R-NSR Model
# ----------------------------
print("\n=== Evaluation on Dev Set for R-NSR Model ===")
print("This evaluation shows predictions versus ground truth on the Dev set using a threshold of 0.5 on model outputs.")
all_dev_preds = []
all_dev_labels = []
num_dev = len(dev_shapes)
dev_indices = list(range(num_dev))
for start in range(0, num_dev, batch_size):
    batch_idx = dev_indices[start:start+batch_size]
    b_shapes = [dev_shapes[i] for i in batch_idx]
    b_colors = [dev_colors[i] for i in batch_idx]
    b_labels = torch.stack([dev_labels[i] for i in batch_idx]).to(device).unsqueeze(1)
    b_seq_lengths = [dev_lengths[i] for i in batch_idx]
    max_len = max(b_seq_lengths)
    padded_shapes = torch.stack([F.pad(seq, (0, max_len - seq.size(0)), "constant", 0) for seq in b_shapes]).to(device)
    padded_colors = torch.stack([F.pad(seq, (0, max_len - seq.size(0)), "constant", 0) for seq in b_colors]).to(device)
    model.eval()
    with torch.no_grad():
        outputs, _ = model(padded_shapes, padded_colors, b_seq_lengths)
        preds = (outputs >= 0.5).float()
    all_dev_preds.append(preds.cpu())
    all_dev_labels.append(b_labels.cpu())
dev_preds_tensor = torch.cat(all_dev_preds)
dev_labels_tensor = torch.cat(all_dev_labels)
dev_accuracy = (dev_preds_tensor == dev_labels_tensor).float().mean().item() * 100.0
print(f"Dev Accuracy: {dev_accuracy:.2f}%")

print("\n=== Evaluation on Test Set for R-NSR Model ===")
print("This evaluation reports overall Test Accuracy and also computes Color-Weighted Accuracy (CWA) and Shape-Weighted Accuracy (SWA).\nThe weights are based respectively on the diversity of colors and shapes in each sequence.")
all_test_preds = []
all_test_labels = []
num_test = len(test_shapes)
test_indices = list(range(num_test))
for start in range(0, num_test, batch_size):
    batch_idx = test_indices[start:start+batch_size]
    b_shapes = [test_shapes[i] for i in batch_idx]
    b_colors = [test_colors[i] for i in batch_idx]
    b_labels = torch.stack([test_labels[i] for i in batch_idx]).to(device).unsqueeze(1)
    b_seq_lengths = [test_lengths[i] for i in batch_idx]
    max_len = max(b_seq_lengths)
    padded_shapes = torch.stack([F.pad(seq, (0, max_len - seq.size(0)), "constant", 0) for seq in b_shapes]).to(device)
    padded_colors = torch.stack([F.pad(seq, (0, max_len - seq.size(0)), "constant", 0) for seq in b_colors]).to(device)
    model.eval()
    with torch.no_grad():
        outputs, _ = model(padded_shapes, padded_colors, b_seq_lengths)
        preds = (outputs >= 0.5).float()
    all_test_preds.append(preds.cpu())
    all_test_labels.append(b_labels.cpu())
test_preds_tensor = torch.cat(all_test_preds)
test_labels_tensor = torch.cat(all_test_labels)
test_accuracy = (test_preds_tensor == test_labels_tensor).float().mean().item() * 100.0
print(f"Standard Test Accuracy: {test_accuracy:.2f}%")

# Compute Color-Weighted Accuracy (CWA) and Shape-Weighted Accuracy (SWA).
test_preds_np = test_preds_tensor.numpy()
test_labels_np = test_labels_tensor.numpy()
cwa_num, cwa_den, swa_num, swa_den = 0.0, 0.0, 0.0, 0.0
for i in range(len(test_preds_np)):
    weight_color = float(test_colorvars[i])
    weight_shape = float(test_shapevars[i])
    correct = 1.0 if test_preds_np[i] == test_labels_np[i] else 0.0
    cwa_num += weight_color * correct
    cwa_den += weight_color
    swa_num += weight_shape * correct
    swa_den += weight_shape
CWA = (cwa_num / cwa_den) * 100 if cwa_den > 0 else 0.0
SWA = (swa_num / swa_den) * 100 if swa_den > 0 else 0.0
print(f"Color-Weighted Accuracy (CWA): {CWA:.2f}% (SOTA: 65.0%)")
print(f"Shape-Weighted Accuracy (SWA): {SWA:.2f}% (SOTA: 70.0%)")

# ----------------------------
# Generate Figures for R-NSR Experiment
# ----------------------------
# Figure 1: Training Loss Curve.
plt.figure()
plt.plot(range(1, num_epochs+1), train_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Average Training Loss")
plt.title("Figure_1.png: R-NSR Training Loss Curve")
plt.grid(True)
plt.savefig("Figure_1.png")
plt.close()

# Figure 2: Bar Chart comparing Test Accuracy metrics.
metrics = ["Standard Accuracy", "CWA", "SWA"]
values = [test_accuracy, CWA, SWA]
plt.figure()
bars = plt.bar(metrics, values, color=["blue", "green", "red"])
plt.ylim(0, 100)
plt.ylabel("Accuracy (%)")
plt.title("Figure_2.png: R-NSR Test Accuracy Metrics Comparison")
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha="center", fontweight="bold")
plt.savefig("Figure_2.png")
plt.close()

print("\nFigures generated: Figure_1.png (Training Loss Curve) and Figure_2.png (R-NSR Test Accuracy Metrics).")

# ----------------------------
# Ablation Experiment: Transformer Baseline (without sparse & symbolic modules)
# ----------------------------
print("\nStarting Ablation Experiment: Transformer Baseline (without sparse and symbolic reasoning modules).")
print("This experiment uses a standard Transformer encoder model that lacks the sparse extraction and symbolic reasoning layers. It is trained on the same 500 examples for 1 epoch and evaluated on the Test set.")

# Define Transformer Baseline model.
class TransformerBaseline(nn.Module):
    def __init__(self, vocab_shape_size, vocab_color_size, embed_dim, total_embed_dim, max_seq_length, num_heads, num_layers):
        super(TransformerBaseline, self).__init__()
        self.shape_emb = nn.Embedding(vocab_shape_size, embed_dim)
        self.color_emb = nn.Embedding(vocab_color_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_length, total_embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=total_embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Final linear classification layer.
        self.linear = nn.Linear(total_embed_dim, 1)
        
    def forward(self, shape_indices, color_indices, seq_lengths):
        batch_size, seq_len = shape_indices.shape
        shape_embeds = self.shape_emb(shape_indices)   # [B, L, embed_dim]
        color_embeds = self.color_emb(color_indices)     # [B, L, embed_dim]
        token_embeds = shape_embeds + color_embeds         # [B, L, embed_dim]
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        pos_embeds = self.pos_embedding(positions)
        token_embeds = token_embeds + pos_embeds          # [B, L, embed_dim]
        transformer_input = token_embeds.transpose(0, 1)   # [L, B, D]
        transformer_output = self.transformer_encoder(transformer_input)  # [L, B, D]
        transformer_output = transformer_output.transpose(0, 1)           # [B, L, D]
        # Mean pooling over valid tokens.
        pooled = []
        for i in range(batch_size):
            valid_len = seq_lengths[i]
            pooled.append(transformer_output[i, :valid_len, :].mean(dim=0))
        pooled = torch.stack(pooled, dim=0)  # [B, D]
        logits = self.linear(pooled)          # [B, 1]
        prob = torch.sigmoid(logits)
        return prob

baseline_model = TransformerBaseline(
    vocab_shape_size=len(shape_to_idx),
    vocab_color_size=len(color_to_idx),
    embed_dim=embed_dim,
    total_embed_dim=total_embed_dim,
    max_seq_length=max_seq_length,
    num_heads=num_heads,
    num_layers=num_transformer_layers
).to(device)

criterion_base = nn.BCELoss()
optimizer_base = optim.Adam(baseline_model.parameters(), lr=lr)

baseline_train_losses = []
num_train = len(train_shapes)
for epoch in range(num_epochs):
    epoch_loss = 0.0
    batch_count = 0
    indices = list(range(num_train))
    random.shuffle(indices)
    for start in range(0, num_train, batch_size):
        batch_idx = indices[start:start+batch_size]
        batch_shape_list = [train_shapes[i] for i in batch_idx]
        batch_color_list = [train_colors[i] for i in batch_idx]
        batch_label_tensor = torch.stack([train_labels[i] for i in batch_idx]).to(device).unsqueeze(1)
        batch_seq_lengths = [train_lengths[i] for i in batch_idx]
        max_len = max(batch_seq_lengths)
        padded_shapes = torch.stack([F.pad(seq, (0, max_len - seq.size(0)), "constant", 0) for seq in batch_shape_list]).to(device)
        padded_colors = torch.stack([F.pad(seq, (0, max_len - seq.size(0)), "constant", 0) for seq in batch_color_list]).to(device)
        
        baseline_model.train()
        optimizer_base.zero_grad()
        outputs = baseline_model(padded_shapes, padded_colors, batch_seq_lengths)
        loss = criterion_base(outputs, batch_label_tensor)
        loss.backward()
        optimizer_base.step()
        epoch_loss += loss.item()
        batch_count += 1
    avg_loss = epoch_loss / batch_count
    baseline_train_losses.append(avg_loss)
    print(f"Ablation Epoch [{epoch+1}/{num_epochs}] - Average Loss: {avg_loss:.4f}")

print("\n=== Evaluation on Test Set for Transformer Baseline (Ablation) ===")
all_test_preds_base = []
all_test_labels_base = []
num_test = len(test_shapes)
test_indices = list(range(num_test))
for start in range(0, num_test, batch_size):
    batch_idx = test_indices[start:start+batch_size]
    b_shapes = [test_shapes[i] for i in batch_idx]
    b_colors = [test_colors[i] for i in batch_idx]
    b_labels = torch.stack([test_labels[i] for i in batch_idx]).to(device).unsqueeze(1)
    b_seq_lengths = [test_lengths[i] for i in batch_idx]
    max_len = max(b_seq_lengths)
    padded_shapes = torch.stack([F.pad(seq, (0, max_len - seq.size(0)), "constant", 0) for seq in b_shapes]).to(device)
    padded_colors = torch.stack([F.pad(seq, (0, max_len - seq.size(0)), "constant", 0) for seq in b_colors]).to(device)
    baseline_model.eval()
    with torch.no_grad():
        outputs = baseline_model(padded_shapes, padded_colors, b_seq_lengths)
        preds = (outputs >= 0.5).float()
    all_test_preds_base.append(preds.cpu())
    all_test_labels_base.append(b_labels.cpu())
test_preds_tensor_base = torch.cat(all_test_preds_base)
test_labels_tensor_base = torch.cat(all_test_labels_base)
baseline_test_accuracy = (test_preds_tensor_base == test_labels_tensor_base).float().mean().item() * 100.0
print(f"Transformer Baseline Test Accuracy: {baseline_test_accuracy:.2f}%")

# Generate Figure 3: Transformer Baseline Training Loss Curve.
plt.figure()
plt.plot(range(1, num_epochs+1), baseline_train_losses, marker="o", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Average Training Loss")
plt.title("Figure_3.png: Transformer Baseline Training Loss Curve")
plt.grid(True)
plt.savefig("Figure_3.png")
plt.close()

print("\nAblation experiment completed. Figure_3.png (Transformer Baseline Training Loss Curve) generated.")
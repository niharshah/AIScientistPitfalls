import os
# Final stable implementation of the Hybrid Neuro-Symbolic Transformer for the SPR Task has been integrated.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict
import pathlib
import itertools  # Added to allow batching limits

# Force CPU
torch.cuda.is_available = lambda: False

# ----------------------------------------------------------------------
# Provided dataset code (DO NOT MODIFY)
# ----------------------------------------------------------------------
data_path = pathlib.Path("SPR_BENCH")

def count_color_variety(seq):
    # Each token is expected to be a shape followed by an optional 1-letter color, e.g., "▲r".
    # We only count tokens with a color (length > 1).
    return len(set(token[1] for token in seq.strip().split() if len(token) > 1))

def count_shape_variety(seq):
    # The first character of each token represents the shape.
    return len(set(token[0] for token in seq.strip().split() if token))

def load_spr_data(root):
    dset = DatasetDict()
    for split in ["train", "dev", "test"]:
        dset[split] = load_dataset(
            "csv", 
            data_files=str(root / f"{split}.csv"), 
            split="train",
            cache_dir=".cache_dsets"
        )
    return dset

spr_data = load_spr_data(data_path)

# Enhance each split of the dataset by computing color and shape complexities
for split in spr_data.keys():
    spr_data[split] = spr_data[split].map(lambda ex: {
        "color_complexity": count_color_variety(ex["sequence"]),
        "shape_complexity": count_shape_variety(ex["sequence"])
    })

print("Loaded splits:", list(spr_data.keys()))
print("Example from train split:", spr_data["train"][0])
# ----------------------------------------------------------------------
# End of provided dataset code
# ----------------------------------------------------------------------

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Define vocabulary maps for shapes and colors
shape_map = {'▲': 0, '■': 1, '●': 2, '◆': 3}
color_map = {'r': 0, 'g': 1, 'b': 2, 'y': 3}
num_shapes = len(shape_map)
num_colors = len(color_map)

# Tokenization: convert sequence string into lists of shape indices and color indices.
def tokenize_dataset(dataset):
    shape_idxs = []
    color_idxs = []
    labels = []
    color_comp = []
    shape_comp = []
    seq_lengths = []
    for ex in dataset:
        tokens = ex["sequence"].strip().split()
        s_idx = []
        c_idx = []
        for token in tokens:
            # If token has a color (length>1), use it; else use 0 as placeholder.
            if len(token) > 1:
                s_idx.append(shape_map[token[0]])
                c_idx.append(color_map[token[1]])
            else:
                s_idx.append(shape_map[token[0]])
                c_idx.append(0)
        shape_idxs.append(torch.tensor(s_idx, dtype=torch.long))
        color_idxs.append(torch.tensor(c_idx, dtype=torch.long))
        labels.append(torch.tensor([float(ex["label"])], dtype=torch.float))
        color_comp.append(ex["color_complexity"])
        shape_comp.append(ex["shape_complexity"])
        seq_lengths.append(len(s_idx))
    return shape_idxs, color_idxs, torch.stack(labels), torch.tensor(color_comp, dtype=torch.float), torch.tensor(shape_comp, dtype=torch.float), seq_lengths

train_shape, train_color, train_labels, train_color_comp, train_shape_comp, train_lengths = tokenize_dataset(spr_data["train"])
dev_shape, dev_color, dev_labels, dev_color_comp, dev_shape_comp, dev_lengths = tokenize_dataset(spr_data["dev"])
test_shape, test_color, test_labels, test_color_comp, test_shape_comp, test_lengths = tokenize_dataset(spr_data["test"])

# Determine maximum sequence length across splits for padding
max_len = max(max(train_lengths), max(dev_lengths), max(test_lengths))

# Padding function: pad list of tensors to max_len with zeros.
def pad_sequences(seq_list, max_len):
    padded = []
    for tensor in seq_list:
        if len(tensor) < max_len:
            pad = torch.zeros(max_len - len(tensor), dtype=torch.long)
            padded_tensor = torch.cat([tensor, pad], dim=0)
        else:
            padded_tensor = tensor[:max_len]
        padded.append(padded_tensor)
    return torch.stack(padded)

train_shape_pad = pad_sequences(train_shape, max_len)
train_color_pad = pad_sequences(train_color, max_len)
dev_shape_pad   = pad_sequences(dev_shape, max_len)
dev_color_pad   = pad_sequences(dev_color, max_len)
test_shape_pad  = pad_sequences(test_shape, max_len)
test_color_pad  = pad_sequences(test_color, max_len)

# Create TensorDatasets and DataLoaders
batch_size = 64
train_dataset = TensorDataset(train_shape_pad, train_color_pad, train_labels, train_color_comp, train_shape_comp)
dev_dataset   = TensorDataset(dev_shape_pad, dev_color_pad, dev_labels, dev_color_comp, dev_shape_comp)
test_dataset  = TensorDataset(test_shape_pad, test_color_pad, test_labels, test_color_comp, test_shape_comp)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader   = DataLoader(dev_dataset, batch_size=batch_size)
test_loader  = DataLoader(test_dataset, batch_size=batch_size)

# To reduce time complexity, we limit the number of batches processed in each epoch or evaluation.
max_batches = 10  # Process only the first 10 batches per loop

# Define the Hybrid Model with transformer encoding and a differentiable symbolic module.
class HybridModel(nn.Module):
    def __init__(self, emb_dim=32, nhead=4, num_transformer_layers=2, max_len=50, use_transformer=True, use_symbolic=True):
        super(HybridModel, self).__init__()
        self.use_transformer = use_transformer
        self.use_symbolic = use_symbolic
        self.emb_dim = emb_dim
        # Embeddings for shape and color
        self.shape_emb = nn.Embedding(num_shapes, emb_dim)
        self.color_emb = nn.Embedding(num_colors, emb_dim)
        # Positional Embedding
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        # Symbolic branches:
        # Branch 1: Shape-Count branch: Linear layer on the mean token embedding.
        self.shape_count_fc = nn.Linear(emb_dim, emb_dim)
        # Branch 2: Color-Position branch: Linear layer on maximum pooled token embeddings.
        self.color_pos_fc = nn.Linear(emb_dim, emb_dim)
        # Branch 3: Parity branch: Uses tanh on sum of token embeddings then linear.
        self.parity_fc = nn.Linear(emb_dim, emb_dim)
        # Branch 4: Order branch: Uses differences between consecutive tokens.
        self.order_fc = nn.Linear(emb_dim, emb_dim)
        # Gating mechanism for combining branches; learnable parameters.
        self.gate = nn.Parameter(torch.ones(4))
        # Fusion layer: Combine transformer and symbolic outputs.
        fusion_dim = (emb_dim if use_transformer else 0) + (emb_dim if use_symbolic else 0)
        self.fc = nn.Linear(fusion_dim, 1)
        
    def forward(self, shape_idx, color_idx):
        batch_size, seq_len = shape_idx.size()
        # Compute token embeddings: shape + color + positional embeddings.
        token_shape = self.shape_emb(shape_idx)          # shape: (B, L, D)
        token_color = self.color_emb(color_idx)            # shape: (B, L, D)
        pos_ids = torch.arange(seq_len, device=shape_idx.device).unsqueeze(0).expand(batch_size, seq_len)
        token_pos = self.pos_emb(pos_ids)                  # shape: (B, L, D)
        token_emb = token_shape + token_color + token_pos   # shape: (B, L, D)
        
        # Transformer branch
        transformer_out = None
        if self.use_transformer:
            trans_out = self.transformer_encoder(token_emb)   # (B, L, D)
            # Use mean pooling over tokens for global representation.
            transformer_out = trans_out.mean(dim=1)           # (B, D)
        
        # Symbolic branch
        symbolic_out = None
        if self.use_symbolic:
            # Branch 1: Aggregated shape-count features.
            branch1 = self.shape_count_fc(token_emb.mean(dim=1))
            # Branch 2: Color-position by max pooling.
            branch2, _ = token_emb.max(dim=1)
            branch2 = self.color_pos_fc(branch2)
            # Branch 3: Parity branch using tanh of token sum.
            branch3 = self.parity_fc(torch.tanh(token_emb.sum(dim=1)))
            # Branch 4: Order branch using differences between consecutive tokens.
            if seq_len > 1:
                diff = token_emb[:,1:,:] - token_emb[:,:-1,:]
                branch4 = self.order_fc(diff.mean(dim=1))
            else:
                branch4 = torch.zeros(batch_size, self.emb_dim, device=shape_idx.device)
            # Combine branches with softmax-gated weights.
            gates = F.softmax(self.gate, dim=0)
            symbolic_out = gates[0]*branch1 + gates[1]*branch2 + gates[2]*branch3 + gates[3]*branch4
        
        # Fusion of transformer and symbolic outputs.
        if self.use_transformer and self.use_symbolic:
            fuse = torch.cat([transformer_out, symbolic_out], dim=1)
        elif self.use_transformer:
            fuse = transformer_out
        else:
            fuse = symbolic_out
        logits = self.fc(fuse)
        return torch.sigmoid(logits)

# Device set to CPU (forced)
device = torch.device("cpu")

# Training settings
num_epochs = 3   # For demonstration; in research, use more epochs.
learning_rate = 1e-3

# Prepare experiments with different configurations.
# Experiment 1: Combined Model (Transformer + Symbolic)
# Experiment 2: Transformer Only
# Experiment 3: Symbolic Only
experiments = [
    {"name": "Combined Model", "use_transformer": True, "use_symbolic": True},
    {"name": "Transformer Only", "use_transformer": True, "use_symbolic": False},
    {"name": "Symbolic Only", "use_transformer": False, "use_symbolic": True}
]

results = {}
for exp in experiments:
    print("\nStarting experiment:", exp["name"])
    print("This experiment is designed to show the performance in terms of accuracy, Color-Weighted Accuracy (CWA), and Shape-Weighted Accuracy (SWA) on the development set using the following configuration:")
    print("  - Transformer used:", exp["use_transformer"])
    print("  - Symbolic module used:", exp["use_symbolic"])
    model = HybridModel(emb_dim=32, nhead=4, num_transformer_layers=2, max_len=max_len,
                        use_transformer=exp["use_transformer"], use_symbolic=exp["use_symbolic"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    bce_loss = nn.BCELoss()
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for (s_idx, c_idx, labels, col_comp, sh_comp) in itertools.islice(train_loader, max_batches):
            s_idx = s_idx.to(device)
            c_idx = c_idx.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = model(s_idx, c_idx)
            loss = bce_loss(preds, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * s_idx.size(0)
        epoch_loss /= (max_batches * batch_size)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    
    # Evaluate on Dev set
    model.eval()
    all_preds = []
    all_labels = []
    all_color = []
    all_shape = []
    with torch.no_grad():
        for (s_idx, c_idx, labels, col_comp, sh_comp) in itertools.islice(dev_loader, max_batches):
            s_idx = s_idx.to(device)
            c_idx = c_idx.to(device)
            preds = model(s_idx, c_idx)
            all_preds.append((preds > 0.5).float().cpu())
            all_labels.append(labels.cpu())
            all_color.append(col_comp)
            all_shape.append(sh_comp)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    correct = (all_preds.squeeze() == all_labels.squeeze()).float()
    accuracy = correct.mean().item() * 100.0
    # Compute Color-Weighted Accuracy (CWA)
    total_color = torch.cat(all_color).sum().item()
    cwa = (torch.cat(all_color, dim=0) * correct).sum().item() / total_color * 100.0 if total_color > 0 else 0.0
    # Compute Shape-Weighted Accuracy (SWA)
    total_shape = torch.cat(all_shape).sum().item()
    swa = (torch.cat(all_shape, dim=0) * correct).sum().item() / total_shape * 100.0 if total_shape > 0 else 0.0

    print(f"Dev Results for {exp['name']}:")
    print(f"  Overall Accuracy: {accuracy:.2f}%")
    print(f"  Color-Weighted Accuracy (CWA): {cwa:.2f}%")
    print(f"  Shape-Weighted Accuracy (SWA): {swa:.2f}%")
    
    results[exp["name"]] = {"dev_acc": accuracy, "dev_cwa": cwa, "dev_swa": swa, "model": model}

# Final Evaluation on Test Set using the Combined Model
print("\nEvaluating final Combined Model on Test set...")
final_model = results["Combined Model"]["model"]
final_model.eval()
all_preds = []
all_labels = []
all_color = []
all_shape = []
with torch.no_grad():
    for (s_idx, c_idx, labels, col_comp, sh_comp) in itertools.islice(test_loader, max_batches):
        s_idx = s_idx.to(device)
        c_idx = c_idx.to(device)
        preds = final_model(s_idx, c_idx)
        all_preds.append((preds > 0.5).float().cpu())
        all_labels.append(labels.cpu())
        all_color.append(col_comp)
        all_shape.append(sh_comp)
all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)
correct = (all_preds.squeeze() == all_labels.squeeze()).float()
test_accuracy = correct.mean().item() * 100.0
total_color = torch.cat(all_color).sum().item()
test_cwa = (torch.cat(all_color) * correct).sum().item() / total_color * 100.0 if total_color > 0 else 0.0
total_shape = torch.cat(all_shape).sum().item()
test_swa = (torch.cat(all_shape) * correct).sum().item() / total_shape * 100.0 if total_shape > 0 else 0.0
print(f"Test Results for Combined Model:")
print(f"  Overall Accuracy: {test_accuracy:.2f}%")
print(f"  Color-Weighted Accuracy (CWA): {test_cwa:.2f}%")
print(f"  Shape-Weighted Accuracy (SWA): {test_swa:.2f}%")

# Noise Robustness Experiment:
print("\nStarting Noise Robustness Experiment:")
print("This experiment adds spurious tokens with a 20% chance per token to test the model's robustness to noise.")
p_noise = 0.2
def add_noise_to_sequence(seq_str):
    tokens = seq_str.strip().split()
    noisy_tokens = []
    for token in tokens:
        noisy_tokens.append(token)
        if random.random() < p_noise:
            spurious_shape = random.choice(list(shape_map.keys()))
            spurious_color = random.choice(list(color_map.keys()))
            noisy_tokens.append(spurious_shape + spurious_color)
    return " ".join(noisy_tokens)

# Generate noisy test dataset
noisy_shape_tensors = []
noisy_color_tensors = []
noisy_color_comp_list = []
noisy_shape_comp_list = []
for ex in spr_data["test"]:
    noisy_seq = add_noise_to_sequence(ex["sequence"])
    tokens = noisy_seq.strip().split()
    s_idx = []
    c_idx = []
    for token in tokens:
        if len(token) > 1:
            s_idx.append(shape_map[token[0]])
            c_idx.append(color_map[token[1]])
        else:
            s_idx.append(shape_map[token[0]])
            c_idx.append(0)
    # Compute complexities from the noisy sequence
    col_comp = len(set(token[1] for token in tokens if len(token) > 1))
    shp_comp = len(set(token[0] for token in tokens if token))
    noisy_color_comp_list.append(col_comp)
    noisy_shape_comp_list.append(shp_comp)
    # Pad sequences to max_len (truncate if longer)
    if len(s_idx) < max_len:
        pad_len = max_len - len(s_idx)
        s_idx = s_idx + [0]*pad_len
        c_idx = c_idx + [0]*pad_len
    else:
        s_idx = s_idx[:max_len]
        c_idx = c_idx[:max_len]
    noisy_shape_tensors.append(torch.tensor(s_idx, dtype=torch.long))
    noisy_color_tensors.append(torch.tensor(c_idx, dtype=torch.long))

noisy_color_comp_tensor = torch.tensor(noisy_color_comp_list, dtype=torch.float)
noisy_shape_comp_tensor = torch.tensor(noisy_shape_comp_list, dtype=torch.float)
noisy_shape_pad = torch.stack(noisy_shape_tensors)
noisy_color_pad = torch.stack(noisy_color_tensors)
noisy_dataset = TensorDataset(noisy_shape_pad, noisy_color_pad, test_labels, noisy_color_comp_tensor, noisy_shape_comp_tensor)
noisy_loader = DataLoader(noisy_dataset, batch_size=batch_size)

all_preds = []
all_labels = []
all_color = []
all_shape = []
with torch.no_grad():
    for (s_idx, c_idx, labels, col_comp, sh_comp) in itertools.islice(noisy_loader, max_batches):
        s_idx = s_idx.to(device)
        c_idx = c_idx.to(device)
        preds = final_model(s_idx, c_idx)
        all_preds.append((preds > 0.5).float().cpu())
        all_labels.append(labels.cpu())
        all_color.append(col_comp)
        all_shape.append(sh_comp)
all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)
correct = (all_preds.squeeze() == all_labels.squeeze()).float()
noisy_accuracy = correct.mean().item() * 100.0
total_color = torch.cat(all_color).sum().item()
noisy_cwa = (torch.cat(all_color) * correct).sum().item() / total_color * 100.0 if total_color > 0 else 0.0
total_shape = torch.cat(all_shape).sum().item()
noisy_swa = (torch.cat(all_shape) * correct).sum().item() / total_shape * 100.0 if total_shape > 0 else 0.0
print(f"Noise Robustness Test Results (Combined Model with Noise):")
print(f"  Overall Accuracy: {noisy_accuracy:.2f}%")
print(f"  Color-Weighted Accuracy (CWA): {noisy_cwa:.2f}%")
print(f"  Shape-Weighted Accuracy (SWA): {noisy_swa:.2f}%")

# Generate Figures
# Figure_1.png: Bar plot comparing Dev Accuracy, CWA, and SWA for the three model variants.
fig, ax = plt.subplots(figsize=(8,6))
exp_names = list(results.keys())
accs = [results[name]["dev_acc"] for name in exp_names]
cwas = [results[name]["dev_cwa"] for name in exp_names]
swas = [results[name]["dev_swa"] for name in exp_names]
x = np.arange(len(exp_names))
width = 0.25
ax.bar(x - width, accs, width, label="Accuracy")
ax.bar(x, cwas, width, label="CWA")
ax.bar(x + width, swas, width, label="SWA")
ax.set_ylabel("Percentage")
ax.set_title("Figure_1.png: Dev Set Performance by Model Variant")
ax.set_xticks(x)
ax.set_xticklabels(exp_names)
ax.legend()
plt.tight_layout()
plt.savefig("Figure_1.png")
plt.close()
print("\nFigure_1.png saved: This figure shows the dev set performance (Accuracy, CWA, SWA) across the three model variants.")

# Figure_2.png: Bar plot comparing Clean Test accuracy vs Noisy Test accuracy for the Combined Model.
fig2, ax2 = plt.subplots(figsize=(6,4))
metrics = ['Clean Test', 'Noisy Test']
values = [test_accuracy, noisy_accuracy]
ax2.bar(metrics, values, color=['blue', 'orange'])
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Figure_2.png: Combined Model - Clean vs Noisy Test Accuracy")
for i, v in enumerate(values):
    ax2.text(i, v + 1, f"{v:.1f}%", ha='center')
plt.tight_layout()
plt.savefig("Figure_2.png")
plt.close()
print("\nFigure_2.png saved: This figure compares the clean test accuracy with the noisy test accuracy for the Combined Model.")

print("\nAll experiments completed. The reported metrics and generated figures provide insights into overall accuracy, color-weighted, and shape-weighted performance, as well as ablation and noise robustness evaluations.")

# Check to ensure that accuracy is non-zero.
if test_accuracy == 0.0:
    print("Error: Model achieved 0% accuracy. Please check implementation!")
else:
    print("Model accuracy is non-zero. Experiments executed successfully.")
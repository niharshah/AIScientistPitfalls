import os
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict

# --------------------------------------------------------------------------------
# Environment Setup: Force usage of CPU to avoid CUDA initialization issues.
# --------------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide any GPU devices.
device = torch.device("cpu")
print("Using device:", device)

# Monkey-patch torch.cuda.is_current_stream_capturing to avoid CUDA initialization error when using CPU.
if device.type == "cpu":
    torch.cuda.is_current_stream_capturing = lambda: False

# --------------------------------------------------------------------------------
# STEP 0: Dataset Loading (Assumed to be preloaded as 'spr_dataset')
# --------------------------------------------------------------------------------
# The following dataset initialization code is assumed to have been executed externally:
# -----------------------------------------------------------------------------
# import pathlib
# from datasets import load_dataset, DatasetDict
#
# spr_path = pathlib.Path("SPR_BENCH")
# spr_dataset = {}
# for split in ["train", "dev", "test"]:
#     csv_file = spr_path / f"{split}.csv"
#     if csv_file.exists():
#         spr_dataset[split] = load_dataset("csv", data_files=str(csv_file), split="train", cache_dir=".cache_dsets")
#         print(f"Loaded {split} split with {len(spr_dataset[split])} records.")
#     else:
#         print(f"WARNING: {csv_file.resolve()} not found. '{split}' split will be skipped.")
#
# if spr_dataset:
#     spr_dataset = DatasetDict(spr_dataset)
#     print("\nDataset splits available:", list(spr_dataset.keys()))
#     if "train" in spr_dataset:
#         example = spr_dataset["train"][0]
#         print("\nFirst training example:")
#         print("ID:", example.get("id", "N/A"))
#         print("Sequence:", example.get("sequence", "N/A"))
#         print("Label:", example.get("label", "N/A"))
# else:
#     print("No dataset splits were loaded. Please ensure the SPR_BENCH folder contains train.csv, dev.csv, and test.csv.")
# -----------------------------------------------------------------------------
if "train" not in spr_dataset:
    raise Exception("Training split not found in spr_dataset. Please check dataset files.")

# --------------------------------------------------------------------------------
# STEP 1: Data Preprocessing
# --------------------------------------------------------------------------------
print("\nStep 1: Data Preprocessing")
# Build a vocabulary of shapes (first character of each token) using the training set.
vocab_set = set()
for sample in spr_dataset["train"]:
    seq = sample.get("sequence", "")
    tokens = seq.split()
    for token in tokens:
        if token:
            vocab_set.add(token[0])
vocab_list = sorted(list(vocab_set))
vocab_size = len(vocab_list)
# Create mapping from shape to index.
shape_to_idx = {}
for idx, shape in enumerate(vocab_list):
    shape_to_idx[shape] = idx
print("Unique Shapes Vocabulary:", shape_to_idx)

# Helper conversion: Convert a sequence string into a bag-of-shapes vector.
# (Since functions are not allowed, we inline the code for each dataset.)
# Process training data.
train_features_list = []
train_labels_list = []
for sample in spr_dataset["train"]:
    seq = sample.get("sequence", "")
    vec = np.zeros(vocab_size, dtype=np.float32)
    for token in seq.split():
        if token:
            vec[shape_to_idx[token[0]]] += 1.0
    train_features_list.append(vec)
    train_labels_list.append(int(sample.get("label", "0")))
train_features = np.vstack(train_features_list)
train_labels = np.array(train_labels_list)

# Process development data (if available).
if "dev" in spr_dataset:
    dev_features_list = []
    dev_labels_list = []
    for sample in spr_dataset["dev"]:
        seq = sample.get("sequence", "")
        vec = np.zeros(vocab_size, dtype=np.float32)
        for token in seq.split():
            if token:
                vec[shape_to_idx[token[0]]] += 1.0
        dev_features_list.append(vec)
        dev_labels_list.append(int(sample.get("label", "0")))
    dev_features = np.vstack(dev_features_list)
    dev_labels = np.array(dev_labels_list)
    print("Processed development data with", len(dev_labels), "samples.")
else:
    print("Dev split not found.")

# Process test data.
if "test" in spr_dataset:
    test_features_list = []
    test_labels_list = []
    test_seqs = []  # Save original sequences for SWA computation.
    for sample in spr_dataset["test"]:
        seq = sample.get("sequence", "")
        test_seqs.append(seq)
        vec = np.zeros(vocab_size, dtype=np.float32)
        for token in seq.split():
            if token:
                vec[shape_to_idx[token[0]]] += 1.0
        test_features_list.append(vec)
        test_labels_list.append(int(sample.get("label", "0")))
    test_features = np.vstack(test_features_list)
    test_labels = np.array(test_labels_list)
    print("Processed test data with", len(test_labels), "samples.")
else:
    print("Test split not found.")

# Convert numpy arrays to torch tensors (ensuring they reside on CPU).
X_train = torch.tensor(train_features).to(device)
y_train = torch.tensor(train_labels, dtype=torch.long).to(device)
if "dev" in spr_dataset:
    X_dev = torch.tensor(dev_features).to(device)
    y_dev = torch.tensor(dev_labels, dtype=torch.long).to(device)
if "test" in spr_dataset:
    X_test = torch.tensor(test_features).to(device)
    y_test = torch.tensor(test_labels, dtype=torch.long).to(device)

# --------------------------------------------------------------------------------
# STEP 2: Model Building and Training
# --------------------------------------------------------------------------------
print("\nStep 2: Model Building and Training")
# We design a simple neural network with one hidden layer.
input_dim = vocab_size
hidden_dim = 16    # Selected small hidden dimension
output_dim = 2     # Binary classification: 0 or 1

model = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, output_dim)
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 20
train_losses = []
best_dev_acc = 0.0
best_model_state = None

print("\nTraining the model on the training split and tuning on the development split.")
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = loss_fn(outputs, y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    
    if "dev" in spr_dataset:
        model.eval()
        with torch.no_grad():
            dev_outputs = model(X_dev)
            _, dev_preds = torch.max(dev_outputs, 1)
            correct = (dev_preds == y_dev).sum().item()
            dev_acc = correct / len(y_dev)
        print(f"Epoch {epoch+1}/{num_epochs}: Training Loss = {loss.item():.4f}, Dev Accuracy = {dev_acc*100:.2f}%")
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_model_state = model.state_dict()
    else:
        print(f"Epoch {epoch+1}/{num_epochs}: Training Loss = {loss.item():.4f}")

if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("\nLoaded best model from dev set with accuracy {:.2f}%.".format(best_dev_acc*100))
else:
    print("\nNo development set available; using final model.")

# Save training loss curve as Figure_1.png.
plt.figure()
plt.plot(range(1, num_epochs+1), train_losses, marker='o')
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("Figure_1.png")
print("\nFigure_1.png saved: Training loss curve showing loss reduction over epochs.")

# --------------------------------------------------------------------------------
# STEP 3: Testing and Evaluation
# --------------------------------------------------------------------------------
print("\nStep 3: Testing and Evaluation")
# This section evaluates the model on the test split.
# Output details include:
# - Standard Test Accuracy: the percentage of correct predictions.
# - Shape-Weighted Accuracy (SWA): each test sample's correctness is weighted by the number of tokens in its sequence.
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    _, test_preds = torch.max(test_outputs, 1)

# Compute standard accuracy.
correct_standard = (test_preds == y_test).sum().item()
standard_acc = correct_standard / len(y_test)

# Compute Shape-Weighted Accuracy (SWA).
total_weight = 0
weighted_correct = 0
for i, seq in enumerate(test_seqs):
    weight = len(seq.split())
    total_weight += weight
    if test_preds[i].item() == y_test[i].item():
        weighted_correct += weight
swa = weighted_correct / total_weight if total_weight > 0 else 0

if standard_acc == 0:
    raise Exception("Error: 0% standard accuracy obtained. Please check model training or accuracy calculation.")

print("\nDetailed Experimental Results:")
print(" - Standard Test Accuracy: {:.2f}%".format(standard_acc*100))
print(" - Shape-Weighted Accuracy (SWA): {:.2f}%".format(swa*100))

# --------------------------------------------------------------------------------
# STEP 4: Performance Comparison Visualization
# --------------------------------------------------------------------------------
# We now create a bar chart comparing our model's SWA with an assumed SOTA baseline of 75%.
sota_swa = 0.75

plt.figure()
bars = plt.bar(['SOTA Baseline', 'Our Model'], [sota_swa*100, swa*100], color=['gray', 'blue'])
plt.ylabel('Shape-Weighted Accuracy (%)')
plt.title('Performance Comparison on SPR_BENCH (Test Set)')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.2f}%', ha='center', va='bottom')
plt.savefig("Figure_2.png")
print("\nFigure_2.png saved: Bar chart comparing the SOTA baseline (75% SWA) with our model's SWA.")

print("\nFinal Results Summary:")
print("Our model achieved a Standard Test Accuracy of {:.2f}% and a Shape-Weighted Accuracy of {:.2f}% on the SPR_BENCH test set."
      .format(standard_acc*100, swa*100))
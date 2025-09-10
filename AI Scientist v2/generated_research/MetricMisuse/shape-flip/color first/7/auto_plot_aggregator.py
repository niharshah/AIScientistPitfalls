#!/usr/bin/env python3
"""
Final Aggregator Script for GNN for SPR Research Paper

This script aggregates and visualizes final experimental results from multiple
experiment summaries: Baseline, Research, Sequential-Only-Graph, No Positional Embedding,
and No Batch Normalization. All final figures are saved in the "figures/" directory.
Each plotting block is wrapped in a try-except block so that one failure does not affect the others.
The final figures consolidate key numbers and detailed plots from the .npy data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Set publication style: high dpi and larger fonts; remove top/right spines.
plt.rcParams.update({'font.size': 14, 'figure.dpi': 300})

def style_ax(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

# Create figures directory
os.makedirs("figures", exist_ok=True)

# --------------------------
# Helper functions
# --------------------------
def count_color_variety(sequence: str) -> int:
    return len({token[1:] for token in sequence.split() if len(token) > 1})

def count_shape_variety(sequence: str) -> int:
    return len({token[0] for token in sequence.split() if token})

def complexity_weight(sequence: str) -> int:
    return count_color_variety(sequence) + count_shape_variety(sequence)

def compute_test_metrics(ds: dict):
    try:
        preds = np.array(ds["predictions"])
        gts = np.array(ds["ground_truth"])
        seqs = np.array(ds["sequences"]) if "sequences" in ds else []
    except Exception as e:
        print("Error extracting test arrays:", e)
        return None, None
    test_acc = (preds == gts).mean() if preds.size else float('nan')
    if len(seqs):
        weights = np.array([complexity_weight(seq) for seq in seqs])
        test_cowa = (weights * (preds == gts)).sum() / weights.sum() if weights.sum() > 0 else float('nan')
    else:
        test_cowa = float('nan')
    return test_acc, test_cowa

def load_and_extract(identifier: str, file_path: str):
    """
    Load the .npy file and extract the dataset dictionary for plotting.
    For Baseline and Research experiments, the structure is stored under key "num_epochs".
    For Sequential-Only-Graph, it is stored under key "SPR_BENCH".
    For No Positional Embedding, under key "NoPosEmb" then "SPR_BENCH".
    For No Batch Normalization, under key "no_batch_norm" then "SPR_BENCH".
    """
    try:
        data = np.load(file_path, allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading file for {identifier}: {e}")
        return None

    ds = None
    if identifier in ["Baseline", "Research"]:
        if "num_epochs" in data:
            ds_name = list(data["num_epochs"].keys())[0]
            ds = data["num_epochs"][ds_name]
        else:
            ds = data.get("SPR_BENCH", None)
    elif identifier == "Sequential-Only-Graph":
        ds = data.get("SPR_BENCH", None)
    elif identifier == "NoPosEmb":
        ds = data.get("NoPosEmb", {}).get("SPR_BENCH", None)
    elif identifier == "NoBatchNorm":
        ds = data.get("no_batch_norm", {}).get("SPR_BENCH", None)
    else:
        ds = None

    if ds is None:
        print(f"Dataset not found in {identifier} file.")
    return ds

# --------------------------
# Experiment file paths (exact full paths from the JSON summaries)
# --------------------------
experiments = {
    "Baseline": "experiment_results/experiment_a9fd589082d44825a6e6a4e1e104d461_proc_1490727/experiment_data.npy",
    "Research": "experiment_results/experiment_5ce9dbd6c23b4847bc8edf8bca1e2ce1_proc_1494830/experiment_data.npy",
    "Sequential-Only-Graph": "experiment_results/experiment_d2e5b465334e456cac454c054bda5522_proc_1497845/experiment_data.npy",
    "NoPosEmb": "experiment_results/experiment_e93eb6632e7a41d6b2866d8cd7f8a62c_proc_1497844/experiment_data.npy",
    "NoBatchNorm": "experiment_results/experiment_70f36fbce73242cf934bef122691d7a8_proc_1497847/experiment_data.npy"
}

# Load each experiment dataset
exp_data = {}
for key, path in experiments.items():
    ds = load_and_extract(key, path)
    if ds is not None:
        exp_data[key] = ds

# --------------------------
# Group 1: Aggregated Per-Epoch Comparison for Baseline, Research, and Sequential-Only-Graph
# Create one figure with three subplots: Loss, Accuracy, and Complexity-Weighted Accuracy (CoWA)
# --------------------------
group1_keys = ["Baseline", "Research", "Sequential-Only-Graph"]
try:
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"Baseline": "tab:blue", "Research": "tab:orange", "Sequential-Only-Graph": "tab:green"}
    # Subplot 1: Loss Curves
    for key in group1_keys:
        ds = exp_data.get(key, None)
        if ds is None: 
            continue
        epochs = range(1, len(ds["losses"]["train"]) + 1)
        axs[0].plot(epochs, ds["losses"]["train"], label=f"{key} Train", color=colors[key], linestyle="-", marker="o")
        axs[0].plot(epochs, ds["losses"]["val"], label=f"{key} Validation", color=colors[key], linestyle="--", marker="s")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Per-Epoch Loss Curves")
    axs[0].legend()
    style_ax(axs[0])
    
    # Subplot 2: Accuracy Curves
    for key in group1_keys:
        ds = exp_data.get(key, None)
        if ds is None:
            continue
        epochs = range(1, len(ds["losses"]["train"]) + 1)
        train_acc = [m["acc"] for m in ds["metrics"]["train"]]
        val_acc = [m["acc"] for m in ds["metrics"]["val"]]
        axs[1].plot(epochs, train_acc, label=f"{key} Train", color=colors[key], linestyle="-", marker="o")
        axs[1].plot(epochs, val_acc, label=f"{key} Validation", color=colors[key], linestyle="--", marker="s")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_title("Per-Epoch Accuracy Curves")
    axs[1].legend()
    style_ax(axs[1])
    
    # Subplot 3: Complexity-Weighted Accuracy (CoWA) Curves (Validation)
    for key in group1_keys:
        ds = exp_data.get(key, None)
        if ds is None:
            continue
        epochs = range(1, len(ds["losses"]["train"]) + 1)
        val_cowa = [m.get("cowa", m.get("CompWA", float('nan'))) for m in ds["metrics"]["val"]]
        axs[2].plot(epochs, val_cowa, label=f"{key} Validation", color=colors[key], linestyle="-", marker="o")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Complexity Weighted Accuracy")
    axs[2].set_title("Per-Epoch CoWA Curves (Validation)")
    axs[2].legend()
    style_ax(axs[2])
    
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "group1_per_epoch_curves.png"))
    plt.close(fig)
except Exception as e:
    print("Error creating aggregated per-epoch curves:", e)
    plt.close()

# --------------------------
# Group 2: Group1 Test Metrics Bar Chart
# Compare final test Accuracy and Test CoWA for Baseline, Research, and Sequential-Only-Graph.
# --------------------------
try:
    labels = []
    test_acc_vals = []
    test_cowa_vals = []
    for key in group1_keys:
        ds = exp_data.get(key, None)
        if ds is None:
            continue
        acc, cowa = compute_test_metrics(ds)
        if acc is None:
            continue
        labels.append(key)
        test_acc_vals.append(acc)
        test_cowa_vals.append(cowa)
    if labels:
        x = range(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(7,5))
        ax.bar([i - width/2 for i in x], test_acc_vals, width, label="Test Accuracy", color="tab:blue")
        ax.bar([i + width/2 for i in x], test_cowa_vals, width, label="Test Complexity Weighted Accuracy", color="tab:green")
        ax.set_xlabel("Experiment")
        ax.set_ylabel("Metric Value")
        ax.set_ylim(0, 1)
        ax.set_title("Test Metrics Comparison")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.legend()
        style_ax(ax)
        fig.tight_layout()
        fig.savefig(os.path.join("figures", "group1_test_metrics.png"))
        plt.close(fig)
except Exception as e:
    print("Error creating Group1 test metrics bar chart:", e)
    plt.close()

# --------------------------
# Group 3: Aggregated Confusion Matrices for Ablation Studies
# Combine No Positional Embedding and No Batch Normalization confusion matrices into one figure (2 subplots)
# --------------------------
ablation_keys = ["NoPosEmb", "NoBatchNorm"]
try:
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for i, key in enumerate(ablation_keys):
        ds = exp_data.get(key, None)
        if ds is None:
            continue
        preds = np.array(ds["predictions"])
        gts = np.array(ds["ground_truth"])
        if preds.size == 0 or gts.size == 0:
            continue
        n_classes = int(max(gts.max(), preds.max()) + 1)
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(gts, preds):
            cm[int(t), int(p)] += 1
        im = axs[i].imshow(cm, cmap="Blues")
        for m in range(n_classes):
            for n in range(n_classes):
                axs[i].text(n, m, str(cm[m, n]), ha="center", va="center", color="black")
        axs[i].set_xticks(range(n_classes))
        axs[i].set_yticks(range(n_classes))
        axs[i].set_xticklabels([f"Pred {j}" for j in range(n_classes)])
        axs[i].set_yticklabels([f"True {j}" for j in range(n_classes)])
        axs[i].set_xlabel("Predicted")
        axs[i].set_ylabel("Ground Truth")
        # Use full descriptive title without underscores
        if key == "NoPosEmb":
            axs[i].set_title("No Positional Embedding Confusion Matrix")
        elif key == "NoBatchNorm":
            axs[i].set_title("No Batch Normalization Confusion Matrix")
        style_ax(axs[i])
        fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "ablation_confusion_matrices.png"))
    plt.close(fig)
except Exception as e:
    print("Error creating ablation confusion matrices figure:", e)
    plt.close()

print("Final aggregated figures are saved in the 'figures/' directory.")
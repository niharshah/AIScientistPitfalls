#!/usr/bin/env python3
"""
Final Aggregator Script for Neural Symbolic Zero-Shot SPR Final Results

This script loads experiment data from .npy files and produces a final set of scientifically 
informative, publication‐quality figures (no more than 12 in total). All figures are saved in 
the "figures/" directory. The plots use key numerical values from the experiments and generate 
detailed curves for loss, accuracy, shape‐weighted accuracy, and test metrics. Figure titles, 
axis labels, and legends are descriptive and without underscores.

All plotting routines check for the presence of expected data before plotting and aggregate 
related plots into one figure when suitable. Each plotting block is wrapped in a try-except block 
so that failures in one do not prevent the others.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Create the figures directory
os.makedirs("figures", exist_ok=True)

# ---------- Helper Functions ----------

def load_experiment_data(npy_path):
    """Load experiment data from .npy file."""
    try:
        data = np.load(npy_path, allow_pickle=True).item()
        return data
    except Exception as e:
        print(f"Error loading {npy_path}: {e}")
        return None

def format_label(label):
    """Remove underscores and replace them with spaces."""
    return label.replace("_", " ")

def save_fig(fig, filename):
    fig.tight_layout()
    fig.savefig(os.path.join("figures", filename), dpi=300)
    plt.close(fig)

# ---------- Baseline Figures ----------
baseline_path = "experiment_results/experiment_095fbd9db36b47aa9825a43a3082d675_proc_2637260/experiment_data.npy"
baseline_data = load_experiment_data(baseline_path)

if baseline_data is not None and "learning_rate" in baseline_data:
    lrs = sorted(baseline_data["learning_rate"].keys(), key=float)
    # Figure 1: Baseline Loss Curves for Multiple Learning Rates
    try:
        fig, ax = plt.subplots(figsize=(8,6))
        for lr in lrs:
            d = baseline_data["learning_rate"][lr]
            if "losses" in d and "train" in d["losses"] and "val" in d["losses"]:
                ax.plot(d["losses"]["train"], label=f"Train LR {lr}")
                ax.plot(d["losses"]["val"], label=f"Val LR {lr}", linestyle="--")
        ax.set_title("Baseline Loss Curves (Train vs Validation)", fontsize=14)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Cross Entropy Loss", fontsize=12)
        ax.legend(fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        save_fig(fig, "Baseline Loss Curves.png")
    except Exception as e:
        print("Baseline Loss Curves error:", e)
    
    # Figure 2: Baseline Accuracy Curves for Multiple Learning Rates
    try:
        fig, ax = plt.subplots(figsize=(8,6))
        for lr in lrs:
            d = baseline_data["learning_rate"][lr]
            if "metrics" in d and "train" in d["metrics"] and "val" in d["metrics"]:
                ax.plot(d["metrics"]["train"], label=f"Train LR {lr}")
                ax.plot(d["metrics"]["val"], label=f"Val LR {lr}", linestyle="--")
        ax.set_title("Baseline Accuracy Curves (Train vs Validation)", fontsize=14)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.legend(fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        save_fig(fig, "Baseline Accuracy Curves.png")
    except Exception as e:
        print("Baseline Accuracy Curves error:", e)
    
    # Figure 3: Baseline Test Metrics Comparison (Bar Chart using key numbers)
    try:
        x = np.arange(len(lrs))
        width = 0.25
        rgs = [baseline_data["learning_rate"][lr]["test_metrics"].get("rgs", 0) for lr in lrs]
        swa = [baseline_data["learning_rate"][lr]["test_metrics"].get("swa", 0) for lr in lrs]
        cwa = [baseline_data["learning_rate"][lr]["test_metrics"].get("cwa", 0) for lr in lrs]
        fig, ax = plt.subplots(figsize=(8,6))
        ax.bar(x - width, rgs, width=width, label="RGS")
        ax.bar(x, swa, width=width, label="SWA")
        ax.bar(x + width, cwa, width=width, label="CWA")
        ax.set_xticks(x)
        ax.set_xticklabels([str(lr) for lr in lrs])
        ax.set_title("Baseline Final Test Metrics: RGS vs SWA vs CWA", fontsize=14)
        ax.set_xlabel("Learning Rate", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.legend(fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        save_fig(fig, "Baseline Test Metrics.png")
    except Exception as e:
        print("Baseline Test Metrics error:", e)

# ---------- Research Figures ----------
research_path = "experiment_results/experiment_708b1a9e10a94bad9603a262d9c9d248_proc_2637983/experiment_data.npy"
research_data = load_experiment_data(research_path)

if research_data is not None:
    # For research, we assume one dataset entry; aggregate loss, accuracy, and SWA into a 1x3 subplot figure.
    try:
        # Pick the first key from research_data
        ds_name = list(research_data.keys())[0]
        ds = research_data[ds_name]
        fig, axes = plt.subplots(1, 3, figsize=(18,5))
        # Loss curves
        if "losses" in ds and "train" in ds["losses"] and "val" in ds["losses"]:
            axes[0].plot(ds["losses"]["train"], label="Train")
            axes[0].plot(ds["losses"]["val"], label="Validation", linestyle="--")
            axes[0].set_title(f"{format_label(ds_name)} Loss Curves", fontsize=14)
            axes[0].set_xlabel("Epoch", fontsize=12)
            axes[0].set_ylabel("Cross Entropy Loss", fontsize=12)
            axes[0].legend(fontsize=10)
            axes[0].spines["top"].set_visible(False)
            axes[0].spines["right"].set_visible(False)
        # Accuracy curves
        if "metrics" in ds and "train" in ds["metrics"] and "val" in ds["metrics"]:
            axes[1].plot(ds["metrics"]["train"], label="Train")
            axes[1].plot(ds["metrics"]["val"], label="Validation", linestyle="--")
            axes[1].set_title(f"{format_label(ds_name)} Accuracy Curves", fontsize=14)
            axes[1].set_xlabel("Epoch", fontsize=12)
            axes[1].set_ylabel("Accuracy", fontsize=12)
            axes[1].legend(fontsize=10)
            axes[1].spines["top"].set_visible(False)
            axes[1].spines["right"].set_visible(False)
        # SWA curves
        if "swa" in ds and "train" in ds["swa"] and "val" in ds["swa"]:
            axes[2].plot(ds["swa"]["train"], label="Train")
            axes[2].plot(ds["swa"]["val"], label="Validation", linestyle="--")
            axes[2].set_title(f"{format_label(ds_name)} SWA Curves", fontsize=14)
            axes[2].set_xlabel("Epoch", fontsize=12)
            axes[2].set_ylabel("Shape Weighted Accuracy", fontsize=12)
            axes[2].legend(fontsize=10)
            axes[2].spines["top"].set_visible(False)
            axes[2].spines["right"].set_visible(False)
        save_fig(fig, f"{format_label(ds_name)} Final Curves.png")
    except Exception as e:
        print("Research Figures error:", e)

# ---------- Ablation Figures ----------
# We aggregate ablation experiments into two aggregated figures:
#   1. A figure with train/val curves for experiments that have key 'metrics', 'losses', and 'swa'
#   2. A figure for confusion matrices if available
ablation_files = {
    "Color Blind Encoder": "experiment_results/experiment_4ce2f6ce02b64cc799307a868499a742_proc_2640122/experiment_data.npy",
    "Histogram Free Symbolic": "experiment_results/experiment_ac18d1382e574817908b9a127a0d5fb9_proc_2640123/experiment_data.npy",
    "Bag of Embeddings": "experiment_results/experiment_acf36273c51c416c8cea41785bb7460b_proc_2640124/experiment_data.npy",
    "Token Only Model": "experiment_results/experiment_54bd630616724b9e802e60b11508cf2d_proc_2640121/experiment_data.npy",
    "Shape Blind Encoder": "experiment_results/experiment_32fc23a1846c454394abe056c1342910_proc_2640123/experiment_data.npy",
    "Multi Synthetic Training": "experiment_results/experiment_1c862a0169fb4227b71c16c61ab71d66_proc_2640124/experiment_data.npy",
    "Scalar Free Symbolic": "experiment_results/experiment_bd8b09b7ca394e8f82deb0577793f03a_proc_2640122/experiment_data.npy"
}

# Prepare aggregated curves (up to 7, but check each has necessary keys)
curve_plots = []
for label, path in ablation_files.items():
    d = load_experiment_data(path)
    # For ablation studies, try to locate the "spr bench" key if present
    if d is None:
        continue
    if "spr_bench" in d:
        d = d["spr_bench"]
    # Check for required keys in this ablation study
    if "metrics" in d and "losses" in d and "swa" in d:
        curve_plots.append((label, d))
        
# To keep final unique plots below 12, we will aggregate these curves in a multi-panel figure.
if curve_plots:
    try:
        n = len(curve_plots)
        # We create one figure with 3 columns per ablation experiment (if n <= 4, one row; else 2 rows)
        cols = 3
        rows = n  # one row per experiment (each row has 3 subplots for accuracy, loss, swa)
        fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*3))
        if rows == 1:
            axes = [axes]  # make it iterable
        for i, (label, d) in enumerate(curve_plots):
            # Check existence of keys
            ax_acc = axes[i][0]
            ax_loss = axes[i][1]
            ax_swa  = axes[i][2]
            epochs = range(1, len(d["metrics"]["train"]) + 1)
            ax_acc.plot(epochs, d["metrics"]["train"], label="Train")
            ax_acc.plot(epochs, d["metrics"]["val"], label="Validation", linestyle="--")
            ax_acc.set_title(f"{label} Accuracy Curves", fontsize=12)
            ax_acc.set_xlabel("Epoch", fontsize=10)
            ax_acc.set_ylabel("Accuracy", fontsize=10)
            ax_acc.legend(fontsize=8)
            ax_acc.spines["top"].set_visible(False)
            ax_acc.spines["right"].set_visible(False)
            
            ax_loss.plot(epochs, d["losses"]["train"], label="Train")
            ax_loss.plot(epochs, d["losses"]["val"], label="Validation", linestyle="--")
            ax_loss.set_title(f"{label} Loss Curves", fontsize=12)
            ax_loss.set_xlabel("Epoch", fontsize=10)
            ax_loss.set_ylabel("Cross Entropy Loss", fontsize=10)
            ax_loss.legend(fontsize=8)
            ax_loss.spines["top"].set_visible(False)
            ax_loss.spines["right"].set_visible(False)
            
            ax_swa.plot(epochs, d["swa"]["train"], label="Train")
            ax_swa.plot(epochs, d["swa"]["val"], label="Validation", linestyle="--")
            ax_swa.set_title(f"{label} SWA Curves", fontsize=12)
            ax_swa.set_xlabel("Epoch", fontsize=10)
            ax_swa.set_ylabel("SWA", fontsize=10)
            ax_swa.legend(fontsize=8)
            ax_swa.spines["top"].set_visible(False)
            ax_swa.spines["right"].set_visible(False)
        save_fig(fig, "Ablation Curves Aggregated.png")
    except Exception as e:
        print("Ablation curves aggregation error:", e)

# For confusion matrices, aggregate those ablations that have validation predictions and ground truth.
cm_plots = []
for label, path in ablation_files.items():
    d = load_experiment_data(path)
    if d is None:
        continue
    if "spr_bench" in d:
        d = d["spr_bench"]
    if "predictions" in d and "ground_truth" in d:
        if "val" in d["predictions"] and "val" in d["ground_truth"]:
            cm_plots.append((label, d))
            
if cm_plots:
    try:
        # Create a figure with one confusion matrix per experiment (arranged in 2 columns)
        n = len(cm_plots)
        cols = 2
        rows = (n + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5))
        axes = axes.flatten() if n > 1 else [axes]
        for i, (label, d) in enumerate(cm_plots):
            preds = np.array(d["predictions"]["val"])
            gts = np.array(d["ground_truth"]["val"])
            num_cls = int(max(gts.max(), preds.max()) + 1)
            cm = np.zeros((num_cls, num_cls), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            ax = axes[i]
            im = ax.imshow(cm, cmap="Blues")
            ax.set_title(f"{label} Confusion Matrix (Validation)", fontsize=12)
            ax.set_xlabel("Predicted", fontsize=10)
            ax.set_ylabel("True", fontsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # Hide any extra subplots
        for j in range(i+1, len(axes)):
            axes[j].axis("off")
        save_fig(fig, "Ablation Confusion Matrices.png")
    except Exception as e:
        print("Ablation confusion matrices aggregation error:", e)

print("Final plots have been saved in the 'figures' directory.")
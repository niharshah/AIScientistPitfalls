#!/usr/bin/env python3
"""
Final Aggregator Script for Interpretable Neural Rule Learning Results

This script loads experiment numpy data files from previously-run experiments,
combines and aggregates the final scientific figures from:
  • Baseline experiments
  • Research experiments
  • Ablation studies

All final figures are saved into the "figures" directory at 300 dpi with enlarged
font sizes. All labels, titles, and legends use spaces (no underscores) and include
informative, descriptive names. No additional comments or notes appear on the figures.

The script aggregates related plots into combined figures where suitable and produces
no more than 12 unique, nonduplicated plots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})
os.makedirs("figures", exist_ok=True)

def load_exp_data(path):
    try:
        data = np.load(path, allow_pickle=True).item()
        # If the data is wrapped under "SPR_BENCH", extract that.
        if isinstance(data, dict) and "SPR_BENCH" in data:
            return data["SPR_BENCH"]
        return data
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return {}

def get_array(d, key, subkey=None):
    try:
        if subkey:
            return np.asarray(d.get(key, {}).get(subkey, []))
        else:
            return np.asarray(d.get(key, []))
    except Exception:
        return np.array([])

def plot_subplots(fig, axs):
    # Remove top and right spines from all axes
    for ax in axs.flatten():
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

def compute_confusion_matrix(preds, gts):
    if preds.size and gts.size and preds.shape == gts.shape:
        n_cls = int(max(preds.max(), gts.max())) + 1
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for p, t in zip(preds, gts):
            cm[t, p] += 1
        return cm
    return None

# ---------------------------------------------------------------------------
# Exact experiment file paths (as given in summaries)
exp_files = {
    "Baseline": "experiment_results/experiment_95dbf5af5c80492d9c949248fc35830e_proc_3198575/experiment_data.npy",
    "Research": "experiment_results/experiment_c9210a1fbb4d478194939a388c3eed1a_proc_3211631/experiment_data.npy",
    "No Gate Ensemble": "experiment_results/experiment_7f137d05c10e4b88a62be45d4ddc56bb_proc_3218403/experiment_data.npy",
    "No Rule Sparsity": "experiment_results/experiment_3f73a5f15525498abad0c0cf3e6e23de_proc_3218404/experiment_data.npy",
    "Rule Free CNN": "experiment_results/experiment_b992ba63c3c344d2b227416dc16c222f_proc_3218406/experiment_data.npy",
    "No Gate Confidence": "experiment_results/experiment_0bce7edbb25b4e71a0f87dcd38eab683_proc_3218405/experiment_data.npy",
    "Rule Only Head": "experiment_results/experiment_130ce571a04940d0b1b299f74ca8032d_proc_3218404/experiment_data.npy",
    "Static Scalar Gate": "experiment_results/experiment_f752f1a9af1448748bfac6b0aa3c82d1_proc_3218406/experiment_data.npy"
}

data_exps = {}
for name, path in exp_files.items():
    data_exps[name] = load_exp_data(path)

# ---------------------------------------------------------------------------
# Plot 1: Baseline Combined Accuracy and Loss Curves (Subplots)
try:
    baseline = data_exps.get("Baseline", {})
    train_acc = get_array(baseline, "metrics", "train_acc")
    val_acc = get_array(baseline, "metrics", "val_acc")
    train_loss = get_array(baseline, "losses", "train")
    val_loss = get_array(baseline, "losses", "val")
    if train_acc.size and val_acc.size and train_loss.size and val_loss.size:
        epochs = np.arange(1, len(train_acc) + 1)
        fig, axs = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
        axs[0].plot(epochs, train_acc, marker='o', label="Train Accuracy")
        axs[0].plot(epochs, val_acc, marker='o', label="Validation Accuracy")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Accuracy")
        axs[0].set_title("Baseline Experiment: Accuracy")
        axs[0].legend()
        axs[1].plot(epochs, train_loss, marker='o', label="Train Loss")
        axs[1].plot(epochs, val_loss, marker='o', label="Validation Loss")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Cross Entropy Loss")
        axs[1].set_title("Baseline Experiment: Loss")
        axs[1].legend()
        plot_subplots(fig, axs)
        fig.tight_layout()
        fig.savefig(os.path.join("figures", "baseline_accuracy_loss.png"))
        plt.close(fig)
except Exception as e:
    print("Error plotting Baseline Combined Accuracy and Loss:", e)

# Plot 2: Baseline Test Set Confusion Matrix
try:
    baseline = data_exps.get("Baseline", {})
    preds = np.asarray(baseline.get("predictions", []))
    gts = np.asarray(baseline.get("ground_truth", []))
    cm = compute_confusion_matrix(preds, gts)
    if cm is not None:
        fig = plt.figure(dpi=300)
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, label="Count")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Baseline Experiment: Test Set Confusion Matrix")
        plt.tight_layout()
        fig.savefig(os.path.join("figures", "baseline_confusion_matrix.png"))
        plt.close(fig)
except Exception as e:
    print("Error plotting Baseline Confusion Matrix:", e)

# Plot 3: Research Combined Accuracy and Loss Curves (Subplots)
try:
    research = data_exps.get("Research", {})
    train_acc = get_array(research, "metrics", "train_acc")
    val_acc = get_array(research, "metrics", "val_acc")
    train_loss = get_array(research, "losses", "train")
    val_loss = get_array(research, "losses", "val")
    if train_acc.size and val_acc.size and train_loss.size and val_loss.size:
        epochs = np.arange(1, len(train_acc) + 1)
        fig, axs = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
        axs[0].plot(epochs, train_acc, marker='o', label="Train Accuracy")
        axs[0].plot(epochs, val_acc, marker='o', label="Validation Accuracy")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Accuracy")
        axs[0].set_title("Research Experiment: Accuracy")
        axs[0].legend()
        axs[1].plot(epochs, train_loss, marker='o', label="Train Loss")
        axs[1].plot(epochs, val_loss, marker='o', label="Validation Loss")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Cross Entropy Loss")
        axs[1].set_title("Research Experiment: Loss")
        axs[1].legend()
        plot_subplots(fig, axs)
        fig.tight_layout()
        fig.savefig(os.path.join("figures", "research_accuracy_loss.png"))
        plt.close(fig)
except Exception as e:
    print("Error plotting Research Accuracy and Loss:", e)

# Plot 4: Research Rule Fidelity Curve
try:
    research = data_exps.get("Research", {})
    rf = get_array(research, "metrics", "Rule_Fidelity")
    if rf.size:
        epochs = np.arange(1, len(rf) + 1)
        fig = plt.figure(dpi=300)
        plt.plot(epochs, rf, marker='o', label="Rule Fidelity")
        plt.xlabel("Epoch")
        plt.ylabel("Rule Fidelity")
        plt.title("Research Experiment: Rule Fidelity Over Epochs")
        if np.any(rf):  # Only add legend if data exists
            plt.legend()
        plt.tight_layout()
        fig.savefig(os.path.join("figures", "research_rule_fidelity.png"))
        plt.close(fig)
except Exception as e:
    print("Error plotting Research Rule Fidelity:", e)

# Plot 5: Research Test Set Confusion Matrix
try:
    research = data_exps.get("Research", {})
    preds = np.asarray(research.get("predictions", []))
    gts = np.asarray(research.get("ground_truth", []))
    cm = compute_confusion_matrix(preds, gts)
    if cm is not None:
        fig = plt.figure(dpi=300)
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, label="Count")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Research Experiment: Test Set Confusion Matrix")
        plt.tight_layout()
        fig.savefig(os.path.join("figures", "research_confusion_matrix.png"))
        plt.close(fig)
except Exception as e:
    print("Error plotting Research Confusion Matrix:", e)

# Plot 6: Ablation Combined Validation Accuracy and Loss (Subplots)
try:
    ablation_names = ["No Gate Ensemble", "No Rule Sparsity", "Rule Free CNN",
                      "No Gate Confidence", "Rule Only Head", "Static Scalar Gate"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    # Left subplot: Validation Accuracy Comparison
    for name in ablation_names:
        exp_data = data_exps.get(name, {})
        val_acc = get_array(exp_data, "metrics", "val_acc")
        if val_acc.size:
            epochs = np.arange(1, len(val_acc) + 1)
            axs[0].plot(epochs, val_acc, marker='o', label=name)
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Validation Accuracy")
    axs[0].set_title("Ablation Studies: Validation Accuracy Comparison")
    axs[0].legend()
    # Right subplot: Validation Loss Comparison
    for name in ablation_names:
        exp_data = data_exps.get(name, {})
        val_loss = get_array(exp_data, "losses", "val")
        if val_loss.size:
            epochs = np.arange(1, len(val_loss) + 1)
            axs[1].plot(epochs, val_loss, marker='o', label=name)
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Validation Loss")
    axs[1].set_title("Ablation Studies: Validation Loss Comparison")
    axs[1].legend()
    plot_subplots(fig, axs)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "ablation_val_accuracy_loss.png"))
    plt.close(fig)
except Exception as e:
    print("Error plotting Ablation Validation Accuracy and Loss:", e)

# Plot 7: Ablation - Grid of Confusion Matrices for Ablation Studies
try:
    ablation_names = ["No Gate Ensemble", "No Rule Sparsity", "Rule Free CNN",
                      "No Gate Confidence", "Rule Only Head", "Static Scalar Gate"]
    fig, axs = plt.subplots(2, 3, figsize=(16, 10), dpi=300)
    axs = axs.ravel()
    for i, name in enumerate(ablation_names):
        exp_data = data_exps.get(name, {})
        preds = np.asarray(exp_data.get("predictions", []))
        gts = np.asarray(exp_data.get("ground_truth", []))
        cm = compute_confusion_matrix(preds, gts)
        if cm is not None:
            im = axs[i].imshow(cm, cmap="Blues")
            axs[i].set_title(name)
            axs[i].set_xlabel("Predicted")
            axs[i].set_ylabel("True")
            fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)
        else:
            axs[i].text(0.5, 0.5, "No Data", ha='center', va='center')
    plt.suptitle("Ablation Studies: Confusion Matrices", fontsize=18)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join("figures", "ablation_confusion_matrices.png"))
    plt.close(fig)
except Exception as e:
    print("Error plotting Ablation Confusion Matrices:", e)

# Plot 8: Ablation - Rule Fidelity Comparison
try:
    fig = plt.figure(dpi=300)
    for name in ablation_names:
        exp_data = data_exps.get(name, {})
        rf = get_array(exp_data, "metrics", "Rule_Fidelity")
        if not rf.size:
            rf = get_array(exp_data, "metrics", "RBA")
        if rf.size:
            epochs = np.arange(1, len(rf) + 1)
            plt.plot(epochs, rf, marker='o', label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Rule Fidelity")
    plt.ylim(0, 1.05)
    plt.title("Ablation Studies: Rule Fidelity Comparison")
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join("figures", "ablation_rule_fidelity.png"))
    plt.close(fig)
except Exception as e:
    print("Error plotting Ablation Rule Fidelity:", e)

# Plot 9: Ablation - Final Test Accuracy Comparison (Scatter Plot)
try:
    names = []
    test_accs = []
    for name in ablation_names:
        exp_data = data_exps.get(name, {})
        preds = np.asarray(exp_data.get("predictions", []))
        gts = np.asarray(exp_data.get("ground_truth", []))
        if preds.size and gts.size:
            acc = (preds == gts).mean()
            names.append(name)
            test_accs.append(acc)
    if names:
        fig = plt.figure(dpi=300)
        plt.scatter(range(len(names)), test_accs, color="green", s=100)
        plt.xticks(range(len(names)), names, rotation=30)
        plt.xlabel("Ablation Experiment")
        plt.ylabel("Test Accuracy")
        plt.ylim(0, 1)
        plt.title("Ablation Studies: Final Test Accuracy Comparison")
        for i, acc in enumerate(test_accs):
            plt.text(i, acc, f"{acc:.2f}", ha='center', va='bottom')
        plt.tight_layout()
        fig.savefig(os.path.join("figures", "ablation_test_accuracy.png"))
        plt.close(fig)
except Exception as e:
    print("Error plotting Ablation Test Accuracy:", e)

print("Final figures have been saved into the 'figures' directory.")
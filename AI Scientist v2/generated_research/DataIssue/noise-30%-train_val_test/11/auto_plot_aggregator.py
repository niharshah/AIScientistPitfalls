#!/usr/bin/env python3
"""
Final Aggregator Script for Comprehensive Research Paper Figures
This script loads experiment data from .npy files (using exact file paths as specified)
for the Baseline, Research, and Ablation studies, and produces final, publication‐quality
plots stored exclusively in the "figures/" directory.
Each figure is wrapped in its own try/except block so that failure in one does not
affect the others.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import itertools

# Set a global style for publishing (remove top/right spines, set larger fonts, dpi, etc.)
plt.rcParams.update({
    'font.size': 14,
    'axes.spines.right': False,
    'axes.spines.top': False
})

# Create output folder for final figures
os.makedirs("figures", exist_ok=True)


#############################
# 1. BASELINE EXPERIMENTS   #
#############################
# File path from BASELINE_SUMMARY:
baseline_file = "experiment_results/experiment_00ce423fb4a041eeb173944638254940_proc_3462735/experiment_data.npy"
try:
    baseline_data = np.load(baseline_file, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading baseline data: {e}")
    baseline_data = {}

# Expecting baseline_data to have a key "dropout" containing experiments by dropout rate.
dropout_dict = baseline_data.get("dropout", {})

# Plot 1: Baseline Macro-F1 curves (Train and Validation)
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300, sharex=True)
    fig.suptitle("Baseline: SPR_BENCH Macro-F1 Curves\nLeft: Train  |  Right: Validation", fontsize=16)
    for key, rec in dropout_dict.items():
        epochs = rec.get("epochs", [])
        train_f1 = rec.get("metrics", {}).get("train_macro_f1", [])
        val_f1 = rec.get("metrics", {}).get("val_macro_f1", [])
        axes[0].plot(epochs, train_f1, label=f"Dropout {key}")
        axes[1].plot(epochs, val_f1, label=f"Dropout {key}")
    for ax, title in zip(axes, ["Train Macro-F1", "Validation Macro-F1"]):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Macro-F1")
        ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join("figures", "Baseline_MacroF1_Curves.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error plotting Baseline Macro-F1 curves: {e}")
    plt.close()

# Plot 2: Baseline Loss curves (Train and Validation)
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300, sharex=True)
    fig.suptitle("Baseline: SPR_BENCH Cross-Entropy Loss\nLeft: Train  |  Right: Validation", fontsize=16)
    for key, rec in dropout_dict.items():
        epochs = rec.get("epochs", [])
        train_loss = rec.get("losses", {}).get("train", [])
        val_loss = rec.get("losses", {}).get("val", [])
        axes[0].plot(epochs, train_loss, label=f"Dropout {key}")
        axes[1].plot(epochs, val_loss, label=f"Dropout {key}")
    for ax, title in zip(axes, ["Train Loss", "Validation Loss"]):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join("figures", "Baseline_Loss_Curves.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error plotting Baseline Loss curves: {e}")
    plt.close()

# Plot 3: Baseline Test Macro-F1 Bar Chart
try:
    keys = []
    test_f1s = []
    for key, rec in dropout_dict.items():
        keys.append(f"Dropout {key}")
        test_f1s.append(rec.get("test_macro_f1", 0.0))
    fig = plt.figure(figsize=(8, 5), dpi=300)
    plt.bar(keys, test_f1s, color="skyblue")
    plt.title("Baseline: SPR_BENCH Test Macro-F1 by Dropout", fontsize=16)
    plt.ylabel("Macro-F1")
    plt.xlabel("Dropout Rate")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(os.path.join("figures", "Baseline_Test_MacroF1_Bar.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error plotting Baseline Test Macro-F1: {e}")
    plt.close()


##################################
# 2. RESEARCH EXPERIMENTS        #
##################################
# File path from RESEARCH_SUMMARY:
research_file = "experiment_results/experiment_9afb6e88daf844f5ac9eb55eb16acd94_proc_3470356/experiment_data.npy"
try:
    research_data = np.load(research_file, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading research data: {e}")
    research_data = {}

# Work on the "SPR_BENCH_reasoning" entry:
reasoning_rec = research_data.get("SPR_BENCH_reasoning", {})

# Plot 4: Research Macro-F1 curves (Train vs Validation)
try:
    epochs = reasoning_rec.get("epochs", [])
    train_f1 = reasoning_rec.get("metrics", {}).get("train_macro_f1", [])
    val_f1 = reasoning_rec.get("metrics", {}).get("val_macro_f1", [])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300, sharex=True)
    fig.suptitle("Research: SPR_BENCH_reasoning Macro-F1 Curves\nLeft: Train  |  Right: Validation", fontsize=16)
    axes[0].plot(epochs, train_f1, label="Train", color="blue")
    axes[1].plot(epochs, val_f1, label="Validation", color="orange")
    for ax, title in zip(axes, ["Train Macro-F1", "Validation Macro-F1"]):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Macro-F1")
        ax.set_ylim(0, 1)
        ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join("figures", "Research_MacroF1_Curves.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error plotting Research Macro-F1 curves: {e}")
    plt.close()

# Plot 5: Research Loss curves (Train vs Validation)
try:
    train_loss = reasoning_rec.get("losses", {}).get("train", [])
    val_loss = reasoning_rec.get("losses", {}).get("val", [])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300, sharex=True)
    fig.suptitle("Research: SPR_BENCH_reasoning Loss Curves\nLeft: Train  |  Right: Validation", fontsize=16)
    axes[0].plot(epochs, train_loss, label="Train", color="blue")
    axes[1].plot(epochs, val_loss, label="Validation", color="orange")
    for ax, title in zip(axes, ["Train Loss", "Validation Loss"]):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join("figures", "Research_Loss_Curves.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error plotting Research Loss curves: {e}")
    plt.close()

# Plot 6: Research Confusion Matrix
try:
    y_true = np.array(reasoning_rec.get("ground_truth", []))
    y_pred = np.array(reasoning_rec.get("predictions", []))
    if y_true.size and y_pred.size:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig = plt.figure(figsize=(6, 5), dpi=300)
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Research: SPR_BENCH_reasoning Test Confusion Matrix", fontsize=16)
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center", 
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=14)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        fig.savefig(os.path.join("figures", "Research_Confusion_Matrix.png"))
        plt.close(fig)
except Exception as e:
    print(f"Error plotting Research Confusion Matrix: {e}")
    plt.close()


##########################################
# 3. ABLATION EXPERIMENTS                #
##########################################
# --- A. No-RelVec Transformer Ablation ---
norel_file = "experiment_results/experiment_1f90f68192e34e3a8a4af4fc2f698efc_proc_3475996/experiment_data.npy"
try:
    norel_data = np.load(norel_file, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading No-RelVec Transformer data: {e}")
    norel_data = {}
norel_rec = norel_data.get("NoRelVec", {}).get("SPR_BENCH", {})

# Plot 7: No-RelVec Loss Curve
try:
    epochs = np.array(norel_rec.get("epochs", []))
    train_loss = np.array(norel_rec.get("losses", {}).get("train", []))
    val_loss = np.array(norel_rec.get("losses", {}).get("val", []))
    fig = plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Ablation (No-RelVec): SPR_BENCH Loss Curve", fontsize=16)
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join("figures", "Ablation_NoRelVec_Loss_Curve.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error plotting No-RelVec Loss Curve: {e}")
    plt.close()

# Plot 8: No-RelVec Macro-F1 Curve
try:
    train_f1 = np.array(norel_rec.get("metrics", {}).get("train_macro_f1", []))
    val_f1 = np.array(norel_rec.get("metrics", {}).get("val_macro_f1", []))
    fig = plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(epochs, train_f1, label="Train Macro-F1")
    plt.plot(epochs, val_f1, label="Validation Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Ablation (No-RelVec): SPR_BENCH Macro-F1 Curve", fontsize=16)
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join("figures", "Ablation_NoRelVec_MacroF1_Curve.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error plotting No-RelVec Macro-F1 Curve: {e}")
    plt.close()

# Plot 9: No-RelVec Confusion Matrix (built from predictions/ground-truth)
try:
    preds = np.array(norel_rec.get("predictions", []))
    trues = np.array(norel_rec.get("ground_truth", []))
    if preds.size and trues.size:
        classes = np.unique(trues)
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for t, p in zip(trues, preds):
            cm[t, p] += 1
        fig = plt.figure(figsize=(6, 5), dpi=300)
        im = plt.imshow(cm, cmap="Blues")
        plt.title("Ablation (No-RelVec): SPR_BENCH Confusion Matrix", fontsize=16)
        plt.colorbar(im)
        plt.xticks(np.arange(len(classes)), classes)
        plt.yticks(np.arange(len(classes)), classes)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, f"{cm[i, j]}", horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=14)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        fig.savefig(os.path.join("figures", "Ablation_NoRelVec_Confusion_Matrix.png"))
        plt.close(fig)
except Exception as e:
    print(f"Error plotting No-RelVec Confusion Matrix: {e}")
    plt.close()

# --- B. No-PosEnc Transformer Ablation ---
nposenc_file = "experiment_results/experiment_f6ea1b4de9184fcb891fa88d4ead94f2_proc_3475997/experiment_data.npy"
try:
    nposenc_data = np.load(nposenc_file, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading No-PosEnc Transformer data: {e}")
    nposenc_data = {}
nposenc_rec = nposenc_data.get("NoPosEnc", {}).get("SPR_BENCH", {})

# Plot 10: No-PosEnc Loss & Macro-F1 Curves (combined with dual y-axis)
try:
    epochs = nposenc_rec.get("epochs", [])
    tr_loss = nposenc_rec.get("losses", {}).get("train", [])
    val_loss = nposenc_rec.get("losses", {}).get("val", [])
    tr_f1 = nposenc_rec.get("metrics", {}).get("train_macro_f1", [])
    val_f1 = nposenc_rec.get("metrics", {}).get("val_macro_f1", [])
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
    ax1.plot(epochs, tr_loss, label="Train Loss", color="blue")
    ax1.plot(epochs, val_loss, label="Val Loss", color="red")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()
    ax2.plot(epochs, tr_f1, "g--", label="Train Macro-F1")
    ax2.plot(epochs, val_f1, "m--", label="Val Macro-F1")
    ax2.set_ylabel("Macro-F1")
    ax2.tick_params(axis='y')
    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    plt.title("Ablation (No-PosEnc): Loss and Macro-F1 Curves", fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "Ablation_NoPosEnc_Loss_MacroF1.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error plotting No-PosEnc Loss/F1 curves: {e}")
    plt.close()

# --- C. No-Transformer Encoder (BoW Baseline) Ablation ---
bow_file = "experiment_results/experiment_8f1a9d28ee124e9cbf4389ea9bddc2b7_proc_3475996/experiment_data.npy"
try:
    bow_data = np.load(bow_file, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading BoW Baseline data: {e}")
    bow_data = {}
bow_rec = bow_data.get("No-Transformer Encoder (BoW Baseline)", {}).get("SPR_BENCH", {})

# Plot 11: Composite plot for BoW Baseline:
# Create one figure with three subplots in one row:
try:
    epochs = bow_rec.get("epochs", [])
    train_loss = bow_rec.get("losses", {}).get("train", [])
    val_loss = bow_rec.get("losses", {}).get("val", [])
    train_f1 = bow_rec.get("metrics", {}).get("train_macro_f1", [])
    val_f1 = bow_rec.get("metrics", {}).get("val_macro_f1", [])
    test_f1 = bow_rec.get("test_macro_f1", None)
    best_val_f1 = max(val_f1) if val_f1 else None

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

    # Subplot 1: Loss Curves
    axs[0].plot(epochs, train_loss, label="Train Loss", color="blue")
    axs[0].plot(epochs, val_loss, label="Val Loss", color="red")
    axs[0].set_title("Loss Curves", fontsize=16)
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # Subplot 2: Macro-F1 Curves
    axs[1].plot(epochs, train_f1, label="Train Macro-F1", color="blue")
    axs[1].plot(epochs, val_f1, label="Val Macro-F1", color="red")
    axs[1].set_title("Macro-F1 Curves", fontsize=16)
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Macro-F1")
    axs[1].legend()

    # Subplot 3: Bar chart comparing Best Validation vs Test Macro-F1
    if (test_f1 is not None) and (best_val_f1 is not None):
        axs[2].bar(["Best Val", "Test"], [best_val_f1, test_f1], color=["skyblue", "salmon"])
        axs[2].set_title("Best Val vs. Test Macro-F1", fontsize=16)
        axs[2].set_ylabel("Macro-F1")
    else:
        axs[2].text(0.5, 0.5, "Data Unavailable", horizontalalignment="center", verticalalignment="center")
    plt.tight_layout()
    fig.savefig(os.path.join("figures", "Ablation_BoW_Composite.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error plotting BoW Baseline composite plot: {e}")
    plt.close()

print("Final figures have been saved in the 'figures/' directory.")
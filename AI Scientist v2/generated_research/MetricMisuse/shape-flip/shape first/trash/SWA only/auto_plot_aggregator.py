#!/usr/bin/env python3
"""
Final Aggregator Script for Zero-Shot Synthetic PolyRule Reasoning Figures

This script loads the final experiment results stored in .npy files as provided
by the baseline, research, and ablation experiments. It then generates a comprehensive
set of final, publication–ready plots, saving them under the "figures/" folder.

Each plotting block is wrapped in its own try–except to ensure one failure does not
halt the overall plotting. Font sizes, DPI, and professional styling are applied
to ensure clarity in the final PDF publication.

NOTE: This script does not fabricate data. It uses only the persisted .npy files.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Set global style and font size for publication quality
plt.rcParams.update({'font.size': 14, 'axes.spines.top': False, 'axes.spines.right': False})

# Create output directory
os.makedirs("figures", exist_ok=True)

#############################
# Utility functions
#############################
def safe_load_npy(path):
    try:
        data = np.load(path, allow_pickle=True).item()
        return data
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return {}

def plot_line(ax, x, y, label):
    ax.plot(x, y, marker='o', label=label)

def add_common_labels(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    
def compute_confusion_matrix(true, pred):
    if isinstance(true, list): true = np.array(true)
    if isinstance(pred, list): pred = np.array(pred)
    return confusion_matrix(true, pred, labels=[0, 1])

#############################
# Define file paths for various experiments
#############################

# Baseline
baseline_path = "experiment_results/experiment_ee56663c45764463b75d13b1a7d69f46_proc_2602724/experiment_data.npy"

# Research
research_path = "experiment_results/experiment_4cfc308f43f044f0ab15e52e9c62853a_proc_2604132/experiment_data.npy"

# Ablation studies:
abl_remove_symbolic = "experiment_results/experiment_0ece83b83bf34733b88532ad2a5e9a50_proc_2605921/experiment_data.npy"
abl_remove_eq       = "experiment_results/experiment_3d5f4e5a65444a4b905a7ff9bbec40dc_proc_2605923/experiment_data.npy"
abl_freeze_emb      = "experiment_results/experiment_39a1f5bc3bad4c1d9ba4247569742047_proc_2605924/experiment_data.npy"
abl_randomized      = "experiment_results/experiment_36cd6f340c3c4144b2dafbf176be7446_proc_2605923/experiment_data.npy"
abl_token_order     = "experiment_results/experiment_7fa5be50242044998d853d971fd879fb_proc_2605922/experiment_data.npy"
abl_multi_synth     = "experiment_results/experiment_55ab259cb8f245eba2b384bf3d0d72f2_proc_2605924/experiment_data.npy"

#############################
# Plot 1: Baseline Loss Curves
#############################
try:
    baseline_data = safe_load_npy(baseline_path)
    ed = baseline_data.get("SPR_BENCH", {})
    loss_train = ed.get("losses", {}).get("train", [])
    loss_val   = ed.get("losses", {}).get("val", [])
    if loss_train and loss_val:
        epochs = list(range(1, len(loss_train)+1))
        plt.figure(figsize=(8,6))
        plt.plot(epochs, loss_train, marker='o', label="Train Loss")
        plt.plot(epochs, loss_val, marker='o', label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Baseline: Loss Curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/baseline_loss_curves.png", dpi=300)
        plt.close()
    else:
        print("Baseline loss data unavailable; skipping baseline loss plot.")
except Exception as e:
    print(f"Error in Baseline Loss Curves: {e}")
    plt.close()

#############################
# Plot 2: Baseline Accuracy Curves
#############################
try:
    acc_train = [m.get("acc") for m in ed.get("metrics", {}).get("train", [])]
    acc_val   = [m.get("acc") for m in ed.get("metrics", {}).get("val", [])]
    if acc_train and acc_val:
        epochs = list(range(1, len(acc_train)+1))
        plt.figure(figsize=(8,6))
        plt.plot(epochs, acc_train, marker='o', label="Train Accuracy")
        plt.plot(epochs, acc_val, marker='o', label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Baseline: Accuracy Curves")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/baseline_accuracy_curves.png", dpi=300)
        plt.close()
    else:
        print("Baseline accuracy data unavailable; skipping baseline accuracy plot.")
except Exception as e:
    print(f"Error in Baseline Accuracy Curves: {e}")
    plt.close()

#############################
# Plot 3: Baseline Test Metrics Bar Chart
#############################
try:
    test_metrics = ed.get("metrics", {}).get("test", {})
    if test_metrics:
        # Using keys: ACC, SWA, CWA, and NRGS if available (lowercase keys used in npy may vary)
        keys = ["acc", "swa", "cwa"]
        vals = [test_metrics.get(k, np.nan) for k in keys]
        if test_metrics.get("NRGS") is not None:
            keys.append("NRGS")
            vals.append(test_metrics.get("NRGS"))
        plt.figure(figsize=(8,6))
        bars = plt.bar([k.upper() for k in keys], vals, color="steelblue")
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x()+bar.get_width()/2., height + 0.02, f"{height:.3f}", ha='center')
        plt.ylim(0, 1)
        plt.title("Baseline: Test Metrics")
        plt.tight_layout()
        plt.savefig("figures/baseline_test_metrics.png", dpi=300)
        plt.close()
    else:
        print("Baseline test metrics unavailable; skipping test metrics plot.")
except Exception as e:
    print(f"Error in Baseline Test Metrics Plot: {e}")
    plt.close()

#############################
# Plot 4: Baseline Prediction Outcomes (Correct vs Incorrect Bar)
#############################
try:
    preds = np.array(ed.get("predictions", []))
    gts   = np.array(ed.get("ground_truth", []))
    if preds.size and gts.size:
        correct = int((preds == gts).sum())
        incorrect = int(preds.size - correct)
        plt.figure(figsize=(8,6))
        plt.bar(["Correct", "Incorrect"], [correct, incorrect], color=["green", "red"])
        plt.title("Baseline: Prediction Outcomes")
        plt.tight_layout()
        plt.savefig("figures/baseline_prediction_outcomes.png", dpi=300)
        plt.close()
    else:
        print("Baseline predictions/ground truth data unavailable; skipping prediction outcomes plot.")
except Exception as e:
    print(f"Error in Baseline Prediction Outcomes: {e}")
    plt.close()

#############################
# Plot 5: Research - Loss, Accuracy & SWA Curves (3 Subplots)
#############################
try:
    research_data = safe_load_npy(research_path)
    ed = research_data.get("SPR_BENCH", {})
    losses_tr = ed.get("losses", {}).get("train", [])
    losses_va = ed.get("losses", {}).get("val", [])
    acc_tr = [m.get("acc") for m in ed.get("metrics", {}).get("train", [])]
    acc_va = [m.get("acc") for m in ed.get("metrics", {}).get("val", [])]
    swa_tr = [m.get("swa") for m in ed.get("metrics", {}).get("train", [])]
    swa_va = [m.get("swa") for m in ed.get("metrics", {}).get("val", [])]
    epochs = list(range(1, len(losses_tr)+1))
    fig, axs = plt.subplots(1, 3, figsize=(18,6))
    # Loss
    axs[0].plot(epochs, losses_tr, marker='o', label="Train Loss")
    axs[0].plot(epochs, losses_va, marker='o', label="Validation Loss")
    add_common_labels(axs[0], "Epoch", "Loss", "Research: Loss Curves")
    # Accuracy
    axs[1].plot(epochs, acc_tr, marker='o', label="Train Acc")
    axs[1].plot(epochs, acc_va, marker='o', label="Validation Acc")
    axs[1].set_ylim(0, 1)
    add_common_labels(axs[1], "Epoch", "Accuracy", "Research: Accuracy")
    # SWA
    axs[2].plot(epochs, swa_tr, marker='o', label="Train SWA")
    axs[2].plot(epochs, swa_va, marker='o', label="Validation SWA")
    axs[2].set_ylim(0, 1)
    add_common_labels(axs[2], "Epoch", "SWA", "Research: Shape-Weighted Accuracy")
    plt.tight_layout()
    plt.savefig("figures/research_curves.png", dpi=300)
    plt.close()
except Exception as e:
    print(f"Error in Research curves: {e}")
    plt.close()

#############################
# Plot 6: Research - Test Metrics Bar & Confusion Matrix (2 Subplots)
#############################
try:
    ed = research_data.get("SPR_BENCH", {})
    test_metrics = ed.get("metrics", {}).get("test", {})
    preds = np.array(ed.get("predictions", []))
    gts = np.array(ed.get("ground_truth", []))
    fig, axs = plt.subplots(1, 2, figsize=(14,6))
    # Bar Chart for test metrics (Accuracy and SWA)
    test_acc = test_metrics.get("acc")
    test_swa = test_metrics.get("swa")
    if test_acc is not None and test_swa is not None:
        axs[0].bar(["Accuracy", "SWA"], [test_acc, test_swa], color=["steelblue", "tan"])
        for i, v in enumerate([test_acc, test_swa]):
            axs[0].text(i, v + 0.02, f"{v:.3f}", ha="center")
        axs[0].set_ylim(0, 1)
        axs[0].set_title("Research: Test Metrics")
    else:
        axs[0].text(0.5, 0.5, "No Test Metrics", ha="center")
    # Confusion Matrix
    if preds.size and gts.size:
        cm = compute_confusion_matrix(gts, preds)
        im = axs[1].imshow(cm, cmap="Blues")
        axs[1].set_title("Research: Confusion Matrix")
        axs[1].set_xticks([0,1])
        axs[1].set_yticks([0,1])
        for i in range(2):
            for j in range(2):
                axs[1].text(j, i, cm[i,j], ha="center", va="center", color="black")
        plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    else:
        axs[1].text(0.5, 0.5, "No Confusion Data", ha="center")
    plt.tight_layout()
    plt.savefig("figures/research_test_metrics_and_cm.png", dpi=300)
    plt.close()
except Exception as e:
    print(f"Error in Research test metrics/confusion matrix: {e}")
    plt.close()

#############################
# Plot 7: Ablation - Remove-Symbolic-Branch (Loss & Accuracy on 2 subplots)
#############################
try:
    abld_data = safe_load_npy(abl_remove_symbolic)
    ed = abld_data.get("Remove-Symbolic-Branch", {}).get("SPR_BENCH", {})
    loss_tr = ed.get("losses", {}).get("train", [])
    loss_va = ed.get("losses", {}).get("val", [])
    acc_tr = [m.get("acc") for m in ed.get("metrics", {}).get("train", [])]
    acc_va = [m.get("acc") for m in ed.get("metrics", {}).get("val", [])]
    epochs = list(range(1, len(loss_tr)+1))
    fig, axs = plt.subplots(1,2, figsize=(14,6))
    axs[0].plot(epochs, loss_tr, marker='o', label="Train Loss")
    axs[0].plot(epochs, loss_va, marker='o', label="Validation Loss")
    add_common_labels(axs[0], "Epoch", "Loss", "Ablation (No Symbolic): Loss")
    axs[1].plot(epochs, acc_tr, marker='o', label="Train Acc")
    axs[1].plot(epochs, acc_va, marker='o', label="Validation Acc")
    axs[1].set_ylim(0,1)
    add_common_labels(axs[1], "Epoch", "Accuracy", "Ablation (No Symbolic): Accuracy")
    plt.tight_layout()
    plt.savefig("figures/ablation_remove_symbolic_branch.png", dpi=300)
    plt.close()
except Exception as e:
    print(f"Error in Ablation Remove-Symbolic-Branch plots: {e}")
    plt.close()

#############################
# Plot 8: Ablation - Remove-Equality-Feature (2x2 Subplots: Loss, Accuracy, SWA, Confusion Matrix)
#############################
try:
    abld_data = safe_load_npy(abl_remove_eq)
    ed = abld_data.get("Remove-Equality-Feature", {}).get("SPR_BENCH", {})
    epochs = list(range(1, len(ed.get("losses", {}).get("train", []))+1))
    loss_tr = ed.get("losses", {}).get("train", [])
    loss_va = ed.get("losses", {}).get("val", [])
    acc_tr = [m.get("acc") for m in ed.get("metrics", {}).get("train", [])]
    acc_va = [m.get("acc") for m in ed.get("metrics", {}).get("val", [])]
    swa_tr = [m.get("swa") for m in ed.get("metrics", {}).get("train", [])]
    swa_va = [m.get("swa") for m in ed.get("metrics", {}).get("val", [])]
    preds = np.array(ed.get("predictions", []))
    gts   = np.array(ed.get("ground_truth", []))
    cm = compute_confusion_matrix(gts, preds) if preds.size and gts.size else None

    fig, axs = plt.subplots(2,2, figsize=(14,12))
    # Loss
    axs[0,0].plot(epochs, loss_tr, marker='o', label="Train Loss")
    axs[0,0].plot(epochs, loss_va, marker='o', label="Val Loss")
    add_common_labels(axs[0,0], "Epoch", "Loss", "Loss Curves")
    # Accuracy
    axs[0,1].plot(epochs, acc_tr, marker='o', label="Train Acc")
    axs[0,1].plot(epochs, acc_va, marker='o', label="Val Acc")
    axs[0,1].set_ylim(0,1)
    add_common_labels(axs[0,1], "Epoch", "Accuracy", "Accuracy Curves")
    # SWA
    axs[1,0].plot(epochs, swa_tr, marker='o', label="Train SWA")
    axs[1,0].plot(epochs, swa_va, marker='o', label="Val SWA")
    axs[1,0].set_ylim(0,1)
    add_common_labels(axs[1,0], "Epoch", "SWA", "Shape-Weighted Accuracy")
    # Confusion Matrix
    if cm is not None:
        im = axs[1,1].imshow(cm, cmap="Blues")
        axs[1,1].set_title("Confusion Matrix")
        axs[1,1].set_xticks([0,1])
        axs[1,1].set_yticks([0,1])
        for i in range(2):
            for j in range(2):
                axs[1,1].text(j, i, cm[i,j], ha="center", va="center", color="black")
        plt.colorbar(im, ax=axs[1,1], fraction=0.046, pad=0.04)
    else:
        axs[1,1].text(0.5, 0.5, "No Data", ha="center")
    plt.tight_layout()
    plt.savefig("figures/ablation_remove_equality_feature.png", dpi=300)
    plt.close()
except Exception as e:
    print(f"Error in Ablation Remove-Equality-Feature plots: {e}")
    plt.close()

#############################
# Plot 9: Ablation - Freeze-Embedding-Learning (2x2 Subplots: Loss, Accuracy, SWA, Confusion Matrix)
#############################
try:
    abld_data = safe_load_npy(abl_freeze_emb)
    ed = abld_data.get("FreezeEmb", {}).get("SPR_BENCH", {})
    epochs = list(range(1, len(ed.get("losses", {}).get("train", []))+1))
    loss_tr = ed.get("losses", {}).get("train", [])
    loss_va = ed.get("losses", {}).get("val", [])
    acc_tr = [m.get("acc") for m in ed.get("metrics", {}).get("train", [])]
    acc_va = [m.get("acc") for m in ed.get("metrics", {}).get("val", [])]
    swa_tr = [m.get("swa") for m in ed.get("metrics", {}).get("train", [])]
    swa_va = [m.get("swa") for m in ed.get("metrics", {}).get("val", [])]
    preds = np.array(ed.get("predictions", []))
    gts   = np.array(ed.get("ground_truth", []))
    cm = compute_confusion_matrix(gts, preds) if preds.size and gts.size else None

    fig, axs = plt.subplots(2,2, figsize=(14,12))
    axs[0,0].plot(epochs, loss_tr, marker='o', label="Train Loss")
    axs[0,0].plot(epochs, loss_va, marker='o', label="Val Loss")
    add_common_labels(axs[0,0], "Epoch", "Loss", "Loss Curves")
    axs[0,1].plot(epochs, acc_tr, marker='o', label="Train Acc")
    axs[0,1].plot(epochs, acc_va, marker='o', label="Val Acc")
    axs[0,1].set_ylim(0,1)
    add_common_labels(axs[0,1], "Epoch", "Accuracy", "Accuracy Curves")
    axs[1,0].plot(epochs, swa_tr, marker='o', label="Train SWA")
    axs[1,0].plot(epochs, swa_va, marker='o', label="Val SWA")
    axs[1,0].set_ylim(0,1)
    add_common_labels(axs[1,0], "Epoch", "SWA", "Shape-Weighted Accuracy")
    if cm is not None:
        im = axs[1,1].imshow(cm, cmap="Blues")
        axs[1,1].set_title("Confusion Matrix")
        axs[1,1].set_xticks([0,1])
        axs[1,1].set_yticks([0,1])
        for i in range(2):
            for j in range(2):
                axs[1,1].text(j, i, cm[i,j], ha="center", va="center", color="black")
        plt.colorbar(im, ax=axs[1,1], fraction=0.046, pad=0.04)
    else:
        axs[1,1].text(0.5, 0.5, "No Data", ha="center")
    plt.tight_layout()
    plt.savefig("figures/ablation_freeze_embedding_learning.png", dpi=300)
    plt.close()
except Exception as e:
    print(f"Error in Ablation Freeze-Embedding-Learning plots: {e}")
    plt.close()

#############################
# Plot 10: Ablation - Randomized-Symbolic-Input (2x2 Subplots: Loss, Accuracy, SWA, Confusion Matrix)
#############################
try:
    abld_data = safe_load_npy(abl_randomized)
    ed = abld_data.get("Randomized-Symbolic-Input", {}).get("SPR_BENCH", {})
    epochs = list(range(1, len(ed.get("losses", {}).get("train", []))+1))
    loss_tr = ed.get("losses", {}).get("train", [])
    loss_va = ed.get("losses", {}).get("val", [])
    # Use helper to get metric list safely
    def get_metric(split, key):
        return [m.get(key) for m in ed.get("metrics", {}).get(split, [])]
    acc_tr = get_metric("train", "acc")
    acc_va = get_metric("val", "acc")
    swa_tr = get_metric("train", "swa")
    swa_va = get_metric("val", "swa")
    preds = np.array(ed.get("predictions", []))
    gts   = np.array(ed.get("ground_truth", []))
    cm = compute_confusion_matrix(gts, preds) if preds.size and gts.size else None

    fig, axs = plt.subplots(2,2, figsize=(14,12))
    axs[0,0].plot(epochs, loss_tr, marker='o', label="Train Loss")
    axs[0,0].plot(epochs, loss_va, marker='o', label="Val Loss")
    add_common_labels(axs[0,0], "Epoch", "Loss", "Loss Curves")
    axs[0,1].plot(epochs, acc_tr, marker='o', label="Train Acc")
    axs[0,1].plot(epochs, acc_va, marker='o', label="Val Acc")
    axs[0,1].set_ylim(0,1)
    add_common_labels(axs[0,1], "Epoch", "Accuracy", "Accuracy Curves")
    axs[1,0].plot(epochs, swa_tr, marker='o', label="Train SWA")
    axs[1,0].plot(epochs, swa_va, marker='o', label="Val SWA")
    axs[1,0].set_ylim(0,1)
    add_common_labels(axs[1,0], "Epoch", "SWA", "Shape-Weighted Accuracy")
    if cm is not None:
        im = axs[1,1].imshow(cm, cmap="Blues")
        axs[1,1].set_title("Confusion Matrix")
        axs[1,1].set_xticks([0,1])
        axs[1,1].set_yticks([0,1])
        for i in range(2):
            for j in range(2):
                axs[1,1].text(j, i, cm[i,j], ha="center", va="center", color="black")
        plt.colorbar(im, ax=axs[1,1], fraction=0.046, pad=0.04)
    else:
        axs[1,1].text(0.5, 0.5, "No Data", ha="center")
    plt.tight_layout()
    plt.savefig("figures/ablation_randomized_symbolic_input.png", dpi=300)
    plt.close()
except Exception as e:
    print(f"Error in Ablation Randomized-Symbolic-Input plots: {e}")
    plt.close()

#############################
# Plot 11: Ablation - Token-Order-Shuffled-Input (2 Subplots: Loss and Combined Accuracy/SWA)
#############################
try:
    abld_data = safe_load_npy(abl_token_order)
    ed = abld_data.get("Token-Order-Shuffled-Input", {}).get("SPR_BENCH", {})
    loss_tr = ed.get("losses", {}).get("train", [])
    loss_va = ed.get("losses", {}).get("val", [])
    acc_tr = [m.get("acc") for m in ed.get("metrics", {}).get("train", [])]
    acc_va = [m.get("acc") for m in ed.get("metrics", {}).get("val", [])]
    swa_tr = [m.get("swa") for m in ed.get("metrics", {}).get("train", [])]
    swa_va = [m.get("swa") for m in ed.get("metrics", {}).get("val", [])]
    epochs = list(range(1, len(loss_tr)+1))
    fig, axs = plt.subplots(1,2, figsize=(14,6))
    # Loss curves
    axs[0].plot(epochs, loss_tr, marker='o', label="Train Loss")
    axs[0].plot(epochs, loss_va, marker='o', label="Val Loss")
    add_common_labels(axs[0], "Epoch", "Loss", "Token-Order Shuffled: Loss")
    # Combined Accuracy and SWA curves in second subplot
    axs[1].plot(epochs, acc_tr, marker='o', label="Train Acc")
    axs[1].plot(epochs, acc_va, marker='o', label="Val Acc")
    axs[1].plot(epochs, swa_tr, marker='o', label="Train SWA")
    axs[1].plot(epochs, swa_va, marker='o', label="Val SWA")
    axs[1].set_ylim(0,1)
    add_common_labels(axs[1], "Epoch", "Metric Value", "Token-Order Shuffled: Acc & SWA")
    plt.tight_layout()
    plt.savefig("figures/ablation_token_order_shuffled_input.png", dpi=300)
    plt.close()
except Exception as e:
    print(f"Error in Ablation Token-Order-Shuffled-Input plots: {e}")
    plt.close()

#############################
# Plot 12 (Appendix): Ablation - Multi-Synthetic-Dataset Generalization (3 Subplots)
#############################
try:
    abld_data = safe_load_npy(abl_multi_synth)
    ed = None
    # Safely extract multi_synth_generalization entry 
    try:
        ed = abld_data["multi_synth_generalization"]["D1-D2-D3"]
    except Exception as ex:
        print("Multi-Synthetic entry not found.")
    if ed is not None:
        train_losses = ed.get("losses", {}).get("train", [])
        val_losses = ed.get("losses", {}).get("val", [])
        train_accs = [m.get("acc") for m in ed.get("metrics", {}).get("train", [])]
        val_accs   = [m.get("acc") for m in ed.get("metrics", {}).get("val", [])]
        train_swa  = [m.get("swa") for m in ed.get("metrics", {}).get("train", [])]
        val_swa    = [m.get("swa") for m in ed.get("metrics", {}).get("val", [])]
        epochs = list(range(1, len(train_losses)+1))
        fig, axs = plt.subplots(1,3, figsize=(18,6))
        # Loss curves
        axs[0].plot(epochs, train_losses, marker='o', label="Train Loss")
        axs[0].plot(epochs, val_losses, marker='o', label="Val Loss")
        add_common_labels(axs[0], "Epoch", "Loss", "Multi-Synth: Loss Curves")
        # Accuracy curves
        axs[1].plot(epochs, train_accs, marker='o', label="Train Accuracy")
        axs[1].plot(epochs, val_accs, marker='o', label="Val Accuracy")
        axs[1].set_ylim(0,1)
        add_common_labels(axs[1], "Epoch", "Accuracy", "Multi-Synth: Accuracy")
        # SWA curves
        axs[2].plot(epochs, train_swa, marker='o', label="Train SWA")
        axs[2].plot(epochs, val_swa, marker='o', label="Val SWA")
        axs[2].set_ylim(0,1)
        add_common_labels(axs[2], "Epoch", "SWA", "Multi-Synth: SWA")
        plt.tight_layout()
        plt.savefig("figures/ablation_multi_synth_generalization.png", dpi=300)
        plt.close()
    else:
        print("Multi-Synthetic-Dataset experiment entry not found; skipping appendix plot.")
except Exception as e:
    print(f"Error in Multi-Synthetic-Dataset Generalization plots: {e}")
    plt.close()

print("Final plots generated and saved under 'figures/' directory.")
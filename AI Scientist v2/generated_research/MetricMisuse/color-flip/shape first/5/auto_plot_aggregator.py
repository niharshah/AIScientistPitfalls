#!/usr/bin/env python3
"""
Aggregator script for final scientific plots for the paper:
"Context-aware Contrastive Learning for Enhanced Symbolic Pattern Recognition"

This script loads experiment .npy files from Baseline, Research, and Ablation studies,
aggregates and visualizes the final results (e.g., HSCA, CWCA, loss curves, confusion matrices),
and saves all finalized figures in the "figures/" folder.
Each plot is created in its own try-except block so that one failure will not stop subsequent plots.
All fonts and labels have increased sizes for publication quality.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import confusion_matrix

# Increase font sizes for publication quality
rcParams.update({'font.size': 12})

# Create figures output directory
os.makedirs("figures", exist_ok=True)

def load_experiment(file_path):
    """Attempts to load a numpy experiment dictionary from file_path."""
    try:
        data = np.load(file_path, allow_pickle=True).item()
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}

# ---------------------------
# BASELINE PLOTS
# File from baseline summary
baseline_file = "experiment_results/experiment_69cc842955bc4716a578b76635796f3e_proc_3069201/experiment_data.npy"
baseline_data = load_experiment(baseline_file)

# Access baseline experiment: key "supervised_finetuning_epochs" -> "SPR_BENCH"
try:
    sweep = baseline_data["supervised_finetuning_epochs"]["SPR_BENCH"]
    epochs_grid = sweep.get("epochs_grid", [])
    train_hsca = sweep.get("metrics", {}).get("train", [])
    val_hsca = sweep.get("metrics", {}).get("val", [])
    test_hsca = sweep.get("test_hsca", [])
except Exception as e:
    print("Error extracting baseline data:", e)
    epochs_grid, train_hsca, val_hsca, test_hsca = [], [], [], []

# Baseline Plot 1: Train and Validation HSCA Curves
try:
    plt.figure(figsize=(6,4), dpi=300)
    colors = plt.cm.tab10(np.linspace(0, 1, len(epochs_grid)))
    for i, max_ep in enumerate(epochs_grid):
        if i < len(train_hsca) and i < len(val_hsca):
            ep_axis = range(1, len(train_hsca[i]) + 1)
            plt.plot(ep_axis, train_hsca[i], color=colors[i], lw=2,
                     label=f"{max_ep} Epoch Train")
            plt.plot(ep_axis, val_hsca[i], color=colors[i], lw=2, linestyle="--",
                     label=f"{max_ep} Epoch Val")
    plt.xlabel("Epoch")
    plt.ylabel("HSCA")
    plt.title("Baseline: HSCA Curves (Train vs Val)")
    plt.legend(loc="best")
    plt.tight_layout()
    # Remove top and right spines for a cleaner look
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fname = os.path.join("figures", "Baseline_HSCA_Curves.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print("Saved Baseline HSCA Curves to", fname)
except Exception as e:
    print("Error creating Baseline HSCA curves:", e)
    plt.close()

# Baseline Plot 2: Test HSCA vs. Allowed Fine-tuning Epochs (Bar Chart)
try:
    plt.figure(figsize=(5,4), dpi=300)
    plt.bar([str(e) for e in epochs_grid], test_hsca, color="steelblue")
    plt.xlabel("Max Fine-tuning Epochs")
    plt.ylabel("Test HSCA")
    plt.title("Baseline: Test HSCA vs Fine-tuning Epochs")
    for idx, score in enumerate(test_hsca):
        plt.text(idx, score + 0.01, f"{score:.3f}", ha="center", fontsize=10)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    fname = os.path.join("figures", "Baseline_Test_HSCA_Bar.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print("Saved Baseline Test HSCA Bar to", fname)
except Exception as e:
    print("Error creating Baseline test HSCA bar chart:", e)
    plt.close()

# ---------------------------
# RESEARCH PLOTS
# File from research summary
research_file = "experiment_results/experiment_4095de4652b346919ac60564df5beb09_proc_3074526/experiment_data.npy"
research_data = load_experiment(research_file)

# Access research experiment: key "SPR_BENCH"
try:
    res_metrics = research_data["SPR_BENCH"].get("metrics", {})
    res_losses = research_data["SPR_BENCH"].get("losses", {})
    # For CWCA curves and loss curves, assume keys "train", "val" with equal length lists.
    epochs_axis = range(1, len(res_metrics.get("train", [])) + 1)
    test_cwca = res_metrics.get("test", [None])[0]
except Exception as e:
    print("Error extracting research data:", e)
    res_metrics, res_losses, epochs_axis, test_cwca = {}, {}, range(0), None

# Research Plot 1: CWCA Curves for Train and Validation (Line Plot)
try:
    plt.figure(figsize=(6,4), dpi=300)
    plt.plot(epochs_axis, res_metrics.get("train", []), label="Train CWCA", color="steelblue", lw=2)
    plt.plot(epochs_axis, res_metrics.get("val", []), label="Val CWCA", color="orange", lw=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("CWCA")
    plt.title("Research: CWCA Curves (Train vs Val)")
    plt.legend(loc="best")
    plt.tight_layout()
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fname = os.path.join("figures", "Research_CWCA_Curves.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print("Saved Research CWCA Curves to", fname)
except Exception as e:
    print("Error creating Research CWCA curves:", e)
    plt.close()

# Research Plot 2: Loss Curves for Train and Validation
try:
    plt.figure(figsize=(6,4), dpi=300)
    plt.plot(epochs_axis, res_losses.get("train", []), label="Train Loss", color="green", lw=2)
    plt.plot(epochs_axis, res_losses.get("val", []), label="Val Loss", color="red", lw=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Research: Loss Curves (Train vs Val)")
    plt.legend(loc="best")
    plt.tight_layout()
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fname = os.path.join("figures", "Research_Loss_Curves.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print("Saved Research Loss Curves to", fname)
except Exception as e:
    print("Error creating Research loss curves:", e)
    plt.close()

# Research Plot 3: Test CWCA as a Bar Chart
try:
    plt.figure(figsize=(4,4), dpi=300)
    if test_cwca is not None:
        plt.bar(["Test"], [test_cwca], color="purple")
        plt.ylabel("CWCA")
        plt.ylim(0, 1.0)
        plt.title("Research: Final Test CWCA")
        plt.text(0, test_cwca + 0.01, f"{test_cwca:.3f}", ha="center", fontsize=12)
    else:
        plt.text(0.5, 0.5, "Test CWCA not available", ha="center", va="center")
    plt.tight_layout()
    fname = os.path.join("figures", "Research_Test_CWCA_Bar.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print("Saved Research Test CWCA Bar to", fname)
except Exception as e:
    print("Error creating Research test CWCA bar chart:", e)
    plt.close()

# ---------------------------
# ABLATION PLOTS: "NoContrastivePretraining"
ncp_file = "experiment_results/experiment_df0775e215054ec2a8865131b33b4ba6_proc_3078651/experiment_data.npy"
ncp_data_all = load_experiment(ncp_file)

try:
    ncp_data = ncp_data_all["NoContrastivePretraining"]["SPR_BENCH"]
    ncp_losses_tr = ncp_data.get("losses", {}).get("train", [])
    ncp_losses_val = ncp_data.get("losses", {}).get("val", [])
    ncp_cwca_tr = ncp_data.get("metrics", {}).get("train", [])
    ncp_cwca_val = ncp_data.get("metrics", {}).get("val", [])
    ncp_cwca_test = ncp_data.get("metrics", {}).get("test", [None])[0]
    ncp_y_pred = np.array(ncp_data.get("predictions", []))
    ncp_y_true = np.array(ncp_data.get("ground_truth", []))
    epochs_ncp = range(1, len(ncp_losses_tr) + 1)
except Exception as e:
    print("Error extracting NoContrastivePretraining data:", e)
    ncp_losses_tr = ncp_losses_val = ncp_cwca_tr = ncp_cwca_val = []
    ncp_cwca_test = None; ncp_y_pred = ncp_y_true = np.array([])
    epochs_ncp = range(0)

# NCP Plot 1: Loss Curves (Train vs Val)
try:
    plt.figure(figsize=(6,4), dpi=300)
    plt.plot(epochs_ncp, ncp_losses_tr, label="Train Loss", color="green", lw=2)
    plt.plot(epochs_ncp, ncp_losses_val, label="Val Loss", color="red", lw=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Ablation (No Contrastive Pretraining): Loss Curves")
    plt.legend(loc="best")
    plt.tight_layout()
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fname = os.path.join("figures", "Ablation_NoContrastivePretraining_Loss_Curves.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print("Saved NCP Loss Curves to", fname)
except Exception as e:
    print("Error creating NoContrastivePretraining loss curves:", e)
    plt.close()

# NCP Plot 2: CWCA Curves (Train vs Val with Test Scatter)
try:
    plt.figure(figsize=(6,4), dpi=300)
    plt.plot(epochs_ncp, ncp_cwca_tr, label="Train CWCA", color="blue", lw=2)
    plt.plot(epochs_ncp, ncp_cwca_val, label="Val CWCA", color="orange", lw=2, linestyle="--")
    if ncp_cwca_test is not None and len(epochs_ncp) > 0:
        plt.scatter([epochs_ncp[-1]], [ncp_cwca_test], color="red", s=50,
                    label=f"Test ({ncp_cwca_test:.3f})")
    plt.xlabel("Epoch")
    plt.ylabel("CWCA")
    plt.title("Ablation (No Contrastive Pretraining): CWCA Curves")
    plt.legend(loc="best")
    plt.tight_layout()
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fname = os.path.join("figures", "Ablation_NoContrastivePretraining_CWCA_Curves.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print("Saved NCP CWCA Curves to", fname)
except Exception as e:
    print("Error creating NoContrastivePretraining CWCA curves:", e)
    plt.close()

# NCP Plot 3: Confusion Matrix Heatmap
try:
    if ncp_y_true.size and ncp_y_pred.size:
        cmatrix = np.zeros((int(max(ncp_y_true.max(), ncp_y_pred.max())+1),
                           int(max(ncp_y_true.max(), ncp_y_pred.max())+1)), int)
        for t, p in zip(ncp_y_true, ncp_y_pred):
            cmatrix[t, p] += 1
        plt.figure(figsize=(5,4), dpi=300)
        im = plt.imshow(cmatrix, cmap="Blues")
        plt.title("Ablation (No Contrastive Pretraining): Confusion Matrix\n(True=rows, Predicted=cols)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        for i in range(cmatrix.shape[0]):
            for j in range(cmatrix.shape[1]):
                plt.text(j, i, cmatrix[i, j], ha="center", va="center", color="black")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        fname = os.path.join("figures", "Ablation_NoContrastivePretraining_Confusion_Matrix.png")
        plt.savefig(fname, bbox_inches="tight")
        plt.close()
        print("Saved NCP Confusion Matrix to", fname)
    else:
        print("No predictions/labels available for NCP Confusion Matrix.")
except Exception as e:
    print("Error creating NoContrastivePretraining confusion matrix:", e)
    plt.close()

# ---------------------------
# ABLATION PLOTS: "BagOfTokensEncoder"
bote_file = "experiment_results/experiment_c078a069eb1b4509afe27acab2d633f8_proc_3078654/experiment_data.npy"
bote_data_all = load_experiment(bote_file)

# For BagOfTokensEncoder, we extract the SPR_BENCH results under that key.
try:
    bote_data = bote_data_all["BagOfTokensEncoder"]["SPR_BENCH"]
    bote_losses = bote_data.get("losses", {})
    bote_metrics = bote_data.get("metrics", {})
    bote_preds = bote_data.get("predictions", [])
    bote_gts = bote_data.get("ground_truth", [])
    epochs_bote = range(1, len(bote_losses.get("train", [])) + 1)
    bote_test_cwca = bote_metrics.get("test", [None])[0]
except Exception as e:
    print("Error extracting BagOfTokensEncoder data:", e)
    bote_losses = {}
    bote_metrics = {}
    bote_preds = bote_gts = []
    epochs_bote = range(0)
    bote_test_cwca = None

# Bote Plot 1: Loss Curves for BagOfTokensEncoder
try:
    plt.figure(figsize=(6,4), dpi=300)
    plt.plot(epochs_bote, bote_losses.get("train", []), label="Train Loss", color="green", lw=2)
    plt.plot(epochs_bote, bote_losses.get("val", []), label="Val Loss", color="red", lw=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("BagOfTokensEncoder: Loss Curves")
    plt.legend(loc="best")
    plt.tight_layout()
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fname = os.path.join("figures", "BagOfTokensEncoder_Loss_Curves.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print("Saved BagOfTokensEncoder Loss Curves to", fname)
except Exception as e:
    print("Error creating BagOfTokensEncoder loss curves:", e)
    plt.close()

# Bote Plot 2: CWCA Curves for BagOfTokensEncoder
try:
    plt.figure(figsize=(6,4), dpi=300)
    plt.plot(epochs_bote, bote_metrics.get("train", []), label="Train CWCA", color="blue", lw=2)
    plt.plot(epochs_bote, bote_metrics.get("val", []), label="Val CWCA", color="orange", lw=2, linestyle="--")
    if bote_test_cwca is not None and len(list(epochs_bote)) > 0:
        plt.scatter([list(epochs_bote)[-1]], [bote_test_cwca], color="red",
                    s=50, label=f"Test ({bote_test_cwca:.3f})")
    plt.xlabel("Epoch")
    plt.ylabel("CWCA")
    plt.title("BagOfTokensEncoder: CWCA Curves")
    plt.legend(loc="best")
    plt.tight_layout()
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fname = os.path.join("figures", "BagOfTokensEncoder_CWCA_Curves.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print("Saved BagOfTokensEncoder CWCA Curves to", fname)
except Exception as e:
    print("Error creating BagOfTokensEncoder CWCA curves:", e)
    plt.close()

# Bote Plot 3: Confusion Matrix for BagOfTokensEncoder
try:
    bote_preds_arr = np.array(bote_preds)
    bote_gts_arr = np.array(bote_gts)
    if bote_preds_arr.size and bote_gts_arr.size:
        n_labels = int(max(bote_gts_arr.max(), bote_preds_arr.max()) + 1)
        cm_bote = np.zeros((n_labels, n_labels), int)
        for t, p in zip(bote_gts_arr, bote_preds_arr):
            cm_bote[t, p] += 1
        plt.figure(figsize=(5,4), dpi=300)
        im = plt.imshow(cm_bote, cmap="Blues")
        plt.title("BagOfTokensEncoder: Confusion Matrix\n(True=rows, Predicted=cols)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        for i in range(n_labels):
            for j in range(n_labels):
                plt.text(j, i, cm_bote[i, j], ha="center", va="center", color="black")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        fname = os.path.join("figures", "BagOfTokensEncoder_Confusion_Matrix.png")
        plt.savefig(fname, bbox_inches="tight")
        plt.close()
        print("Saved BagOfTokensEncoder Confusion Matrix to", fname)
    else:
        print("No predictions/labels available for BagOfTokensEncoder Confusion Matrix.")
except Exception as e:
    print("Error creating BagOfTokensEncoder confusion matrix:", e)
    plt.close()

print("Final plotting complete. All figures are saved in the 'figures/' directory.")
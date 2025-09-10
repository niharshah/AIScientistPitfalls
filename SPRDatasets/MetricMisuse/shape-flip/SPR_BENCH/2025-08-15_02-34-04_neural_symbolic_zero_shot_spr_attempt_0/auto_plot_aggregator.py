#!/usr/bin/env python3
"""
Final Aggregator Script for Zero‐Shot Synthetic PolyRule Reasoning with Neural Symbolic Integration
This script loads experiment data from multiple .npy files and creates final, publication‐ready figures.
All figures are saved under the "figures/" directory.
Each plot is wrapped in its own try‐except block so that a failure in one does not stop the overall aggregation.
Author: Ambitious AI Researcher
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Set global parameters for publication-quality plots
plt.rcParams.update({'font.size': 12})
DPI = 300

# Ensure output directory exists
os.makedirs("figures", exist_ok=True)

###########################################
# Figure 1: Baseline Aggregated Plots
# Using experiment_results/experiment_86b5a5c9df62419583d3312a774ae27e_proc_2815569/experiment_data.npy
###########################################
try:
    baseline_path = "experiment_results/experiment_86b5a5c9df62419583d3312a774ae27e_proc_2815569/experiment_data.npy"
    baseline_data = np.load(baseline_path, allow_pickle=True).item()
    runs = baseline_data.get("num_epochs", {})
    if not runs:
        raise ValueError("No 'num_epochs' key found in baseline experiment data.")
    
    # Create a figure with 3 subplots: Loss curves, Validation HWA evolution, Final Test HWA bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=DPI)
    
    # Subplot 1: Combined Loss Curves (Train vs Val)
    for run_name, run in runs.items():
        epochs = np.arange(len(run["losses"]["train"]))
        axes[0].plot(epochs, run["losses"]["train"], linestyle="--", label=f"{run_name} Train")
        axes[0].plot(epochs, run["losses"]["val"], linestyle="-", label=f"{run_name} Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves (Train vs Val)")
    axes[0].legend()
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # Subplot 2: Validation HWA Evolution
    for run_name, run in runs.items():
        hwa_vals = [m[2] for m in run["metrics"]["val"]]
        step = max(1, len(hwa_vals) // 50)
        axes[1].plot(np.arange(len(hwa_vals))[::step], np.array(hwa_vals)[::step], label=run_name)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation HWA")
    axes[1].set_title("Validation HWA Evolution")
    axes[1].legend()
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    # Subplot 3: Final Test HWA Bar Chart
    names = []
    test_hwa = []
    for run_name, run in runs.items():
        names.append(run_name.replace("epochs_", "e"))
        test_hwa.append(run["metrics"]["test"][2])
    axes[2].bar(names, test_hwa, color="skyblue")
    axes[2].set_xlabel("Run")
    axes[2].set_ylabel("Test HWA")
    axes[2].set_title("Final Test HWA")
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    
    plt.suptitle("Baseline: Aggregated Training Metrics", fontsize=14)
    out_path = os.path.join("figures", "baseline_aggregated.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Baseline Aggregated Figure to {out_path}")
except Exception as e:
    print(f"Error creating Baseline Aggregated Figure: {e}")
    plt.close("all")

###########################################
# Figure 2: Unidirectional GRU Encoder (Ablation)
# Using experiment_results/experiment_a90ccad026064b71b19f0b4b1f6d6b5a_proc_2828410/experiment_data.npy
###########################################
try:
    uni_path = "experiment_results/experiment_a90ccad026064b71b19f0b4b1f6d6b5a_proc_2828410/experiment_data.npy"
    uni_data = np.load(uni_path, allow_pickle=True).item()
    # Data stored under key "unidirectional_gru" -> "spr_bench"
    uni_exp = uni_data.get("unidirectional_gru", {}).get("spr_bench", {})
    if not uni_exp:
        raise ValueError("Unidirectional GRU data not found.")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=DPI)
    
    # Subplot 1: Loss Curves
    losses = uni_exp.get("losses", {})
    if losses:
        epochs = range(1, len(losses.get("train", [])) + 1)
        axes[0].plot(epochs, losses.get("train", []), label="Train Loss")
        axes[0].plot(epochs, losses.get("val", []), label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Unidirectional GRU Loss")
        axes[0].legend()
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
    
    # Subplot 2: HWA Curves (Train and Val)
    metrics = uni_exp.get("metrics", {})
    if metrics:
        hwa_train = [m[2] for m in metrics.get("train", [])]
        hwa_val = [m[2] for m in metrics.get("val", [])]
        epochs = range(1, len(hwa_train) + 1)
        axes[1].plot(epochs, hwa_train, label="Train HWA")
        axes[1].plot(epochs, hwa_val, label="Val HWA")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("HWA")
        axes[1].set_title("HWA Evolution")
        axes[1].legend()
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
    
    # Subplot 3: Confusion Matrix
    preds = uni_exp.get("predictions", [])
    gts = uni_exp.get("ground_truth", [])
    if preds and gts:
        labels = sorted(set(gts) | set(preds))
        cm = confusion_matrix(gts, preds, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=axes[2], cmap="Blues", values_format="d")
        axes[2].set_title("Test Confusion Matrix")
        axes[2].spines['top'].set_visible(False)
        axes[2].spines['right'].set_visible(False)
    else:
        axes[2].text(0.5, 0.5, "No Prediction Data", ha="center")
    
    plt.suptitle("Ablation: Unidirectional GRU Encoder", fontsize=14)
    out_path = os.path.join("figures", "unidirectional_gru.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Unidirectional GRU Figure to {out_path}")
except Exception as e:
    print(f"Error creating Unidirectional GRU Figure: {e}")
    plt.close("all")

###########################################
# Figure 3: Mean-Pooled Encoder Outputs (Ablation)
# Using experiment_results/experiment_a497f41c457e4cdc831aaf49639a728d_proc_2828411/experiment_data.npy
###########################################
try:
    meanpool_path = "experiment_results/experiment_a497f41c457e4cdc831aaf49639a728d_proc_2828411/experiment_data.npy"
    meanpool_data = np.load(meanpool_path, allow_pickle=True).item()
    runs_mean = meanpool_data.get("mean_pool", {}).get("SPR", {}).get("num_epochs", {})
    if not runs_mean:
        raise ValueError("Mean-Pooled data not found.")
    
    # Create figure with 3 subplots: Training Loss, Validation Loss, and Test HWA Bar Chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=DPI)
    
    # Subplot 1: Aggregate Training Loss Curves
    for run_name, run in runs_mean.items():
        epochs = range(len(run["losses"]["train"]))
        axes[0].plot(epochs, run["losses"]["train"], label=run_name)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title("Training Loss Curves")
    axes[0].legend()
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # Subplot 2: Aggregate Validation Loss Curves
    for run_name, run in runs_mean.items():
        epochs = range(len(run["losses"]["val"]))
        axes[1].plot(epochs, run["losses"]["val"], label=run_name)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Loss")
    axes[1].set_title("Validation Loss Curves")
    axes[1].legend()
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    # Subplot 3: Final Test HWA Bar Chart
    names = []
    test_hwa = []
    for run_name, run in runs_mean.items():
        names.append(run_name)
        test_hwa.append(run["metrics"]["test"][2])
    axes[2].bar(names, test_hwa, color="skyblue")
    axes[2].set_xlabel("Run")
    axes[2].set_ylabel("Test HWA")
    axes[2].set_title("Test HWA Comparison")
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    
    plt.suptitle("Ablation: Mean-Pooled Encoder Outputs", fontsize=14)
    out_path = os.path.join("figures", "mean_pool.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Mean-Pooled Encoder Figure to {out_path}")
except Exception as e:
    print(f"Error creating Mean-Pooled Encoder Figure: {e}")
    plt.close("all")

###########################################
# Figure 4: Randomly Shuffled Token Order (Ablation)
# Using experiment_results/experiment_201d0449862d4120945ffb7b26586b10_proc_2828412/experiment_data.npy
###########################################
try:
    shuffle_path = "experiment_results/experiment_201d0449862d4120945ffb7b26586b10_proc_2828412/experiment_data.npy"
    shuffle_data = np.load(shuffle_path, allow_pickle=True).item()
    runs_shuffle = shuffle_data.get("random_shuffle", {}).get("num_epochs", {})
    if not runs_shuffle:
        raise ValueError("Randomly shuffled token data not found.")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=DPI)
    
    # Subplot 1: Loss Curves (Train vs Val)
    for run_name, run in runs_shuffle.items():
        epochs = range(len(run["losses"]["train"]))
        axes[0].plot(epochs, run["losses"]["train"], "--", label=f"{run_name} Train")
        axes[0].plot(epochs, run["losses"]["val"], "-", label=f"{run_name} Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # Subplot 2: Validation HWA Curves
    for run_name, run in runs_shuffle.items():
        hwa_vals = [m[2] for m in run["metrics"]["val"]]
        axes[1].plot(range(len(hwa_vals)), hwa_vals, label=run_name)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("HWA")
    axes[1].set_title("Validation HWA")
    axes[1].legend()
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    # Subplot 3: Final Test HWA Bar Chart
    names = []
    test_hwa = []
    for run_name, run in runs_shuffle.items():
        names.append(run_name)
        test_hwa.append(run["metrics"]["test"][2])
    axes[2].bar(names, test_hwa, color="skyblue")
    axes[2].set_xlabel("Run")
    axes[2].set_ylabel("Test HWA")
    axes[2].set_title("Test HWA Comparison")
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    
    plt.suptitle("Ablation: Randomly Shuffled Token Order", fontsize=14)
    out_path = os.path.join("figures", "random_shuffle.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Randomly Shuffled Token Order Figure to {out_path}")
except Exception as e:
    print(f"Error creating Randomly Shuffled Token Order Figure: {e}")
    plt.close("all")

###########################################
# Figure 5: Remove Length Masking (Unpacked GRU) - Confusion Matrix
# Using experiment_results/experiment_0faf2fce22f04349ac0bd1f334e90fe5_proc_2828413/experiment_data.npy
###########################################
try:
    unpacked_path = "experiment_results/experiment_0faf2fce22f04349ac0bd1f334e90fe5_proc_2828413/experiment_data.npy"
    unpacked_data = np.load(unpacked_path, allow_pickle=True).item()
    runs_unpacked = unpacked_data.get("unpacked_gru", {})
    if not runs_unpacked:
        raise ValueError("Unpacked GRU data not found.")
    # Choose best run by final test HWA
    best_run_key = max(runs_unpacked, key=lambda k: runs_unpacked[k]["metrics"]["test"][2])
    best_run = runs_unpacked[best_run_key]
    y_true = np.array(best_run.get("ground_truth", []))
    y_pred = np.array(best_run.get("predictions", []))
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("Missing predictions or ground truth for unpacked GRU.")
    
    fig, ax = plt.subplots(figsize=(5, 5), dpi=DPI)
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix (Best: {best_run_key})")
    plt.tight_layout()
    out_path = os.path.join("figures", "unpacked_gru_confusion_matrix.png")
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Unpacked GRU Confusion Matrix Figure to {out_path}")
except Exception as e:
    print(f"Error creating Unpacked GRU Confusion Matrix Figure: {e}")
    plt.close("all")

###########################################
# Figure 6: Frozen Embedding Layer - Best Run HWA Curves
# Using experiment_results/experiment_981e65fea7914444a75f4e7774b495c5_proc_2828410/experiment_data.npy
###########################################
try:
    frozen_path = "experiment_results/experiment_981e65fea7914444a75f4e7774b495c5_proc_2828410/experiment_data.npy"
    frozen_data = np.load(frozen_path, allow_pickle=True).item()
    runs_frozen = frozen_data.get("frozen_embeddings", {}).get("SPR_BENCH", {}).get("num_epochs", {})
    if not runs_frozen:
        raise ValueError("Frozen Embedding data not found.")
    # Pick best run by test HWA
    best_key = max(runs_frozen, key=lambda k: runs_frozen[k]["metrics"]["test"][2])
    best_run = runs_frozen[best_key]
    train_hwa = [m[2] for m in best_run["metrics"]["train"]]
    val_hwa = [m[2] for m in best_run["metrics"]["val"]]
    
    fig, ax = plt.subplots(figsize=(6, 4), dpi=DPI)
    ax.plot(train_hwa, label="Train HWA")
    ax.plot(val_hwa, label="Val HWA")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("HWA")
    ax.set_title(f"Frozen Embeddings Best Run ({best_key})\nTrain vs Val HWA")
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    out_path = os.path.join("figures", "frozen_embeddings_best_hwa.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Frozen Embeddings Best HWA Figure to {out_path}")
except Exception as e:
    print(f"Error creating Frozen Embeddings Best HWA Figure: {e}")
    plt.close("all")

###########################################
# Figure 7: One-Hot Input Representation (No Embedding Layer)
# Using experiment_results/experiment_60518092b9fe4bcb811f2a8d5dee66f0_proc_2828413/experiment_data.npy
###########################################
try:
    onehot_path = "experiment_results/experiment_60518092b9fe4bcb811f2a8d5dee66f0_proc_2828413/experiment_data.npy"
    onehot_data = np.load(onehot_path, allow_pickle=True).item()
    runs_onehot = onehot_data.get("onehot_no_embedding", {})
    if not runs_onehot:
        raise ValueError("One-Hot representation data not found.")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=DPI)
    # Left: Combined Loss Curves (Train & Val)
    for run_name, run in runs_onehot.items():
        axes[0].plot(run["losses"]["train"], label=f"{run_name} Train")
        axes[0].plot(run["losses"]["val"], linestyle="--", label=f"{run_name} Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # Right: Validation HWA Curves
    for run_name, run in runs_onehot.items():
        hwa_vals = [m[2] for m in run["metrics"]["val"]]
        axes[1].plot(hwa_vals, label=run_name)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("HWA")
    axes[1].set_title("Validation HWA")
    axes[1].legend()
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    plt.suptitle("Ablation: One-Hot Input Representation", fontsize=14)
    out_path = os.path.join("figures", "onehot_no_embedding.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved One-Hot Input Representation Figure to {out_path}")
except Exception as e:
    print(f"Error creating One-Hot Input Representation Figure: {e}")
    plt.close("all")

###########################################
# Figure 8: Random Token Masking (15% Training-Time Dropout)
# Using experiment_results/experiment_63db08cf574244a782536e0be20e9197_proc_2828412/experiment_data.npy
###########################################
try:
    tokenmask_path = "experiment_results/experiment_63db08cf574244a782536e0be20e9197_proc_2828412/experiment_data.npy"
    tokenmask_data = np.load(tokenmask_path, allow_pickle=True).item()
    runs_token = tokenmask_data.get("random_token_mask_15", {}).get("SPR_BENCH", {}).get("runs", {})
    if not runs_token:
        raise ValueError("Random token masking data not found.")
    # For clarity, pick run "epochs_20" if available; otherwise, the first run.
    run_key = "epochs_20" if "epochs_20" in runs_token else list(runs_token.keys())[0]
    run = runs_token[run_key]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=DPI)
    # Left: Loss Curves
    axes[0].plot(run["losses"]["train"], "--", label="Train Loss")
    axes[0].plot(run["losses"]["val"], "-", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # Right: HWA Curves
    hwa_vals_train = [m[2] for m in run["metrics"]["train"]]
    hwa_vals_val = [m[2] for m in run["metrics"]["val"]]
    axes[1].plot(hwa_vals_train, "--", label="Train HWA")
    axes[1].plot(hwa_vals_val, "-", label="Val HWA")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("HWA")
    axes[1].set_title("HWA Curves")
    axes[1].legend()
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    plt.suptitle(f"Random Token Masking ({run_key})", fontsize=14)
    out_path = os.path.join("figures", "random_token_mask_15.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Random Token Masking Figure to {out_path}")
except Exception as e:
    print(f"Error creating Random Token Masking Figure: {e}")
    plt.close("all")

print("Aggregation complete. All final figures are saved in the 'figures/' directory.")
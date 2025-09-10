#!/usr/bin/env python3
"""
Final Aggregator Script for Publishing Figures
This script loads experimental .npy result files from Baseline, Research, and Ablation experiments,
aggregates and visualizes the final results, and saves all figures in the "figures/" folder.
Each plotting block is wrapped in a try-except block so that failure of one plot does not
prevent the other plots from being generated.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import Counter

# Increase global font sizes for publication-quality plots
plt.rcParams.update({'font.size': 14})

# Ensure output directory exists
os.makedirs("figures", exist_ok=True)


#######################################
# Helper functions for plotting styles
#######################################
def style_ax(ax):
    # Remove top and right spines for a cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.5)


def compute_confusion_matrix(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    label_to_index = {l: idx for idx, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[label_to_index[t], label_to_index[p]] += 1
    return cm, labels


#######################################
# 1. BASELINE EXPERIMENTS
#######################################
baseline_path = os.path.join("experiment_results", 
                               "experiment_8357db79fad14a16ac4ae32f59b8972c_proc_1664068", 
                               "experiment_data.npy")
try:
    baseline_data = np.load(baseline_path, allow_pickle=True).item()
except Exception as e:
    print(f"Baseline: Error loading file {baseline_path}: {e}")
    baseline_data = {}

# Plot A: Aggregated Loss Curves per Adam Beta2 setting from Baseline
try:
    beta2_dict = baseline_data.get("adam_beta2", {})
    if not beta2_dict:
        raise ValueError("No 'adam_beta2' key found in baseline_data.")
    beta2_keys = sorted(beta2_dict.keys(), key=lambda x: float(x))
    n_plots = len(beta2_keys)
    cols = 3
    rows = math.ceil(n_plots / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3), squeeze=False)
    for idx, beta in enumerate(beta2_keys):
        ax = axes[idx // cols][idx % cols]
        try:
            exp_entry = beta2_dict[beta]["SPR_BENCH"]
            losses = exp_entry["losses"]
            # Each losses is dictionary with keys "train" and "val", values are list of (epoch, loss)
            train_loss = losses.get("train", [])
            val_loss = losses.get("val", [])
            if train_loss:
                epochs_tr, loss_tr = zip(*train_loss)
                ax.plot(epochs_tr, loss_tr, label="Train")
            if val_loss:
                epochs_val, loss_val = zip(*val_loss)
                ax.plot(epochs_val, loss_val, label="Validation")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Cross-Entropy Loss")
            ax.set_title(f"Beta2 = {beta}")
            ax.legend()
            style_ax(ax)
        except Exception as inner_e:
            print(f"Baseline: Error plotting loss curve for beta2={beta}: {inner_e}")
    # Remove unused subplots if any
    for j in range(idx+1, rows*cols):
        fig.delaxes(axes[j//cols][j % cols])
    fig.suptitle("Baseline: Loss Curves by Adam Beta2 Setting", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join("figures", "baseline_loss_curves.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print(f"Baseline: Error creating aggregated loss curves: {e}")
    plt.close("all")

# Plot B: Summary Final Validation Metrics bar chart from Baseline
try:
    metrics = ["CWA", "SWA", "EWA"]
    beta2_values = []
    metrics_vals = {m: [] for m in metrics}
    for beta in beta2_keys:
        beta2_values.append(beta)
        metr_list = beta2_dict[beta]["SPR_BENCH"]["metrics"]["val"]
        # Final epoch metrics assumed to be the last entry; each entry is a tuple (_, dict)
        if metr_list:
            _, final_metrics = metr_list[-1]
            for m in metrics:
                metrics_vals[m].append(final_metrics.get(m, 0))
        else:
            for m in metrics:
                metrics_vals[m].append(0)
    x = np.arange(len(beta2_values))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, metrics_vals[m], width, label=m)
    ax.set_xticks(x + width)
    ax.set_xticklabels(beta2_values)
    ax.set_ylabel("Score")
    ax.set_title("Baseline: Final Validation Metrics by Beta2 Setting")
    ax.legend()
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "baseline_final_val_metrics.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print(f"Baseline: Error creating summary metric plot: {e}")
    plt.close("all")


#######################################
# 2. RESEARCH EXPERIMENTS (Dual-Channel)
#######################################
research_path = os.path.join("experiment_results", 
                             "experiment_418bc2d626e542ff8ad92271619fb28f_proc_1695462", 
                             "experiment_data.npy")
try:
    research_data = np.load(research_path, allow_pickle=True).item()
except Exception as e:
    print(f"Research: Error loading file {research_path}: {e}")
    research_data = {}

dual_data = research_data.get("dual_channel", {})

# Plot C: Dual-Channel Loss Curve
try:
    losses = dual_data.get("losses", {})
    train_loss = losses.get("train", [])
    val_loss = losses.get("val", [])
    fig, ax = plt.subplots(figsize=(6, 4))
    if train_loss:
        epochs_tr, loss_tr = zip(*train_loss)
        ax.plot(epochs_tr, loss_tr, label="Train")
    if val_loss:
        epochs_val, loss_val = zip(*val_loss)
        ax.plot(epochs_val, loss_val, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Research (Dual-Channel): Loss Curve")
    ax.legend()
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "research_dual_channel_loss_curve.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print(f"Research: Error plotting dual-channel loss curve: {e}")
    plt.close("all")

# Plot D: Dual-Channel Metric Curves (CWA, SWA, PCWA)
try:
    metrics_val = dual_data.get("metrics", {}).get("val", [])
    epochs = []
    cwa_vals, swa_vals, pcwa_vals = [], [], []
    for ep, m in metrics_val:
        epochs.append(ep)
        cwa_vals.append(m.get("CWA", 0))
        swa_vals.append(m.get("SWA", 0))
        pcwa_vals.append(m.get("PCWA", 0))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, cwa_vals, marker="o", label="CWA")
    ax.plot(epochs, swa_vals, marker="o", label="SWA")
    ax.plot(epochs, pcwa_vals, marker="o", label="PCWA")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Research (Dual-Channel): Validation Metrics Over Epochs")
    ax.legend()
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "research_dual_channel_metric_curves.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print(f"Research: Error plotting dual-channel metric curves: {e}")
    plt.close("all")

# Plot E: Dual-Channel Confusion Matrix
try:
    preds = dual_data.get("predictions", [])
    gts = dual_data.get("ground_truth", [])
    if preds and gts:
        cm, labels = compute_confusion_matrix(gts, preds)
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.imshow(cm, cmap="Blues")
        plt.colorbar(cax, ax=ax)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")
        ax.set_title("Research (Dual-Channel): Confusion Matrix")
        style_ax(ax)
        fig.tight_layout()
        fig.savefig(os.path.join("figures", "research_dual_channel_confusion_matrix.png"), dpi=300)
        plt.close(fig)
    else:
        print("Research: Predictions or ground_truth data missing; skipping confusion matrix.")
except Exception as e:
    print(f"Research: Error plotting confusion matrix: {e}")
    plt.close("all")


#######################################
# 3. ABLATION EXPERIMENTS
#######################################
# 3.A Shape-Only Channel (Remove Color Embedding)
shape_only_path = os.path.join("experiment_results", 
                               "experiment_db63747412b449a7b627d037ed5ce099_proc_1705232", 
                               "experiment_data.npy")
try:
    shape_only_data_all = np.load(shape_only_path, allow_pickle=True).item()
except Exception as e:
    print(f"Shape-Only: Error loading file {shape_only_path}: {e}")
    shape_only_data_all = {}

# Data is under key "shape_only" -> "SPR"
shape_only = shape_only_data_all.get("shape_only", {}).get("SPR", {})

# Plot F: Shape-Only Loss Curve
try:
    losses = shape_only.get("losses", {})
    tr = losses.get("train", [])
    va = losses.get("val", [])
    fig, ax = plt.subplots(figsize=(6, 4))
    if tr:
        epochs_tr, loss_tr = zip(*tr)
        ax.plot(epochs_tr, loss_tr, label="Train Loss")
    if va:
        epochs_va, loss_va = zip(*va)
        ax.plot(epochs_va, loss_va, label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Ablation (Shape-Only): Loss Curve")
    ax.legend()
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "ablation_shape_only_loss_curve.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print(f"Shape-Only: Error plotting loss curve: {e}")
    plt.close("all")

# Plot G: Shape-Only Metric Curves
try:
    metrics_val = shape_only.get("metrics", {}).get("val", [])
    epochs, cwa_vals, swa_vals, pcwa_vals = [], [], [], []
    for ep, m in metrics_val:
        epochs.append(ep)
        cwa_vals.append(m.get("CWA", 0))
        swa_vals.append(m.get("SWA", 0))
        pcwa_vals.append(m.get("PCWA", 0))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, cwa_vals, marker="o", label="CWA")
    ax.plot(epochs, swa_vals, marker="o", label="SWA")
    ax.plot(epochs, pcwa_vals, marker="o", label="PCWA")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Ablation (Shape-Only): Validation Metrics")
    ax.legend()
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "ablation_shape_only_metric_curves.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print(f"Shape-Only: Error plotting metric curves: {e}")
    plt.close("all")

# Plot H: Shape-Only Confusion Matrix
try:
    y_true = shape_only.get("ground_truth", [])
    y_pred = shape_only.get("predictions", [])
    if y_true and y_pred:
        cm, labels = compute_confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.imshow(cm, cmap="Blues")
        plt.colorbar(cax, ax=ax)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")
        ax.set_title("Ablation (Shape-Only): Confusion Matrix")
        style_ax(ax)
        fig.tight_layout()
        fig.savefig(os.path.join("figures", "ablation_shape_only_confusion_matrix.png"), dpi=300)
        plt.close(fig)
    else:
        print("Shape-Only: Missing ground truth or predictions; skipping confusion matrix.")
except Exception as e:
    print(f"Shape-Only: Error plotting confusion matrix: {e}")
    plt.close("all")


# 3.B Late-Fusion Dual-LSTM (No Early Concatenation)
late_fusion_path = os.path.join("experiment_results", 
                                  "experiment_02730bed85f84e408526ba253fa48fa7_proc_1705233", 
                                  "experiment_data.npy")
try:
    late_fusion_data = np.load(late_fusion_path, allow_pickle=True).item()
except Exception as e:
    print(f"Late-Fusion: Error loading file {late_fusion_path}: {e}")
    late_fusion_data = {}

lf_data = late_fusion_data.get("late_fusion_dual_lstm", {}).get("dual_channel", {})

# Plot I: Late-Fusion – Aggregated Loss and Metrics Curves (using subplots)
try:
    # Loss curves
    lf_losses = lf_data.get("losses", {})
    loss_train = lf_losses.get("train", [])
    loss_val = lf_losses.get("val", [])
    # Metrics curves from validation metrics data: list of (epoch, dict)
    lf_metrics = lf_data.get("metrics", {}).get("val", [])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Loss Curve
    ax1 = axes[0]
    if loss_train:
        ep_tr, l_tr = zip(*loss_train)
        ax1.plot(ep_tr, l_tr, label="Train")
    if loss_val:
        ep_va, l_va = zip(*loss_val)
        ax1.plot(ep_va, l_va, label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Late-Fusion: Loss Curve")
    ax1.legend()
    style_ax(ax1)
    
    # Right: Metric curves
    epochs, cwa_vals, swa_vals, pcwa_vals = [], [], [], []
    for ep, m in lf_metrics:
        epochs.append(ep)
        cwa_vals.append(m.get("CWA", 0))
        swa_vals.append(m.get("SWA", 0))
        pcwa_vals.append(m.get("PCWA", 0))
    ax2.plot(epochs, cwa_vals, marker="o", label="CWA")
    ax2.plot(epochs, swa_vals, marker="o", label="SWA")
    ax2.plot(epochs, pcwa_vals, marker="o", label="PCWA")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Late-Fusion: Validation Metrics")
    ax2.legend()
    style_ax(ax2)
    
    fig.suptitle("Ablation (Late-Fusion Dual-LSTM)")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join("figures", "ablation_late_fusion_duallstm.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print(f"Late-Fusion: Error plotting aggregated loss/metrics: {e}")
    plt.close("all")


# 3.C Shared-Embedding (Tied Shape & Color Embeddings)
shared_embedding_path = os.path.join("experiment_results", 
                                       "experiment_82e596ccacb646609f8ea42362a4e93f_proc_1705232", 
                                       "experiment_data.npy")
try:
    shared_data = np.load(shared_embedding_path, allow_pickle=True).item()
except Exception as e:
    print(f"Shared-Embedding: Error loading file {shared_embedding_path}: {e}")
    shared_data = {}

se_data = shared_data.get("shared_embedding", {})

# Plot J: Shared-Embedding Loss and Metrics (2 subplots)
try:
    se_losses = se_data.get("losses", {})
    se_tr = se_losses.get("train", [])
    se_va = se_losses.get("val", [])
    se_metrics = se_data.get("metrics", {}).get("val", [])
    
    fig, axes = plt.subplots(2, 1, figsize=(7, 8))
    # Top: Loss Curve
    ax1 = axes[0]
    if se_tr:
        ep_tr, l_tr = zip(*se_tr)
        ax1.plot(ep_tr, l_tr, label="Train Loss")
    if se_va:
        ep_va, l_va = zip(*se_va)
        ax1.plot(ep_va, l_va, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Shared-Embedding: Loss Curve")
    ax1.legend()
    style_ax(ax1)
    
    # Bottom: Metrics curve
    epochs, cwa_vals, swa_vals, pcwa_vals = [], [], [], []
    for ep, m in se_metrics:
        epochs.append(ep)
        cwa_vals.append(m.get("CWA", 0))
        swa_vals.append(m.get("SWA", 0))
        pcwa_vals.append(m.get("PCWA", 0))
    ax2.plot(epochs, cwa_vals, marker="o", label="CWA")
    ax2.plot(epochs, swa_vals, marker="o", label="SWA")
    ax2.plot(epochs, pcwa_vals, marker="o", label="PCWA")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Shared-Embedding: Validation Metrics")
    ax2.legend()
    style_ax(ax2)
    
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "ablation_shared_embedding.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print(f"Shared-Embedding: Error plotting loss/metrics: {e}")
    plt.close("all")


# 3.D Unmasked Mean Pooling (Padding-Aware Removed)
unmasked_path = os.path.join("experiment_results", 
                             "experiment_b39904b7db7b46a7a70f6b4309cf9806_proc_1705234", 
                             "experiment_data.npy")
try:
    unmasked_data_all = np.load(unmasked_path, allow_pickle=True).item()
except Exception as e:
    print(f"Unmasked Mean Pooling: Error loading file {unmasked_path}: {e}")
    unmasked_data_all = {}

ump_data = unmasked_data_all.get("unmasked_mean_pooling", {}).get("dual_channel", {})

# Plot K: Unmasked Mean Pooling – Loss and Metrics curves (aggregated in one figure)
try:
    ump_losses = ump_data.get("losses", {})
    ump_tr = ump_losses.get("train", [])
    ump_va = ump_losses.get("val", [])
    ump_metrics = ump_data.get("metrics", {}).get("val", [])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Left: Loss Curves
    ax1 = axes[0]
    if ump_tr:
        ep_tr, l_tr = zip(*ump_tr)
        ax1.plot(ep_tr, l_tr, label="Train Loss")
    if ump_va:
        ep_va, l_va = zip(*ump_va)
        ax1.plot(ep_va, l_va, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Unmasked Mean Pooling: Loss Curve")
    ax1.legend()
    style_ax(ax1)
    
    # Right: Metrics curves
    epochs, cwa_vals, swa_vals, pcwa_vals = [], [], [], []
    for ep, m in ump_metrics:
        epochs.append(ep)
        cwa_vals.append(m.get("CWA", 0))
        swa_vals.append(m.get("SWA", 0))
        pcwa_vals.append(m.get("PCWA", 0))
    ax2.plot(epochs, cwa_vals, marker="o", label="CWA")
    ax2.plot(epochs, swa_vals, marker="o", label="SWA")
    ax2.plot(epochs, pcwa_vals, marker="o", label="PCWA")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Unmasked Mean Pooling: Validation Metrics")
    ax2.legend()
    style_ax(ax2)
    
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "ablation_unmasked_mean_loss_metrics.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print(f"Unmasked Mean Pooling: Error plotting loss/metrics: {e}")
    plt.close("all")

# Plot L: Unmasked Mean Pooling – Confusion Matrix
try:
    y_true = ump_data.get("ground_truth", [])
    y_pred = ump_data.get("predictions", [])
    if y_true and y_pred:
        cm, labels = compute_confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.imshow(cm, cmap="Blues")
        plt.colorbar(cax, ax=ax)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Unmasked Mean Pooling: Confusion Matrix (Test)")
        style_ax(ax)
        fig.tight_layout()
        fig.savefig(os.path.join("figures", "ablation_unmasked_mean_confusion.png"), dpi=300)
        plt.close(fig)
    else:
        print("Unmasked Mean Pooling: Missing ground truth or predictions; skipping confusion matrix.")
except Exception as e:
    print(f"Unmasked Mean Pooling: Error plotting confusion matrix: {e}")
    plt.close("all")


print("All figures have been generated and saved in the 'figures/' directory.")
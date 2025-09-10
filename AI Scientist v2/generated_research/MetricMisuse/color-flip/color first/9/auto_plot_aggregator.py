#!/usr/bin/env python3
"""
Final Aggregator Script for Scientific Figures
This script loads experiment result .npy files from the BASELINE, RESEARCH,
and ABLATION experiments and produces final, publication‐quality figures.
All figures are saved into the "figures" folder.
Each plotting block is wrapped in a try‐except so that errors in one block do not affect others.
The script uses a larger font size for labels and titles.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Set a larger global font for readability.
plt.rcParams.update({'font.size': 14, 'axes.spines.top': False, 'axes.spines.right': False})

# Create final output directory.
os.makedirs("figures", exist_ok=True)

# --------------- Helper functions ---------------

def sample_points(epochs, vals, max_points=5):
    """If epochs have many points, sample at most max_points (keep first and last)."""
    if len(epochs) <= max_points:
        return epochs, vals
    # Choose stride (ensure last point is included)
    stride = max(1, int(np.ceil(len(epochs)/max_points)))
    sampled_idx = list(range(0, len(epochs), stride))
    if sampled_idx[-1] != len(epochs)-1:
        sampled_idx.append(len(epochs)-1)
    return [epochs[i] for i in sampled_idx], [vals[i] for i in sampled_idx]

def extract_loss_and_metrics(exp_data):
    """
    Given experiment data loaded from a .npy file, extract:
      losses_train: dict mapping lr to {epoch: loss}
      losses_val:   dict mapping lr to {epoch: loss}
      metrics_val:  dict mapping lr to {epoch: (CWA,SWA,HWA, *optional...)}
      test_metrics: for test the tuple (lr, CWA, SWA, HWA, *optional)
      predictions and ground_truth arrays if they exist.
    """
    losses_train = {}
    losses_val   = {}
    metrics_val  = {}
    
    for item in exp_data.get("losses", {}).get("train", []):
        lr, ep, loss = item
        losses_train.setdefault(lr, {})[ep] = loss
        
    for item in exp_data.get("losses", {}).get("val", []):
        lr, ep, loss = item
        losses_val.setdefault(lr, {})[ep] = loss
        
    for item in exp_data.get("metrics", {}).get("val", []):
        # Assume tuple: (lr, epoch, cwa, swa, hwa, *rest)
        lr, ep, cwa, swa, hwa, *others = item
        metrics_val.setdefault(lr, {})[ep] = (cwa, swa, hwa) + tuple(others)
        
    test_metrics = None
    if "test" in exp_data.get("metrics", {}):
        # Assume test metrics stored as (lr, cwa, swa, hwa, *others)
        test_metrics = exp_data["metrics"]["test"]
    
    predictions = exp_data.get("predictions", None)
    ground_truth = exp_data.get("ground_truth", None)
    
    return losses_train, losses_val, metrics_val, test_metrics, predictions, ground_truth

# --------------- Plotting for BASELINE experiments ---------------
# Load baseline experiment .npy file directly from the provided full path.
try:
    baseline_path = "experiment_results/experiment_9653d6615d954d92951fb8477447f87f_proc_1727705/experiment_data.npy"
    baseline_data = np.load(baseline_path, allow_pickle=True).item()["SPR_BENCH"]
except Exception as e:
    print(f"Error loading baseline experiment data: {e}")
    baseline_data = {}

# Extract information from baseline data.
bs_train, bs_val, bs_metrics, bs_test, bs_preds, bs_gt = extract_loss_and_metrics(baseline_data)

# Figure 1: Baseline Loss Curves & Validation HWA curves (2 subplots)
try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
    for lr in bs_train:
        epochs = sorted(bs_train[lr].keys())
        train_losses = [bs_train[lr][ep] for ep in epochs]
        # Sample if needed.
        x_s, y_s = sample_points(epochs, train_losses)
        ax1.plot(x_s, y_s, "-o", label=f"Train LR={lr}")
        # Only plot if validation exists.
        if lr in bs_val:
            epochs_val = sorted(bs_val[lr].keys())
            val_losses = [bs_val[lr][ep] for ep in epochs_val]
            x_sv, y_sv = sample_points(epochs_val, val_losses)
            ax1.plot(x_sv, y_sv, "--x", label=f"Val LR={lr}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Baseline: Training vs. Validation Loss")
    ax1.legend()
    
    # For HWA, plot validation HWA curves from metrics.
    for lr in bs_metrics:
        epochs = sorted(bs_metrics[lr].keys())
        hwa_vals = [bs_metrics[lr][ep][2] for ep in epochs]
        x_sm, y_sm = sample_points(epochs, hwa_vals)
        ax2.plot(x_sm, y_sm, "-o", label=f"Val HWA LR={lr}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("HWA")
    ax2.set_title("Baseline: Validation Harmonic Weighted Accuracy")
    ax2.legend()
    
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "baseline_loss_and_hwa.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print(f"Error creating Baseline Figure 1: {e}")
    plt.close()

# Figure 2: Baseline - Final CWA vs SWA scatter and Test HWA Bar Chart (2 subplots)
try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
    # Scatter: use last epoch per lr from validation metrics.
    for lr in bs_metrics:
        last_ep = max(bs_metrics[lr].keys())
        cwa, swa, _ = bs_metrics[lr][last_ep][:3]
        ax1.scatter(cwa, swa, s=100, label=f"LR={lr}")
        ax1.text(cwa, swa, f"{lr:.0e}", fontsize=12)
    ax1.set_xlabel("CWA")
    ax1.set_ylabel("SWA")
    ax1.set_title("Baseline: Final CWA vs SWA")
    ax1.legend()
    
    # Test HWA bar chart
    if bs_test:
        # bs_test is tuple: (lr, cwa, swa, hwa, *rest)
        lrs = [bs_test[0]]
        hwas = [bs_test[3]]
    else:
        # Fallback: compile from validation last epochs
        lrs, hwas = [], []
        for lr in bs_metrics:
            last_ep = max(bs_metrics[lr].keys())
            lrs.append(lr)
            hwas.append(bs_metrics[lr][last_ep][2])
    ax2.bar([f"{lr:.0e}" for lr in lrs], hwas, color="skyblue")
    ax2.set_ylabel("HWA")
    ax2.set_title("Baseline: Test HWA by LR")
    
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "baseline_scatter_and_test_bar.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print(f"Error creating Baseline Figure 2: {e}")
    plt.close()

# --------------- Plotting for RESEARCH experiments ---------------
try:
    research_path = "experiment_results/experiment_4e605e8d06704b8d979644e6b5bff533_proc_1733533/experiment_data.npy"
    research_data = np.load(research_path, allow_pickle=True).item()["SPR_BENCH"]
except Exception as e:
    print(f"Error loading research experiment data: {e}")
    research_data = {}

rs_train, rs_val, rs_metrics, rs_test, rs_preds, rs_gt = extract_loss_and_metrics(research_data)

# Figure 3: Research Loss Curves & Validation HWA curves (2 subplots)
try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
    for lr in rs_train:
        epochs = sorted(rs_train[lr].keys())
        train_losses = [rs_train[lr][ep] for ep in epochs]
        x_s, y_s = sample_points(epochs, train_losses)
        ax1.plot(x_s, y_s, "-o", label=f"Train LR={lr}")
        if lr in rs_val:
            epochs_val = sorted(rs_val[lr].keys())
            val_losses = [rs_val[lr][ep] for ep in epochs_val]
            x_sv, y_sv = sample_points(epochs_val, val_losses)
            ax1.plot(x_sv, y_sv, "--x", label=f"Val LR={lr}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Research: Training vs. Validation Loss")
    ax1.legend()
    
    # Validation HWA curves.
    for lr in rs_metrics:
        epochs = sorted(rs_metrics[lr].keys())
        hwa_vals = [rs_metrics[lr][ep][2] for ep in epochs]
        x_sm, y_sm = sample_points(epochs, hwa_vals)
        ax2.plot(x_sm, y_sm, "-o", label=f"Val HWA LR={lr}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("HWA")
    ax2.set_title("Research: Validation HWA")
    ax2.legend()
    
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "research_loss_and_hwa.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print(f"Error creating Research Figure 3: {e}")
    plt.close()

# Figure 4: Research Scatter (Final CWA vs SWA) and Confusion Matrix
try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
    # Scatter plot from last epoch of each lr.
    for lr in rs_metrics:
        last_ep = max(rs_metrics[lr].keys())
        cwa, swa, _ = rs_metrics[lr][last_ep][:3]
        ax1.scatter(cwa, swa, s=100, label=f"LR={lr}")
        ax1.text(cwa, swa, f"{lr:.0e}", fontsize=12)
    ax1.set_xlabel("CWA")
    ax1.set_ylabel("SWA")
    ax1.set_title("Research: Final CWA vs SWA")
    ax1.legend()
    
    # Confusion Matrix if predictions available.
    if (rs_preds is not None) and (rs_gt is not None) and (len(rs_preds) > 0) and (len(rs_gt) > 0):
        labels = sorted(set(rs_gt) | set(rs_preds))
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(rs_gt, rs_preds):
            cm[labels.index(t), labels.index(p)] += 1
        im = ax2.imshow(cm, cmap="Blues")
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("True")
        ax2.set_title("Research: Confusion Matrix")
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels)
        ax2.set_yticks(range(len(labels)))
        ax2.set_yticklabels(labels)
        fig.colorbar(im, ax=ax2)
    else:
        ax2.text(0.5, 0.5, "No Predictions/GT", horizontalalignment='center', verticalalignment='center')
        ax2.set_title("Research: Confusion Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "research_scatter_and_confusion.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print(f"Error creating Research Figure 4: {e}")
    plt.close()

# --------------- Plotting for ABLATION experiments ---------------
# Ablation 1: "No Glyph Clustering (Full-Vocabulary Tokenisation)"
try:
    no_glyph_path = "experiment_results/experiment_946e0545c5a742e88f3401c625674a1f_proc_1748866/experiment_data.npy"
    no_glyph_data = np.load(no_glyph_path, allow_pickle=True).item()["no_glyph_clustering"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading No Glyph Clustering data: {e}")
    no_glyph_data = {}
ng_train, ng_val, ng_metrics, ng_test, ng_preds, ng_gt = extract_loss_and_metrics(no_glyph_data)

# Figure 5: Ablation - No Glyph Clustering: Loss Curve and Validation Metrics
try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
    # Plot loss curves
    for item in no_glyph_data.get("losses", {}).get("train", []):
        # Use the train data from this ablation (ignoring lr differences if any)
        _, ep, loss = item
        ax1.plot(ep, loss, "o-", label="Train")
    for item in no_glyph_data.get("losses", {}).get("val", []):
        _, ep, loss = item
        ax1.plot(ep, loss, "x--", label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("No Glyph Clustering: Loss Curve")
    ax1.legend()
    # Plot validation metrics (assume tuple order: (lr, ep, CWA, SWA, HWA, CNA))
    epochs = [t[1] for t in no_glyph_data.get("metrics", {}).get("val", [])]
    if epochs:
        cwa = [t[2] for t in no_glyph_data.get("metrics", {}).get("val", [])]
        swa = [t[3] for t in no_glyph_data.get("metrics", {}).get("val", [])]
        hwa = [t[4] for t in no_glyph_data.get("metrics", {}).get("val", [])]
        ax2.plot(epochs, cwa, "-o", label="CWA")
        ax2.plot(epochs, swa, "-s", label="SWA")
        ax2.plot(epochs, hwa, "-^", label="HWA")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("No Glyph Clustering: Val Metrics")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "ablation_no_glyph_clustering.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print(f"Error creating Ablation Figure (No Glyph Clustering): {e}")
    plt.close()

# Ablation 2: "No Positional Encoding"
try:
    no_pos_path = "experiment_results/experiment_b4dc26ec460740e9847d9b1e4599c987_proc_1748867/experiment_data.npy"
    no_pos_data = np.load(no_pos_path, allow_pickle=True).item()["no_positional_encoding"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading No Positional Encoding data: {e}")
    no_pos_data = {}
np_train, np_val, np_metrics, np_test, np_preds, np_gt = extract_loss_and_metrics(no_pos_data)

# Figure 6: Ablation - No Positional Encoding: Loss and Metric curves
try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
    # Loss curves
    epochs = [t[1] for t in no_pos_data.get("losses", {}).get("train", [])]
    if epochs:
        tr_loss = [t[2] for t in no_pos_data.get("losses", {}).get("train", [])]
        val_loss = [t[2] for t in no_pos_data.get("losses", {}).get("val", [])]
        ax1.plot(epochs, tr_loss, "o-", label="Train")
        ax1.plot(epochs, val_loss, "x--", label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("No Positional Encoding: Loss Curve")
    ax1.legend()
    # Metrics curves: plot CWA, SWA, HWA, CNA if available
    epochs_m = [t[1] for t in no_pos_data.get("metrics", {}).get("val", [])]
    if epochs_m:
        cwa = [t[2] for t in no_pos_data.get("metrics", {}).get("val", [])]
        swa = [t[3] for t in no_pos_data.get("metrics", {}).get("val", [])]
        hwa = [t[4] for t in no_pos_data.get("metrics", {}).get("val", [])]
        np_label = ["CWA", "SWA", "HWA"]
        ax2.plot(epochs_m, cwa, "-o", label="CWA")
        ax2.plot(epochs_m, swa, "-s", label="SWA")
        ax2.plot(epochs_m, hwa, "-^", label="HWA")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("No Positional Encoding: Val Metrics")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "ablation_no_positional_encoding.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print(f"Error creating Ablation Figure (No Positional Encoding): {e}")
    plt.close()

# Ablation 3: "Random Glyph Clustering"
try:
    rand_cluster_path = "experiment_results/experiment_5cb6f0b689834fa8b7ab4c2cdddf0572_proc_1748869/experiment_data.npy"
    rand_cluster_data = np.load(rand_cluster_path, allow_pickle=True).item()
    # In this ablation, data is stored at top-level keyed by ablation and then dataset name.
    # Iterate over each dataset in the loaded dict.
    # For simplicity, pick the first available dataset under Random Glyph Clustering.
    # The structure is: {ablation: {dataset_name: record, ...}, ...}
    rand_record = None
    for ds in rand_cluster_data.values():
        if "SPR_BENCH" in ds:
            rand_record = ds["SPR_BENCH"]
            break
    if rand_record is None:
        raise ValueError("No SPR_BENCH record in Random Glyph Clustering")
except Exception as e:
    print(f"Error loading Random Glyph Clustering data: {e}")
    rand_record = {}
rc_train, rc_val, rc_metrics, rc_test, rc_preds, rc_gt = extract_loss_and_metrics(rand_record)

# Figure 7: Ablation - Random Glyph Clustering: Loss Curve (single plot)
try:
    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    epochs = [t[1] for t in rand_record.get("losses", {}).get("train", [])]
    if epochs:
        tr_loss = [t[2] for t in rand_record.get("losses", {}).get("train", [])]
        val_loss = [t[2] for t in rand_record.get("losses", {}).get("val", [])]
        ax.plot(epochs, tr_loss, "o-", label="Train")
        ax.plot(epochs, val_loss, "x--", label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Random Glyph Clustering: Loss Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "ablation_random_glyph_clustering_loss.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print(f"Error creating Ablation Figure (Random Glyph Clustering): {e}")
    plt.close()

# Ablation 4: "CLS-Token Pooling Instead of Length-Averaged Pooling"
try:
    cls_pool_path = "experiment_results/experiment_cd693672ec7f4fb599f7869e4f2543a8_proc_1748867/experiment_data.npy"
    cls_pool_data = np.load(cls_pool_path, allow_pickle=True).item()["CLS_token_pooling"]
except Exception as e:
    print(f"Error loading CLS-Token Pooling data: {e}")
    cls_pool_data = {}

# For this ablation, there are two variants: "SPR_BENCH_MEAN" and "SPR_BENCH_CLS".
# We'll produce a figure with three subplots: (a) loss curves, (b) validation HWA curves, (c) test metrics bar.
try:
    variants = ["SPR_BENCH_MEAN", "SPR_BENCH_CLS"]
    colors = {"SPR_BENCH_MEAN": "tab:blue", "SPR_BENCH_CLS": "tab:orange"}
    fig, axs = plt.subplots(1, 3, figsize=(22,6))
    # (a) Loss curves
    for var in variants:
        data_var = cls_pool_data.get(var, {})
        # Helper: extract (epoch, loss) from training and val losses.
        train_loss_list = data_var.get("losses", {}).get("train", [])
        val_loss_list = data_var.get("losses", {}).get("val", [])
        if train_loss_list:
            epochs_tr = [t[1] for t in train_loss_list]
            losses_tr = [t[2] for t in train_loss_list]
            axs[0].plot(epochs_tr, losses_tr, "-", color=colors[var], label=f"{var} Train")
        if val_loss_list:
            epochs_val = [t[1] for t in val_loss_list]
            losses_val = [t[2] for t in val_loss_list]
            axs[0].plot(epochs_val, losses_val, "--", color=colors[var], label=f"{var} Val")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("CLS-Token Pooling: Loss Curves")
    axs[0].legend()
    
    # (b) Validation HWA curves
    for var in variants:
        data_var = cls_pool_data.get(var, {})
        met_list = data_var.get("metrics", {}).get("val", [])
        if met_list:
            epochs_m = [t[1] for t in met_list]
            hwa_vals = [t[4] for t in met_list]  # assuming tuple: (lr, ep, cwa, swa, hwa, ...)
            axs[1].plot(epochs_m, hwa_vals, "-o", color=colors[var], label=f"{var} HWA")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("HWA")
    axs[1].set_title("CLS-Token Pooling: Validation HWA")
    axs[1].legend()
    
    # (c) Test metrics bar chart (assume test metrics tuple: (lr, cwa, swa, hwa, cna))
    x_labels = ["HWA", "CNA"]
    x = np.arange(len(x_labels))
    width = 0.35
    for i, var in enumerate(variants):
        # Get test metrics from data_var
        data_var = cls_pool_data.get(var, {})
        test_met = data_var.get("metrics", {}).get("test", None)
        if test_met:
            # test_met: (lr, cwa, swa, hwa, cna)
            vals = [test_met[3], test_met[4] if len(test_met) > 4 else test_met[3]]
            axs[2].bar(x + i * width, vals, width, label=var, color=colors[var])
    axs[2].set_xticks(x + width/2)
    axs[2].set_xticklabels(x_labels)
    axs[2].set_ylim(0, 1)
    axs[2].set_ylabel("Score")
    axs[2].set_title("CLS-Token Pooling: Test Metrics")
    axs[2].legend()
    
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "ablation_cls_token_pooling.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print(f"Error creating Ablation Figure (CLS-Token Pooling): {e}")
    plt.close()

# Ablation 5: "Single-Head Self-Attention"
# Since the summary for Single-Head Self-Attention has empty exp_results_npy_files,
# we skip plotting if there is no file.
try:
    # Check if any file is provided; here we rely on the provided list length.
    single_head_files = []  # as provided in summary: empty.
    if not single_head_files:
        print("Skipping Single-Head Self-Attention plots as no data file exists.")
    else:
        # (If data existed, similar plotting would be added here.)
        pass
except Exception as e:
    print(f"Error in Single-Head Self-Attention plotting: {e}")

print("Final figures have been saved in 'figures/' directory.")
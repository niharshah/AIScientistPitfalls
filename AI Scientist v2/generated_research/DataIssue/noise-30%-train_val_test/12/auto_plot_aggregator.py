#!/usr/bin/env python3
"""
Aggregator script for final scientific plots for the SPR_BENCH experiments.
This script loads experiment summary .npy files from various experiments 
(baseline/research and ablation studies) and generates a set of final figures 
(saved under "figures/"). Each plotting section is wrapped in try-except blocks.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Set a higher base font size for publication quality
plt.rcParams.update({'font.size': 14})
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# Create output folder for final figures
os.makedirs("figures", exist_ok=True)

##############################
# Group 1: Baseline / Research (common file)
##############################

base_file = "experiment_results/experiment_b32bdd5b53a343c49f095e778e93bb97_proc_3462725/experiment_data.npy"
try:
    base_data = np.load(base_file, allow_pickle=True).item()
    bs_data = base_data.get("batch_size", {})
except Exception as e:
    print(f"Error loading baseline file: {e}")
    bs_data = {}

def get_best_f1_from(bs_stats):
    arr = np.array(bs_stats.get("metrics", {}).get("val_f1", []))
    return arr.max() if arr.size else float('nan')

# Figure 1: Training Loss Curves (Baseline/Research)
try:
    plt.figure(figsize=(8,6))
    for bs, stats in sorted(bs_data.items(), key=lambda x: int(x[0])):
        epochs = np.array(stats.get("epochs"))
        train_losses = np.array(stats.get("losses", {}).get("train"))
        plt.plot(epochs, train_losses, marker="o", label=f"Batch Size {bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training Loss vs Epoch (Baseline)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "baseline_train_loss.png"), dpi=300)
    plt.close()
except Exception as e:
    print(f"Error creating Figure 1 (Training Loss): {e}")
    plt.close()

# Figure 2: Validation Loss Curves (Baseline/Research)
try:
    plt.figure(figsize=(8,6))
    for bs, stats in sorted(bs_data.items(), key=lambda x: int(x[0])):
        epochs = np.array(stats.get("epochs"))
        val_losses = np.array(stats.get("losses", {}).get("val"))
        plt.plot(epochs, val_losses, marker="s", label=f"Batch Size {bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Validation Loss vs Epoch (Baseline)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "baseline_val_loss.png"), dpi=300)
    plt.close()
except Exception as e:
    print(f"Error creating Figure 2 (Validation Loss): {e}")
    plt.close()

# Figure 3: Validation Macro-F1 Curves (Baseline/Research)
try:
    plt.figure(figsize=(8,6))
    for bs, stats in sorted(bs_data.items(), key=lambda x: int(x[0])):
        epochs = np.array(stats.get("epochs"))
        val_f1 = np.array(stats.get("metrics", {}).get("val_f1"))
        plt.plot(epochs, val_f1, marker="^", label=f"Batch Size {bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title("Validation Macro F1 vs Epoch (Baseline)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "baseline_val_f1.png"), dpi=300)
    plt.close()
except Exception as e:
    print(f"Error creating Figure 3 (Validation F1): {e}")
    plt.close()

# Figure 4: Best Validation Macro-F1 Bar Chart (Baseline/Research)
try:
    bs_list = []
    f1_best_list = []
    for bs, stats in sorted(bs_data.items(), key=lambda x: int(x[0])):
        bs_list.append(bs)
        f1_best_list.append(get_best_f1_from(stats))
    plt.figure(figsize=(8,6))
    plt.bar(bs_list, f1_best_list, color="lightblue")
    plt.xlabel("Batch Size")
    plt.ylabel("Best Macro F1")
    plt.title("Best Validation Macro F1 by Batch Size (Baseline)")
    for i, v in enumerate(f1_best_list):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "baseline_best_val_f1.png"), dpi=300)
    plt.close()
except Exception as e:
    print(f"Error creating Figure 4 (Best F1 Bar Chart): {e}")
    plt.close()

##############################
# Group 2: No-Padding-Mask Ablation
##############################

no_pad_file = "experiment_results/experiment_9d5d07360666442cba5c1d791b629371_proc_3475577/experiment_data.npy"
try:
    no_pad_all = np.load(no_pad_file, allow_pickle=True).item()
    no_pad_data = no_pad_all.get("no_padding_mask_ablation", {})
except Exception as e:
    print(f"Error loading no-padding-mask file: {e}")
    no_pad_data = {}

# Figure 5: Dual Subplot for Loss Curves (Train & Val) for No-Padding-Mask Ablation
try:
    colors = plt.cm.tab10.colors
    fig, axs = plt.subplots(1, 2, figsize=(12,5))
    for i, (bs, stats) in enumerate(sorted(no_pad_data.items(), key=lambda x: int(x[0]))):
        epochs = np.array(stats.get("epochs"))
        train_losses = np.array(stats.get("losses", {}).get("train"))
        val_losses = np.array(stats.get("losses", {}).get("val"))
        axs[0].plot(epochs, train_losses, marker="o", color=colors[i % len(colors)], label=f"Batch {bs}")
        axs[1].plot(epochs, val_losses, marker="s", color=colors[i % len(colors)], label=f"Batch {bs}")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Training Loss (No-Pad Ablation)")
    axs[0].legend()
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].set_title("Validation Loss (No-Pad Ablation)")
    axs[1].legend()
    fig.suptitle("No-Padding-Mask Ablation: Loss Curves")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join("figures", "nopad_loss_curves.png"), dpi=300)
    plt.close()
except Exception as e:
    print(f"Error creating Figure 5 (No-Pad Loss Curves): {e}")
    plt.close()

# Figure 6: Validation Macro-F1 Curves for No-Padding-Mask Ablation
try:
    plt.figure(figsize=(8,6))
    for i, (bs, stats) in enumerate(sorted(no_pad_data.items(), key=lambda x: int(x[0]))):
        epochs = np.array(stats.get("epochs"))
        val_f1 = np.array(stats.get("metrics", {}).get("val_f1"))
        plt.plot(epochs, val_f1, marker="^", color=colors[i % len(colors)], label=f"Batch {bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title("Validation Macro F1 vs Epoch (No-Pad Ablation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "nopad_val_f1.png"), dpi=300)
    plt.close()
except Exception as e:
    print(f"Error creating Figure 6 (No-Pad F1 Curves): {e}")
    plt.close()

# Figure 7: Final Epoch Validation F1 Bar Chart (No-Pad Ablation)
try:
    bs_list_np = []
    final_f1_list = []
    for bs, stats in sorted(no_pad_data.items(), key=lambda x: int(x[0])):
        bs_list_np.append(bs)
        f1_values = np.array(stats.get("metrics", {}).get("val_f1"))
        final_f1 = f1_values[-1] if f1_values.size else float('nan')
        final_f1_list.append(final_f1)
    plt.figure(figsize=(8,6))
    plt.bar(bs_list_np, final_f1_list, color="salmon")
    plt.xlabel("Batch Size")
    plt.ylabel("Final Epoch Macro F1")
    plt.title("Final Validation Macro F1 by Batch Size (No-Pad Ablation)")
    for i, v in enumerate(final_f1_list):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "nopad_final_val_f1.png"), dpi=300)
    plt.close()
except Exception as e:
    print(f"Error creating Figure 7 (No-Pad Final F1 Bar Chart): {e}")
    plt.close()

##############################
# Group 3: Single-Head-Attention Ablation (Aggregated Best F1 Bar Chart)
##############################

single_head_file = "experiment_results/experiment_7fea0f89632e4f26a4b7544e5ceb90d6_proc_3475578/experiment_data.npy"
try:
    single_head_data = np.load(single_head_file, allow_pickle=True).item()
    # Assume the file contains entries for different ablations (e.g. "multi_head_4" and "single_head")
    best_scores = {}
    for ablation_name, run in single_head_data.items():
        # For each, assume run is a dict with key "SPR-BENCH"
        spr = run.get("SPR-BENCH", {})
        f1_vals = np.array(spr.get("metrics", {}).get("val_f1", []))
        if f1_vals.size:
            best_scores[ablation_name] = f1_vals.max()
except Exception as e:
    print(f"Error loading single-head attention data: {e}")
    best_scores = {}

# Figure 8: Aggregated Bar Chart for Best Validation F1 (Single-Head vs Multi-Head)
try:
    if best_scores:
        names = list(best_scores.keys())
        scores = [best_scores[name] for name in names]
        plt.figure(figsize=(8,6))
        plt.bar(names, scores, color="mediumseagreen")
        plt.xlabel("Ablation Type")
        plt.ylabel("Best Macro F1")
        plt.title("Best Validation Macro F1 Across Ablations (Attention Variants)")
        for i, s in enumerate(scores):
            plt.text(i, s + 0.01, f"{s:.2f}", ha="center")
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "singlehead_best_val_f1.png"), dpi=300)
        plt.close()
except Exception as e:
    print(f"Error creating Figure 8 (Single-Head Best F1 Bar): {e}")
    plt.close()

##############################
# Group 4: No-FeedForward-Ablation (Combined Loss & F1 curves)
##############################

feedforward_file = "experiment_results/experiment_abcbd94041774ef5b328ad514d064b2a_proc_3475578/experiment_data.npy"
try:
    ff_data_all = np.load(feedforward_file, allow_pickle=True).item()
    # Pick one experiment/dataset from the dict (assume first key and then "SPR-BENCH")
    exp_key = list(ff_data_all.keys())[0]
    dset = ff_data_all[exp_key].get("SPR-BENCH", {})
    ff_bs_data = dset.get("batch_size", {})
except Exception as e:
    print(f"Error loading No-FeedForward-Ablation data: {e}")
    ff_bs_data = {}

# Figure 9: Combined plot with two subplots (Loss curves and F1 curves) for No-FeedForward-Ablation
try:
    fig, axs = plt.subplots(2, 1, figsize=(8,10))
    for bs, stats in sorted(ff_bs_data.items(), key=lambda x: int(x[0])):
        epochs = np.array(stats.get("epochs"))
        # Plot Loss curves: train and val on same axes
        train_loss = np.array(stats.get("losses", {}).get("train"))
        val_loss = np.array(stats.get("losses", {}).get("val"))
        axs[0].plot(epochs, train_loss, marker="o", label=f"Train Loss (bs {bs})")
        axs[0].plot(epochs, val_loss, marker="s", linestyle="--", label=f"Val Loss (bs {bs})")
        # Plot F1 curves
        f1_curve = np.array(stats.get("metrics", {}).get("val", []))
        axs[1].plot(epochs, f1_curve, marker="^", label=f"Val F1 (bs {bs})")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Train & Validation Loss (No-FeedForward Ablation)")
    axs[0].legend()
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Macro F1")
    axs[1].set_title("Validation Macro F1 vs Epoch (No-FeedForward Ablation)")
    axs[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "no_feedforward_combined.png"), dpi=300)
    plt.close()
except Exception as e:
    print(f"Error creating Figure 9 (No-FeedForward combined plot): {e}")
    plt.close()

##############################
# Group 5: Frozen-Embedding Ablation
##############################

frozen_file = "experiment_results/experiment_9759bdd5085b4370985595c54a4033dc_proc_3475577/experiment_data.npy"
try:
    frozen_all = np.load(frozen_file, allow_pickle=True).item()
    frozen_data = frozen_all.get("frozen_embedding", {}).get("SPR_BENCH", {}).get("batch_size", {})
except Exception as e:
    print(f"Error loading Frozen-Embedding data: {e}")
    frozen_data = {}

# Figure 10: Combined Loss Curves (Train and Val) for Frozen-Embedding Ablation
try:
    plt.figure(figsize=(8,6))
    for bs, stats in sorted(frozen_data.items(), key=lambda x: int(x[0])):
        epochs = np.array(stats.get("epochs"))
        train_loss = np.array(stats.get("losses", {}).get("train"))
        val_loss = np.array(stats.get("losses", {}).get("val"))
        plt.plot(epochs, train_loss, marker="o", linestyle="-", label=f"Train Loss (bs {bs})")
        plt.plot(epochs, val_loss, marker="s", linestyle="--", label=f"Val Loss (bs {bs})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Combined Loss Curves (Frozen-Embedding)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "frozen_loss_curves.png"), dpi=300)
    plt.close()
except Exception as e:
    print(f"Error creating Figure 10 (Frozen Loss Curves): {e}")
    plt.close()

# Figure 11: Combined Validation Macro-F1 Curves for Frozen-Embedding Ablation
try:
    plt.figure(figsize=(8,6))
    for bs, stats in sorted(frozen_data.items(), key=lambda x: int(x[0])):
        epochs = np.array(stats.get("epochs"))
        val_f1 = np.array(stats.get("metrics", {}).get("val_f1"))
        plt.plot(epochs, val_f1, marker="^", label=f"Val F1 (bs {bs})")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title("Validation Macro F1 vs Epoch (Frozen-Embedding)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "frozen_val_f1.png"), dpi=300)
    plt.close()
except Exception as e:
    print(f"Error creating Figure 11 (Frozen Val F1): {e}")
    plt.close()

# Figure 12: Final Epoch Validation Macro-F1 Bar Chart for Frozen-Embedding Ablation
try:
    bs_list_frozen = []
    final_f1_frozen = []
    for bs, stats in sorted(frozen_data.items(), key=lambda x: int(x[0])):
        bs_list_frozen.append(bs)
        f1_vals = np.array(stats.get("metrics", {}).get("val_f1"))
        final_f1 = f1_vals[-1] if f1_vals.size else float('nan')
        final_f1_frozen.append(final_f1)
    plt.figure(figsize=(8,6))
    plt.bar(bs_list_frozen, final_f1_frozen, color="plum")
    plt.xlabel("Batch Size")
    plt.ylabel("Final Epoch Macro F1")
    plt.title("Final Validation Macro F1 by Batch Size (Frozen-Embedding)")
    for i, v in enumerate(final_f1_frozen):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "frozen_final_val_f1.png"), dpi=300)
    plt.close()
except Exception as e:
    print(f"Error creating Figure 12 (Frozen Final F1 Bar): {e}")
    plt.close()

print("Final figures generated and saved in the 'figures/' folder.")
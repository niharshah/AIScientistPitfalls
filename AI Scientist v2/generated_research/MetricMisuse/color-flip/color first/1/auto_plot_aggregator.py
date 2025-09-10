#!/usr/bin/env python3
"""
Final aggregator script for publication‐ready figures.
This script loads existing .npy experiment data from the Baseline, Research, and Ablation experiments
and produces final, unique figures saved in the "figures/" directory.
All figures have descriptive titles and legends (with no underscores) and aggregate related plots
into a single figure when appropriate.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Global font size for readability in final PDF
plt.rcParams.update({'font.size': 16})

# Create output directory for final figures
os.makedirs("figures", exist_ok=True)


# ------------------------- Helper Functions -------------------------

def load_experiment(file_path):
    """Load experiment data from the given npy file path."""
    try:
        data = np.load(file_path, allow_pickle=True).item()
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}


def add_clean_axes(ax):
    """Remove top and right spines for a clean appearance."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fix_xtick_labels(ax, labels):
    """Set xticks and fix labels by replacing underscores with spaces."""
    ax.set_xticks(range(len(labels)))
    fixed_labels = [lab.replace("_", " ") for lab in labels]
    ax.set_xticklabels(fixed_labels, rotation=45, ha="right")


# ------------------------- Baseline Figure (Main Paper) -------------------------
def plot_baseline():
    """
    Baseline experiment varies hidden dimensions.
    Data file:
      experiment_results/experiment_6f2670dcd72e48358f7073da2fb945be_proc_1604393/experiment_data.npy
    Produces an aggregated figure with three subplots:
      A: Train and Validation Loss curves
      B: Validation HMWA over Epochs
      C: Test HMWA (Bar Chart)
    """
    file_path = "experiment_results/experiment_6f2670dcd72e48358f7073da2fb945be_proc_1604393/experiment_data.npy"
    data = load_experiment(file_path)
    if not data:
        print("No baseline data found.")
        return

    # The correct key is "SPR_BENCH" (with an underscore) as in the summary.
    tags = list(data.keys())
    if not tags:
        print("Baseline experiment data empty.")
        return

    epochs_dict, train_loss, val_loss, val_hmw, test_hmw = {}, {}, {}, {}, {}
    for tag in tags:
        try:
            ed = data[tag]["SPR_BENCH"]
            train_loss[tag] = ed["losses"]["train"]
            val_loss[tag] = ed["losses"]["val"]
            epochs_dict[tag] = list(range(1, len(train_loss[tag]) + 1))
            val_hmw[tag] = [m["hmwa"] for m in ed["metrics"]["val"]]
            test_hmw[tag] = ed["metrics"]["test"]["hmwa"]
        except Exception as e:
            print(f"Error processing tag {tag}: {e}")
            continue

    # If some tags were skipped, only plot those with complete data.
    valid_tags = [tag for tag in tags if tag in train_loss and tag in val_loss and tag in epochs_dict]

    if not valid_tags:
        print("No valid baseline tags to plot.")
        return

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=300)

    # Subplot A: Loss Curves
    for tag in valid_tags:
        axs[0].plot(epochs_dict[tag], train_loss[tag], label=f"{tag} Train")
        axs[0].plot(epochs_dict[tag], val_loss[tag], linestyle="--", label=f"{tag} Validation")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Cross Entropy")
    axs[0].set_title("Loss Curves")
    add_clean_axes(axs[0])
    axs[0].legend()

    # Subplot B: Validation HMWA vs Epoch
    for tag in valid_tags:
        axs[1].plot(epochs_dict[tag], val_hmw[tag], label=f"{tag}")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("HMWA")
    axs[1].set_title("Validation HMWA")
    add_clean_axes(axs[1])
    axs[1].legend()

    # Subplot C: Test HMWA Bar Chart
    names = [tag for tag in valid_tags if tag in test_hmw]
    scores = [test_hmw[tag] for tag in names]
    axs[2].bar(range(len(names)), scores, color="skyblue")
    axs[2].set_ylabel("HMWA")
    axs[2].set_title("Test HMWA by Hidden Dimension")
    fix_xtick_labels(axs[2], names)
    add_clean_axes(axs[2])

    fig.suptitle("Baseline Experiment (SPR BENCH)", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_file = os.path.join("figures", "Figure_1_Baseline.png")
    plt.savefig(out_file)
    plt.close(fig)
    print("Saved", out_file)


# ------------------------- Research Figure (Main Paper) -------------------------
def plot_research():
    """
    Research experiment using advanced representation learning.
    Data file:
      experiment_results/experiment_750ef8765be7495db4f73f0abc9ea3fd_proc_1608752/experiment_data.npy
    Produces an aggregated figure with three subplots:
      A: Train and Validation Loss curves
      B: Validation Metrics (CWA, SWA, CVA) over Epochs
      C: Grouped Test Metrics (Bar Chart)
    """
    file_path = "experiment_results/experiment_750ef8765be7495db4f73f0abc9ea3fd_proc_1608752/experiment_data.npy"
    data = load_experiment(file_path)
    if not data:
        print("No research data found.")
        return

    tags = list(data.keys())
    if not tags:
        print("Research experiment data empty.")
        return

    loss_train, loss_val = {}, {}
    val_cwa, val_swa, val_cva, epochs_dict = {}, {}, {}, {}
    test_metrics = {}
    for tag in tags:
        try:
            ed = data[tag]["SPR"]
            loss_train[tag] = ed["losses"]["train"]
            loss_val[tag] = ed["losses"]["val"]
            epochs_dict[tag] = list(range(1, len(loss_train[tag]) + 1))
            vm = ed["metrics"]["val"]
            val_cwa[tag] = [m["cwa"] for m in vm]
            val_swa[tag] = [m["swa"] for m in vm]
            val_cva[tag] = [m["cva"] for m in vm]
            test_metrics[tag] = ed["metrics"].get("test", {})
        except Exception as e:
            print(f"Error processing research tag {tag}: {e}")
            continue

    valid_tags = [tag for tag in tags if tag in loss_train and tag in epochs_dict]

    if not valid_tags:
        print("No valid research tags to plot.")
        return

    fig, axs = plt.subplots(1, 3, figsize=(20, 6), dpi=300)

    # Subplot A: Loss Curves
    for tag in valid_tags:
        axs[0].plot(epochs_dict[tag], loss_train[tag], label=f"{tag} Train")
        axs[0].plot(epochs_dict[tag], loss_val[tag], linestyle="--", label=f"{tag} Validation")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Cross Entropy")
    axs[0].set_title("Loss Curves")
    add_clean_axes(axs[0])
    axs[0].legend()

    # Subplot B: Validation Metrics (CWA, SWA, CVA)
    for tag in valid_tags:
        axs[1].plot(epochs_dict[tag], val_cwa[tag], label=f"{tag} CWA")
        axs[1].plot(epochs_dict[tag], val_swa[tag], label=f"{tag} SWA")
        axs[1].plot(epochs_dict[tag], val_cva[tag], label=f"{tag} CVA")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_title("Validation Metrics")
    add_clean_axes(axs[1])
    axs[1].legend()

    # Subplot C: Grouped Test Metrics
    n = len(valid_tags)
    width = 0.25
    indices = np.arange(n)
    cwa_vals = [test_metrics[tag].get("cwa", 0) for tag in valid_tags]
    swa_vals = [test_metrics[tag].get("swa", 0) for tag in valid_tags]
    cva_vals = [test_metrics[tag].get("cva", 0) for tag in valid_tags]
    axs[2].bar(indices - width, cwa_vals, width, label="CWA", color="salmon")
    axs[2].bar(indices, swa_vals, width, label="SWA", color="lightgreen")
    axs[2].bar(indices + width, cva_vals, width, label="CVA", color="skyblue")
    axs[2].set_ylim(0, 1)
    axs[2].set_ylabel("Accuracy")
    axs[2].set_title("Test Metrics")
    fix_xtick_labels(axs[2], valid_tags)
    add_clean_axes(axs[2])
    axs[2].legend()

    fig.suptitle("Research Experiment (SPR)", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_file = os.path.join("figures", "Figure_2_Research.png")
    plt.savefig(out_file)
    plt.close(fig)
    print("Saved", out_file)


# ------------------------- Ablation Figures (Appendix) -------------------------
def plot_ablation_loss(tag_key, ds_key, out_name, display_title):
    """
    Generic function to plot loss curves for an ablation experiment.
    It produces a single plot comparing Training and Validation Loss.
    """
    mapping = {
        "no pos emb": "experiment_results/experiment_9cc1b16eb35f4bbbabe7798178e0d638_proc_1614362/experiment_data.npy",
        "no transformer mean pool": "experiment_results/experiment_6e24840372594178bc0e890baca9c419_proc_1614363/experiment_data.npy",
        "no color embedding": "experiment_results/experiment_4c90f58aaaa74ffea835b1cefa9027da_proc_1614365/experiment_data.npy",
        "frozen embeddings": "experiment_results/experiment_5b81886804194508aab11348677db6bf_proc_1614364/experiment_data.npy",
        "no shape embedding": "experiment_results/experiment_b6f321e2d6a74fde8d51298e294c875d_proc_1614362/experiment_data.npy"
    }
    # For Single-Head Self-Attention (MonoHead), use its own mapping.
    if tag_key == "monohead":
        file_path = "experiment_results/experiment_f5613f032c9b4f2c8518964b5b5a4dc1_proc_1614365/experiment_data.npy"
    else:
        file_path = mapping.get(tag_key, None)
    if file_path is None:
        print(f"No file mapping for tag {tag_key}")
        return

    data = load_experiment(file_path)
    if tag_key == "monohead":
        try:
            ed = data["monohead"][ds_key]
        except Exception as e:
            print("Error extracting monohead data:", e)
            return
    else:
        try:
            ed = data[tag_key][ds_key]
        except Exception as e:
            print(f"Error extracting data for {tag_key}:", e)
            return

    try:
        train_losses = ed.get("losses", {}).get("train", [])
        val_losses = ed.get("losses", {}).get("val", [])
        if not train_losses or not val_losses:
            print(f"No loss data for {display_title}")
            return
        epochs = list(range(1, len(train_losses) + 1))
    except Exception as e:
        print(f"Error retrieving losses for {display_title}: {e}")
        return

    try:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        ax.plot(epochs, train_losses, label="Train")
        ax.plot(epochs, val_losses, label="Validation", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross Entropy")
        ax.set_title(display_title + " Loss Curves")
        add_clean_axes(ax)
        ax.legend()
        fig.tight_layout()
        out_file = os.path.join("figures", out_name)
        plt.savefig(out_file)
        plt.close(fig)
        print("Saved", out_file)
    except Exception as e:
        print(f"Error plotting loss for {display_title}: {e}")
        plt.close()


def plot_ablation():
    """
    Produce unique ablation loss figures for the Appendix.
    Total ablation plots will be 6, each unique.
    """
    plot_ablation_loss("no pos emb", "SPR", "Figure_A1_No_Pos_Emb_Loss.png", "Ablation No Position Embeddings")
    plot_ablation_loss("no transformer mean pool", "SPR", "Figure_A2_No_Transformer_Mean_Pool_Loss.png", "Ablation No Transformer (Mean Pool)")
    plot_ablation_loss("no color embedding", "SPR", "Figure_A3_No_Color_Emb_Loss.png", "Ablation No Color Embedding")
    plot_ablation_loss("frozen embeddings", "SPR", "Figure_A4_Frozen_Embeddings_Loss.png", "Ablation Frozen Embeddings")
    plot_ablation_loss("no shape embedding", "SPR", "Figure_A5_No_Shape_Emb_Loss.png", "Ablation No Shape Embedding")
    plot_ablation_loss("monohead", "SPR", "Figure_A6_Monohead_Loss.png", "Ablation Single Head Self Attention")


# ------------------------- Main Aggregator -------------------------
def main():
    print("Generating final publication figures...")
    # Main paper figures -- ensure no duplicates and all labels are descriptive.
    plot_baseline()   # Aggregated Baseline Figure (Figure 1)
    plot_research()   # Aggregated Research Figure (Figure 2)

    # Ablation figures (Appendix): 6 unique figures
    plot_ablation()

    # Total figures: 2 main + 6 ablation = 8 figures (< 12)
    print("All figures saved in the 'figures/' directory.")


if __name__ == "__main__":
    main()
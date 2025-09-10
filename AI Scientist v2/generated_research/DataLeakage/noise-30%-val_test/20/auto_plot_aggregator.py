#!/usr/bin/env python3
"""
Final Aggregator Script for SPR Research Paper Figures

This script loads finalized experiment results from .npy files (paths taken exactly
from experiment summaries) and aggregates them into a set of publication‐ready figures,
which are saved into the folder 'figures/'.
Each figure (or sub-figure group) is wrapped in a try/except block so that one failure
does not block the remaining plots.

The experiments include:
  • Baseline experiments (using different batch sizes)
  • Research experiments (Transformer based model with additional metrics)
  • Ablation studies:
       - No Curriculum Complexity Weighting
       - Remove Learned Positional Embedding (Mean-Pooling Read-out)
       - No Label Smoothing
       - Remove Gradient Clipping
       - Replace Transformer Encoder with Bi-LSTM Backbone

All figures use increased font sizes and have professional styling.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Increase font size and set dpi for professional quality.
plt.rcParams.update({'font.size': 14, 'figure.dpi': 300})

# Helper to remove top/right spines from an axis.
def remove_spines(ax):
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

# Create the figures/ directory.
os.makedirs("figures", exist_ok=True)

############################
# Baseline Plots Functions #
############################
def plot_baseline():
    # Load Baseline experiment data.
    # File path from summary:
    baseline_file = "experiment_results/experiment_e9901d40fea04cdb8730bd3a5d6ea854_proc_3442580/experiment_data.npy"
    try:
        data = np.load(baseline_file, allow_pickle=True).item()
        # Structure: data["batch_size"]["SPR_BENCH"]
        exp = data["batch_size"]["SPR_BENCH"]
        batch_sizes = sorted([int(k.split("_")[-1]) for k in exp.keys()])
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    except Exception as e:
        print("Error loading baseline data:", e)
        return

    # Figure 1: Loss Curves and Macro-F1 curves (2 subplots)
    try:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        # Left subplot: Training and Validation Loss curves for each batch size.
        ax = axs[0]
        for c, bs in zip(colors, batch_sizes):
            logs = exp[f"bs_{bs}"]
            epochs = np.arange(1, len(logs["losses"]["train"]) + 1)
            ax.plot(epochs, logs["losses"]["train"], color=c, linestyle="-", label=f"Train bs={bs}")
            ax.plot(epochs, logs["losses"]["val"], color=c, linestyle="--", label=f"Val bs={bs}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.set_title("Baseline: Loss Curves")
        ax.legend()
        remove_spines(ax)
        
        # Right subplot: Macro-F1 curves for each batch size.
        ax = axs[1]
        for c, bs in zip(colors, batch_sizes):
            logs = exp[f"bs_{bs}"]
            epochs = np.arange(1, len(logs["metrics"]["val"]) + 1)
            ax.plot(epochs, logs["metrics"]["val"], color=c, label=f"Val F1 bs={bs}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Macro-F1")
        ax.set_title("Baseline: Macro-F1 Curves")
        ax.legend()
        remove_spines(ax)
        
        fig.tight_layout()
        fig.savefig(os.path.join("figures", "Baseline_loss_and_f1.png"))
        plt.close(fig)
    except Exception as e:
        print("Baseline Loss and F1 plot error:", e)

    # Figure 2: Final Macro-F1 Bar Plot and Confusion Matrix (2 subplots)
    try:
        # Reuse baseline data.
        final_f1 = [exp[f"bs_{bs}"]["metrics"]["val"][-1] for bs in batch_sizes]
        best_idx = int(np.argmax(final_f1))
        best_bs = batch_sizes[best_idx]
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        # Left: Bar plot of final Macro-F1 scores.
        ax = axs[0]
        ax.bar([str(bs) for bs in batch_sizes], final_f1, color=colors[:len(batch_sizes)])
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Final Macro-F1")
        ax.set_title("Baseline: Final Macro-F1 by Batch Size")
        for bs, f1_val in zip(batch_sizes, final_f1):
            ax.text(str(bs), f1_val + 0.01, f"{f1_val:.2f}", ha="center", va="bottom")
        remove_spines(ax)
        
        # Right: Confusion Matrix for the best batch size.
        logs = exp[f"bs_{best_bs}"]
        preds = np.array(logs["predictions"])
        gts = np.array(logs["ground_truth"])
        cm = np.zeros((2, 2), dtype=int)
        for p, t in zip(preds, gts):
            cm[t, p] += 1
        ax = axs[1]
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Baseline: Confusion Matrix (bs={best_bs})")
        plt.colorbar(im, ax=ax)
        remove_spines(ax)
        
        fig.tight_layout()
        fig.savefig(os.path.join("figures", "Baseline_final_f1_and_confusion.png"))
        plt.close(fig)
    except Exception as e:
        print("Baseline Final F1 and Confusion plot error:", e)

##########################
# Research Plots Methods #
##########################
def plot_research():
    # File path from summary:
    research_file = "experiment_results/experiment_ae43f6b6b7cd4aa384820f4d0632a952_proc_3445459/experiment_data.npy"
    try:
        data = np.load(research_file, allow_pickle=True).item()
        # Assume the data dict keys represent dataset names. Use the first key.
        keys = list(data.keys())
        if not keys:
            raise ValueError("Empty research data.")
        dname = keys[0]
        logs = data[dname]
        epochs = np.arange(1, len(logs["losses"]["train"]) + 1)
    except Exception as e:
        print("Error loading research data:", e)
        return

    # Figure 3: Research Loss, Macro-F1, and Complexity-Weighted Accuracy curves (3 subplots)
    try:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        # Loss curves.
        ax = axs[0]
        ax.plot(epochs, logs["losses"]["train"], label="Train", color="tab:blue")
        ax.plot(epochs, logs["losses"]["val"], label="Val", color="tab:orange", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.set_title(f"Research {dname}: Loss Curves")
        ax.legend()
        remove_spines(ax)
        
        # Macro-F1 curve.
        ax = axs[1]
        macro_f1 = np.array([m["macro_f1"] for m in logs["metrics"]["val"]])
        ax.plot(epochs, macro_f1, color="tab:green")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Macro-F1")
        ax.set_title(f"Research {dname}: Macro-F1")
        remove_spines(ax)
        
        # Complexity-Weighted Accuracy (CWA) curve.
        ax = axs[2]
        cwa = np.array([m["cwa"] for m in logs["metrics"]["val"]])
        ax.plot(epochs, cwa, color="tab:red")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("CWA")
        ax.set_title(f"Research {dname}: CWA")
        remove_spines(ax)
        
        fig.tight_layout()
        fig.savefig(os.path.join("figures", "Research_loss_f1_cwa.png"))
        plt.close(fig)
    except Exception as e:
        print("Research Loss/F1/CWA plot error:", e)

    # Figure 4: Research Confusion Matrix and Weight Distribution (2 subplots)
    try:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        # Confusion Matrix.
        preds = np.array(logs["predictions"])
        gts = np.array(logs["ground_truth"])
        num_cls = int(max(preds.max(), gts.max()) + 1)
        cm = np.zeros((num_cls, num_cls), dtype=int)
        for p, t in zip(preds, gts):
            cm[t, p] += 1
        ax = axs[0]
        im = ax.imshow(cm, cmap="Blues")
        for i in range(num_cls):
            for j in range(num_cls):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Research: Confusion Matrix")
        plt.colorbar(im, ax=ax)
        remove_spines(ax)
        
        # Weight Histogram (if available).
        ax = axs[1]
        if "weights" in logs and np.array(logs["weights"]).size:
            weights = np.array(logs["weights"])
            ax.hist(weights, bins=min(30, len(np.unique(weights))), color="tab:purple")
            ax.set_xlabel("Example Weight")
            ax.set_ylabel("Count")
            ax.set_title("Research: Weight Distribution")
        else:
            ax.text(0.5, 0.5, "No weights available", ha="center", va="center")
        remove_spines(ax)
        
        fig.tight_layout()
        fig.savefig(os.path.join("figures", "Research_confusion_weight.png"))
        plt.close(fig)
    except Exception as e:
        print("Research Confusion/Weight plot error:", e)

############################
# Ablation Plotting Method #
############################
def plot_ablation():
    # Ablation 1: No Curriculum Complexity Weighting
    try:
        file_nc = "experiment_results/experiment_7522ce3363a74682b67e5e13d5753ce6_proc_3448830/experiment_data.npy"
        data = np.load(file_nc, allow_pickle=True).item()
        # Structure: data["no_curriculum_weighting"]["SPR_BENCH"]
        exp = data["no_curriculum_weighting"]["SPR_BENCH"]
        epochs = np.arange(1, len(exp["losses"]["train"]) + 1)
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        # Loss curves.
        ax = axs[0]
        ax.plot(epochs, exp["losses"]["train"], label="Train", color="tab:blue")
        ax.plot(epochs, exp["losses"]["val"], label="Val", color="tab:orange", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Ablation (No Curriculum): Loss Curves")
        ax.legend()
        remove_spines(ax)
        
        # Macro-F1 curve.
        ax = axs[1]
        macro_f1 = [m["macro_f1"] for m in exp["metrics"]["val"]]
        ax.plot(epochs, macro_f1, marker="o", color="tab:green")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Macro-F1")
        ax.set_title("Ablation (No Curriculum): Macro-F1")
        remove_spines(ax)
        fig.tight_layout()
        fig.savefig(os.path.join("figures", "Ablation_NoCurriculum_loss_macroF1.png"))
        plt.close(fig)
    except Exception as e:
        print("Ablation No Curriculum plot 1 error:", e)
        
    try:
        # Additional Ablation 1: CWA and Confusion Matrix.
        exp = data["no_curriculum_weighting"]["SPR_BENCH"]
        epochs = np.arange(1, len(exp["losses"]["train"]) + 1)
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        # CWA curve.
        ax = axs[0]
        cwa = [m["cwa"] for m in exp["metrics"]["val"]]
        ax.plot(epochs, cwa, marker="s", color="tab:red")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("CWA")
        ax.set_title("Ablation (No Curriculum): CWA")
        remove_spines(ax)
        
        # Confusion matrix.
        preds = np.array(exp["predictions"])
        gts = np.array(exp["ground_truth"])
        cm = np.zeros((2, 2), dtype=int)
        for p, t in zip(preds, gts):
            cm[t, p] += 1
        ax = axs[1]
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Ablation (No Curriculum): Confusion Matrix")
        plt.colorbar(im, ax=ax)
        remove_spines(ax)
        
        fig.tight_layout()
        fig.savefig(os.path.join("figures", "Ablation_NoCurriculum_CWA_Confusion.png"))
        plt.close(fig)
    except Exception as e:
        print("Ablation No Curriculum plot 2 error:", e)
        
    # Ablation 2: Remove Learned Positional Embedding (using key "mean_pooling_no_cls")
    try:
        file_nlpe = "experiment_results/experiment_94bc854fb02044c1ad7c303cb4c163b7_proc_3448832/experiment_data.npy"
        data = np.load(file_nlpe, allow_pickle=True).item()
        exp = data["mean_pooling_no_cls"]["SPR_BENCH"]
        epochs = np.arange(1, len(exp["losses"]["train"]) + 1)
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        # Loss curves.
        ax = axs[0]
        ax.plot(epochs, exp["losses"]["train"], label="Train", color="tab:blue")
        ax.plot(epochs, exp["losses"]["val"], label="Val", color="tab:orange", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Ablation (No PosEmb): Loss Curves")
        ax.legend()
        remove_spines(ax)
        
        # Macro-F1.
        ax = axs[1]
        macro_f1 = [m["macro_f1"] for m in exp["metrics"]["val"]]
        ax.plot(epochs, macro_f1, marker="o", color="tab:green")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Macro-F1")
        ax.set_title("Ablation (No PosEmb): Macro-F1")
        remove_spines(ax)
        
        # CWA.
        ax = axs[2]
        cwa = [m["cwa"] for m in exp["metrics"]["val"]]
        ax.plot(epochs, cwa, marker="s", color="tab:red")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("CWA")
        ax.set_title("Ablation (No PosEmb): CWA")
        remove_spines(ax)
        
        fig.tight_layout()
        fig.savefig(os.path.join("figures", "Ablation_NoPosEmb_metrics.png"))
        plt.close(fig)
    except Exception as e:
        print("Ablation Remove Learned Positional Embedding plot error:", e)
        
    try:
        # Label Distribution for Remove Learned Positional Embedding.
        exp = data["mean_pooling_no_cls"]["SPR_BENCH"]
        preds = np.array(exp["predictions"])
        gts = np.array(exp["ground_truth"])
        labels = sorted(set(np.concatenate([gts, preds])))
        gt_counts = [np.sum(gts == lbl) for lbl in labels]
        pr_counts = [np.sum(preds == lbl) for lbl in labels]
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].bar(labels, gt_counts, color="steelblue")
        axs[0].set_title("Ablation (No PosEmb): Ground Truth")
        axs[0].set_xlabel("Label")
        axs[0].set_ylabel("Count")
        axs[1].bar(labels, pr_counts, color="darkorange")
        axs[1].set_title("Ablation (No PosEmb): Predictions")
        axs[1].set_xlabel("Label")
        fig.suptitle("Ablation (No PosEmb): Label Distribution", fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(os.path.join("figures", "Ablation_NoPosEmb_label_distribution.png"))
        plt.close(fig)
    except Exception as e:
        print("Ablation No PosEmb label distribution error:", e)
        
    # Ablation 3: No Label Smoothing
    try:
        file_nls = "experiment_results/experiment_9b425d3406a64611922e8ff523c148cf_proc_3448833/experiment_data.npy"
        data = np.load(file_nls, allow_pickle=True).item()
        # Loop over experiments; use first one that contains SPR_BENCH.
        for exp_name, datasets in data.items():
            if "SPR_BENCH" in datasets:
                exp = datasets["SPR_BENCH"]
                break
        epochs = np.arange(1, len(exp["losses"]["train"]) + 1)
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        # Loss curves.
        ax = axs[0]
        ax.plot(epochs, exp["losses"]["train"], label="Train", color="tab:blue")
        ax.plot(epochs, exp["losses"]["val"], label="Val", color="tab:orange", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Ablation (No Label Smoothing): Loss Curves")
        ax.legend()
        remove_spines(ax)
        # Macro-F1 curve.
        ax = axs[1]
        macro_f1 = [m["macro_f1"] for m in exp["metrics"]["val"]]
        ax.plot(epochs, macro_f1, marker="o", color="tab:green", label="Macro-F1")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title("Ablation (No Label Smoothing): Macro-F1")
        ax.legend()
        remove_spines(ax)
        # Ground Truth vs Predictions scatter.
        ax = axs[2]
        preds = np.array(exp["predictions"])
        gts = np.array(exp["ground_truth"])
        if len(preds) > 200:
            idx = np.linspace(0, len(preds) - 1, num=200).astype(int)
        else:
            idx = np.arange(len(preds))
        ax.scatter(gts[idx], preds[idx], alpha=0.6, s=10)
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Predictions")
        ax.set_title("Ablation (No Label Smoothing): GT vs Predictions")
        remove_spines(ax)
        
        fig.tight_layout()
        fig.savefig(os.path.join("figures", "Ablation_NoLabelSmoothing.png"))
        plt.close(fig)
    except Exception as e:
        print("Ablation No Label Smoothing plot error:", e)
        
    # Ablation 4: Remove Gradient Clipping
    try:
        file_ngc = "experiment_results/experiment_933a0d27b5eb4c729e7ea159626d4ad9_proc_3448831/experiment_data.npy"
        data = np.load(file_ngc, allow_pickle=True).item()
        # Loop over experiments and use the first found.
        for exp_name, exp_dict in data.items():
            for dset, res in exp_dict.items():
                exp = res
                break
            break
        epochs = np.arange(1, len(exp["losses"]["train"]) + 1)
        fig, axs = plt.subplots(3, 2, figsize=(12, 10))
        axs = axs.flatten()
        # (1) Loss curves.
        ax = axs[0]
        ax.plot(epochs, exp["losses"]["train"], label="Train", color="tab:blue")
        ax.plot(epochs, exp["losses"]["val"], label="Val", color="tab:orange", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("No Grad Clip: Loss Curves")
        ax.legend()
        remove_spines(ax)
        # (2) Metrics: Macro-F1 and CWA.
        ax = axs[1]
        mf1 = [m.get("macro_f1", np.nan) for m in exp["metrics"]["val"]]
        cwa = [m.get("cwa", np.nan) for m in exp["metrics"]["val"]]
        ax.plot(epochs, mf1, label="Macro-F1", color="tab:green")
        ax.plot(epochs, cwa, label="CWA", color="tab:red")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title("No Grad Clip: Metrics")
        ax.legend()
        remove_spines(ax)
        # (3) Confusion Matrix.
        preds = np.array(exp["predictions"])
        gts = np.array(exp["ground_truth"])
        cm = np.zeros((2, 2), dtype=int)
        for p, t in zip(preds, gts):
            cm[t, p] += 1
        ax = axs[2]
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("No Grad Clip: Confusion Matrix")
        plt.colorbar(im, ax=ax)
        remove_spines(ax)
        # (4) Weight Distribution.
        if "weights" in exp and np.array(exp["weights"]).size:
            ax = axs[3]
            ax.hist(exp["weights"], bins=30, color="gray")
            ax.set_xlabel("Weight")
            ax.set_ylabel("Count")
            ax.set_title("No Grad Clip: Weight Distribution")
            remove_spines(ax)
        # (5) Correctness vs Weight scatter.
        if "weights" in exp and np.array(exp["weights"]).size:
            ax = axs[4]
            ws = np.array(exp["weights"])
            correct = (preds == gts).astype(int)
            ax.scatter(ws, correct, alpha=0.3, s=10)
            ax.set_xlabel("Weight")
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Wrong", "Correct"])
            ax.set_title("No Grad Clip: Correctness vs Weight")
            remove_spines(ax)
        fig.tight_layout()
        fig.savefig(os.path.join("figures", "Ablation_NoGradClip.png"))
        plt.close(fig)
    except Exception as e:
        print("Ablation Remove Gradient Clipping plot error:", e)
        
    # Ablation 5: Replace Transformer Encoder with Bi-LSTM Backbone
    try:
        file_bilstm = "experiment_results/experiment_8a9877485fd84ffc9b4c856e4d6f2356_proc_3448833/experiment_data.npy"
        data = np.load(file_bilstm, allow_pickle=True).item()
        exp = data["bi_lstm_backbone"]["SPR_BENCH"]
        epochs = np.arange(1, len(exp["losses"]["train"]) + 1)
        fig, axs = plt.subplots(3, 2, figsize=(12, 10))
        axs = axs.flatten()
        # (1) Loss curves.
        ax = axs[0]
        ax.plot(epochs, exp["losses"]["train"], label="Train", color="tab:blue")
        ax.plot(epochs, exp["losses"]["val"], label="Val", color="tab:orange", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Bi-LSTM: Loss Curves")
        ax.legend()
        remove_spines(ax)
        # (2) Macro-F1 curve.
        ax = axs[1]
        mf1 = [m["macro_f1"] for m in exp["metrics"]["val"]]
        ax.plot(epochs, mf1, marker="o", color="tab:green")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Macro-F1")
        ax.set_title("Bi-LSTM: Macro-F1")
        remove_spines(ax)
        # (3) CWA curve.
        ax = axs[2]
        cwa = [m["cwa"] for m in exp["metrics"]["val"]]
        ax.plot(epochs, cwa, marker="s", color="tab:red")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("CWA")
        ax.set_title("Bi-LSTM: CWA")
        remove_spines(ax)
        # (4) Confusion Matrix.
        preds = np.array(exp["predictions"])
        gts = np.array(exp["ground_truth"])
        cm = np.zeros((2, 2), dtype=int)
        for p, t in zip(preds, gts):
            cm[t, p] += 1
        ax = axs[3]
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="red")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Bi-LSTM: Confusion Matrix")
        plt.colorbar(im, ax=ax)
        remove_spines(ax)
        # (5) Weight Histogram.
        if "weights" in exp and np.array(exp["weights"]).size:
            ax = axs[4]
            ax.hist(exp["weights"], bins=20, color="purple")
            ax.set_xlabel("Weight")
            ax.set_ylabel("Count")
            ax.set_title("Bi-LSTM: Weight Histogram")
            remove_spines(ax)
        fig.tight_layout()
        fig.savefig(os.path.join("figures", "Ablation_BiLSTM_backbone.png"))
        plt.close(fig)
    except Exception as e:
        print("Ablation Bi-LSTM plot error:", e)

###########################
# Main: Create all plots. #
###########################
if __name__ == "__main__":
    plot_baseline()
    plot_research()
    plot_ablation()
    print("All final figures have been saved in the 'figures/' directory.")
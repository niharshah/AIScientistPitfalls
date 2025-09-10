#!/usr/bin/env python3
"""
Aggregator script for final scientific plots for the paper submission.
This script loads experiment results from existing .npy files (using full paths
from the summaries) and produces final, publication‐ready plots stored in the
"figures" directory. Each plotting section is wrapped in a try‐except block so that an
error in one plot does not affect the others.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Increase font size for publication quality
plt.rcParams.update({'font.size': 14})
plt.rcParams['figure.dpi'] = 300

# Create figures directory
os.makedirs("figures", exist_ok=True)


def remove_top_right_spines(ax):
    """Remove top and right spines for a cleaner look."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    

def plot_baseline():
    # Baseline experiment file from summary
    baseline_file = "experiment_results/experiment_21f62ea497054eed8c774237c58ed2da_proc_2945034/experiment_data.npy"
    try:
        baseline_data = np.load(baseline_file, allow_pickle=True).item()
        runs = baseline_data.get("num_epochs", {})
        num_runs = len(runs)
        if num_runs == 0:
            print("No runs found in baseline data.")
            return

        # Create a figure with up to 4 subplots (max 3 per row)
        n_plots = min(num_runs, 4)
        ncols = 3 if n_plots > 1 else 1
        nrows = int(np.ceil(n_plots / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        # For each run, plot train vs validation loss curves
        for ax_i, (run_name, run_data) in enumerate(runs.items()):
            if ax_i >= n_plots:
                break
            try:
                train_losses = run_data["losses"]["train"]
                val_losses = run_data["losses"]["val"]
                epochs = np.arange(1, len(train_losses)+1)
                ax = axes[ax_i]
                ax.plot(epochs, train_losses, label="Train", marker="o")
                ax.plot(epochs, val_losses, label="Validation", marker="o")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Cross-Entropy Loss")
                ax.set_title(f"{run_name}: Loss Curves (SPR_BENCH)")
                ax.legend()
                remove_top_right_spines(ax)
            except Exception as sub_e:
                print(f"Error plotting loss curve for run {run_name}: {sub_e}")
        plt.tight_layout()
        save_path = os.path.join("figures", "baseline_loss_curves.png")
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved Baseline loss curves to {save_path}")
    except Exception as e:
        print(f"Error processing baseline loss curves: {e}")

    # Aggregated test metrics bar chart
    try:
        labels, crwa_vals, swa_vals, cwa_vals = [], [], [], []
        for run_name, run_data in runs.items():
            labels.append(run_name)
            test_metrics = run_data.get("metrics", {}).get("test", {})
            crwa_vals.append(test_metrics.get("CRWA", 0))
            swa_vals.append(test_metrics.get("SWA", 0))
            cwa_vals.append(test_metrics.get("CWA", 0))
        x = np.arange(len(labels))
        w = 0.25
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x - w, crwa_vals, width=w, label="CRWA")
        ax.bar(x, swa_vals, width=w, label="SWA")
        ax.bar(x + w, cwa_vals, width=w, label="CWA")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Baseline: Test Metrics Comparison (SPR_BENCH)")
        ax.legend()
        remove_top_right_spines(ax)
        plt.tight_layout()
        save_path = os.path.join("figures", "baseline_test_metrics.png")
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved Baseline test metrics chart to {save_path}")
    except Exception as e:
        print(f"Error creating baseline aggregated metrics plot: {e}")


def plot_research():
    # Research experiment file from summary
    research_file = "experiment_results/experiment_d4fc4965d09f4d549853d5ad6b6f0f04_proc_2950686/experiment_data.npy"
    try:
        research_data = np.load(research_file, allow_pickle=True).item()
        # Assume the research dictionary is keyed by dataset (e.g., "SPR_BENCH")
        for dataset, ds_data in research_data.items():
            # Get loss curves and SWA from stored validation metrics (assumed list over epochs)
            train_losses = ds_data["losses"]["train"]
            val_losses = ds_data["losses"]["val"]
            epochs = np.arange(1, len(train_losses)+1)
            val_swa = ds_data["metrics"]["val"]
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            # Loss curve subplot
            ax = axes[0]
            ax.plot(epochs, train_losses, label="Train", marker="o")
            ax.plot(epochs, val_losses, label="Validation", marker="o")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Cross-Entropy Loss")
            ax.set_title(f"{dataset}: Loss Curve")
            ax.legend()
            remove_top_right_spines(ax)
            # SWA subplot
            ax = axes[1]
            ax.plot(epochs, val_swa, marker="o", color="green", label="Validation SWA")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Shape-Weighted Accuracy")
            ax.set_title(f"{dataset}: Validation SWA")
            ax.set_ylim(0, 1.05)
            ax.legend()
            remove_top_right_spines(ax)
            plt.tight_layout()
            save_path = os.path.join("figures", f"research_{dataset}_plots.png")
            plt.savefig(save_path)
            plt.close(fig)
            print(f"Saved Research plots for {dataset} to {save_path}")
    except Exception as e:
        print(f"Error processing research plots: {e}")


def plot_ablation_no_symbolic_features():
    # Ablation experiment: No-Symbolic-Features (Pure Transformer Baseline)
    file_path = "experiment_results/experiment_3555ada69eb549efbbd3437db5d8ce86_proc_2952777/experiment_data.npy"
    try:
        data = np.load(file_path, allow_pickle=True).item()
        # Access nested dictionary: "no_symbolic_features" -> "SPR_BENCH"
        exp_data = data.get("no_symbolic_features", {}).get("SPR_BENCH", {})
        epochs = np.array(exp_data.get("epochs", []))
        train_loss = np.array(exp_data.get("losses", {}).get("train", []))
        val_loss = np.array(exp_data.get("losses", {}).get("val", []))
        val_swa = np.array(exp_data.get("metrics", {}).get("val", []))
        # Predictions for confusion matrix (use last epoch)
        y_true = np.array(exp_data.get("ground_truth", [])[-1])
        y_pred = np.array(exp_data.get("predictions", [])[-1])
        # Create one figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        # Left: loss curves
        ax = axes[0]
        ax.plot(epochs, train_loss, label="Train Loss", marker="o")
        ax.plot(epochs, val_loss, label="Val Loss", marker="o")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.set_title("No-Symbolic-Features: Loss Curves")
        ax.legend()
        remove_top_right_spines(ax)
        # Middle: SWA curve
        ax = axes[1]
        ax.plot(epochs, val_swa, marker="o", color="purple")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Shape-Weighted Accuracy")
        ax.set_title("No-Symbolic-Features: Validation SWA")
        ax.set_ylim(0, 1.05)
        remove_top_right_spines(ax)
        # Right: Confusion matrix at final epoch
        n_cls = len(np.unique(np.concatenate([y_true, y_pred])))
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        ax = axes[2]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("No-Symbolic-Features: Confusion Matrix")
        for i in range(n_cls):
            for j in range(n_cls):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.colorbar(im, ax=axes[2])
        plt.tight_layout()
        save_path = os.path.join("figures", "ablation_no_symbolic_features.png")
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved Ablation No-Symbolic-Features plots to {save_path}")
    except Exception as e:
        print(f"Error in No-Symbolic-Features ablation plots: {e}")


def plot_ablation_no_interaction_symbolic():
    # Ablation experiment: No-Interaction-Symbolic (Drop n_shape×n_color Feature)
    file_path = "experiment_results/experiment_07b6756a0ef4491e927374d353dd070f_proc_2952779/experiment_data.npy"
    try:
        data = np.load(file_path, allow_pickle=True).item()
        exp_data = data.get("no_interaction_symbolic", {}).get("SPR_BENCH", {})
        epochs = np.array(exp_data.get("epochs", []))
        tr_loss = np.array(exp_data.get("losses", {}).get("train", []))
        va_loss = np.array(exp_data.get("losses", {}).get("val", []))
        val_swa = np.array(exp_data.get("metrics", {}).get("val", []))
        preds = exp_data.get("predictions", [])
        gts = exp_data.get("ground_truth", [])
        # Determine best epoch index based on lowest val loss
        best_idx = int(np.argmin(va_loss)) if va_loss.size else 0
        y_true = np.array(gts[best_idx])
        y_pred = np.array(preds[best_idx])
        # Create figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        # Left subplot: Loss curves
        ax = axes[0]
        ax.plot(epochs, tr_loss, label="Train Loss", marker="o")
        ax.plot(epochs, va_loss, label="Val Loss", marker="o")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.set_title("No-Interaction-Symbolic: Loss Curves")
        ax.legend()
        remove_top_right_spines(ax)
        # Middle subplot: SWA curve
        ax = axes[1]
        ax.plot(epochs, val_swa, marker="o", color="orange")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Shape-Weighted Accuracy")
        ax.set_title("No-Interaction-Symbolic: Validation SWA")
        ax.set_ylim(0, 1.05)
        remove_top_right_spines(ax)
        # Right subplot: Confusion matrix at best epoch
        n_cls = len(set(y_true) | set(y_pred))
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        ax = axes[2]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"No-Interaction-Symbolic: Confusion Matrix (Epoch {epochs[best_idx]})")
        for i in range(n_cls):
            for j in range(n_cls):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.colorbar(im, ax=axes[2])
        plt.tight_layout()
        save_path = os.path.join("figures", "ablation_no_interaction_symbolic.png")
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved Ablation No-Interaction-Symbolic plots to {save_path}")
    except Exception as e:
        print(f"Error in No-Interaction-Symbolic ablation plots: {e}")


def plot_ablation_bag_of_embeddings():
    # Ablation experiment: Bag-of-Embeddings (Transformer-Encoder Removed)
    file_path = "experiment_results/experiment_33f587c3087248319118a46ff02257f2_proc_2952780/experiment_data.npy"
    try:
        data = np.load(file_path, allow_pickle=True).item()
        exp_data = data.get("bag_of_embeddings", {}).get("SPR_BENCH", {})
        epochs = np.array(exp_data.get("epochs", []))
        train_loss = np.array(exp_data.get("losses", {}).get("train", []))
        val_loss = np.array(exp_data.get("losses", {}).get("val", []))
        val_metric = np.array(exp_data.get("metrics", {}).get("val", []))
        # Create a figure with 2 subplots: loss curves and SWA
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax = axes[0]
        ax.plot(epochs, train_loss, label="Train Loss", marker="o")
        ax.plot(epochs, val_loss, label="Val Loss", marker="o")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.set_title("Bag-of-Embeddings: Loss Curves")
        ax.legend()
        remove_top_right_spines(ax)
        ax = axes[1]
        ax.plot(epochs, val_metric, marker="o", color="teal")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Shape-Weighted Accuracy")
        ax.set_title("Bag-of-Embeddings: Validation SWA")
        ax.set_ylim(0, 1.05)
        remove_top_right_spines(ax)
        plt.tight_layout()
        save_path = os.path.join("figures", "ablation_bag_of_embeddings.png")
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved Ablation Bag-of-Embeddings plots to {save_path}")
    except Exception as e:
        print(f"Error in Bag-of-Embeddings ablation plots: {e}")


def plot_ablation_frozen_random_embeddings():
    # Ablation experiment: Frozen-Random-Embeddings
    file_path = "experiment_results/experiment_1c142afb55ef4e0c9508838c8d55b0bf_proc_2952778/experiment_data.npy"
    try:
        data = np.load(file_path, allow_pickle=True).item()
        exp_data = data.get("Frozen-Random-Embeddings", {}).get("SPR_BENCH", {})
        epochs = np.array(exp_data.get("epochs", []))
        train_losses = np.array(exp_data.get("losses", {}).get("train", []))
        val_losses = np.array(exp_data.get("losses", {}).get("val", []))
        val_swa = np.array(exp_data.get("metrics", {}).get("val", []))
        preds_all = exp_data.get("predictions", [])
        gts_all = exp_data.get("ground_truth", [])
        best_idx = int(np.argmin(val_losses)) if len(val_losses) else 0
        y_true = np.array(gts_all[best_idx])
        y_pred = np.array(preds_all[best_idx])
        # Create a figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        # Loss curves
        ax = axes[0]
        ax.plot(epochs, train_losses, label="Train Loss", marker="o")
        ax.plot(epochs, val_losses, label="Val Loss", marker="o")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.set_title("Frozen-Random-Embeddings: Loss Curves")
        ax.legend()
        remove_top_right_spines(ax)
        # SWA curve
        ax = axes[1]
        ax.plot(epochs, val_swa, marker="o", color="brown")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Shape-Weighted Accuracy")
        ax.set_title("Frozen-Random-Embeddings: Validation SWA")
        ax.set_ylim(0, 1.05)
        remove_top_right_spines(ax)
        # Confusion matrix for best epoch
        n_cls = len(np.unique(np.concatenate([y_true, y_pred])))
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        ax = axes[2]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Frozen-Random-Embeddings: Confusion Matrix\n(Epoch {epochs[best_idx]})")
        for i in range(n_cls):
            for j in range(n_cls):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.colorbar(im, ax=axes[2])
        plt.tight_layout()
        save_path = os.path.join("figures", "ablation_frozen_random_embeddings.png")
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved Ablation Frozen-Random-Embeddings plots to {save_path}")
    except Exception as e:
        print(f"Error in Frozen-Random-Embeddings ablation plots: {e}")


def main():
    print("Generating final publication plots...")
    plot_baseline()
    plot_research()
    plot_ablation_no_symbolic_features()
    plot_ablation_no_interaction_symbolic()
    plot_ablation_bag_of_embeddings()
    plot_ablation_frozen_random_embeddings()
    print("All plots generated and saved in the 'figures/' directory.")


if __name__ == "__main__":
    main()
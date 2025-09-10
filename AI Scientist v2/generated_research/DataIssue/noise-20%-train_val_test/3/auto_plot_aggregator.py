#!/usr/bin/env python3
"""
Final Aggregator Script for SPR Final Figures

This script loads experiment data from .npy files saved during the baseline,
research, and ablation experiments, and produces final publication‐quality
aggregated figures. For each experiment, three related plots are combined into
a single aggregated figure with three subplots:
  1) Loss Curves (Train vs Validation Loss)
  2) Validation Macro F1 Curve
  3) Confusion Matrix

All figures are saved in the "figures/" directory. No extra or duplicate plots
are produced. All titles, legends, and labels use descriptive names without
underscores. Each aggregated figure is produced in a single try-except block so
that a failure in one experiment does not affect the others.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score

# Global style settings for publication-quality final plots.
plt.rcParams.update({'font.size': 14})
os.makedirs("figures", exist_ok=True)

def safe_plot(ax, x, y, label, plot_type="line", marker=None):
    """Helper function to plot a line if data lengths are consistent, otherwise annotate."""
    if len(x) != len(y) or len(x) == 0:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        ax.set_title(label)
    else:
        if plot_type == "line":
            ax.plot(x, y, marker=marker if marker else "")
        else:
            ax.plot(x, y)
        ax.set_title(label)

def aggregate_experiment(exp_name, file_path, key_path=None):
    """
    Load experiment data from file_path and select the sub-dictionary if key_path provided.
    Plot an aggregated figure (one per experiment) with the following subplots:
      1) Loss Curves (Train vs Validation Loss)
      2) Validation Macro F1 Curve
      3) Confusion Matrix
    All axis labels and titles do not use underscores and are descriptive.
    """
    try:
        data = np.load(file_path, allow_pickle=True).item()
    except Exception as e:
        print(f"{exp_name}: Error loading data: {e}")
        return

    # For ablation experiments use key_path to select the correct entry.
    if key_path:
        for key in key_path:
            data = data.get(key, {})
        ed = data
    else:
        # For baseline: data is nested under "epochs_tuning" -> "SPR_BENCH"
        if exp_name.lower() == "baseline":
            ed = data.get("epochs_tuning", {}).get("SPR_BENCH", {})
        else:
            # For research, assume single dataset entry under key "SPR_BENCH" if present,
            # otherwise use the first entry.
            ed = data.get("SPR_BENCH", {})
            if not ed and isinstance(data, dict) and data:
                ed = list(data.values())[0]

    # Extract epoch and metric arrays.
    epochs = ed.get("epochs", [])
    preds = np.array(ed.get("predictions", []))
    gts = np.array(ed.get("ground_truth", []))

    # For baseline and research the losses and metrics are under "metrics".
    if exp_name.lower() in ["baseline", "research"]:
        metrics = ed.get("metrics", {})
        train_loss = metrics.get("train_loss", [])
        val_loss = metrics.get("val_loss", [])
        val_f1 = metrics.get("val_f1", [])
    else:
        # For ablation experiments the losses are under "losses" and metrics under "metrics"
        losses = ed.get("losses", {})
        train_loss = losses.get("train", [])
        val_loss = losses.get("val", [])
        metrics = ed.get("metrics", {})
        val_f1 = metrics.get("val", [])

    # Create aggregated figure with 3 subplots in one row.
    try:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

        # Subplot 1: Loss Curves
        axs[0].plot(epochs, train_loss, label="Train Loss")
        axs[0].plot(epochs, val_loss, label="Validation Loss")
        axs[0].set_title("Loss Curves")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].legend()
        axs[0].spines["top"].set_visible(False)
        axs[0].spines["right"].set_visible(False)

        # Subplot 2: Validation Macro F1 Curve
        safe_plot(axs[1], epochs, val_f1, "Validation Macro F1", marker="o")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Macro F1")
        axs[1].legend(["Validation Macro F1"])  # Basic legend

        axs[1].spines["top"].set_visible(False)
        axs[1].spines["right"].set_visible(False)

        # Subplot 3: Confusion Matrix
        if preds.size and gts.size:
            cm = confusion_matrix(gts, preds)
            im = axs[2].imshow(cm, interpolation="nearest", cmap="Blues")
            axs[2].set_title("Confusion Matrix")
            axs[2].set_xlabel("Predicted")
            axs[2].set_ylabel("True")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    axs[2].text(j, i, cm[i, j], ha="center", va="center", color="black")
            fig.colorbar(im, ax=axs[2])
        else:
            axs[2].text(0.5, 0.5, "No data", ha="center", va="center")
            axs[2].set_title("Confusion Matrix")
            axs[2].set_xlabel("Predicted")
            axs[2].set_ylabel("True")

        fig.suptitle(f"{exp_name.title()} Experiment - SPR BENCH", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join("figures", f"{exp_name.lower()}_aggregated.png")
        fig.savefig(save_path)
        plt.close(fig)
    except Exception as e:
        print(f"{exp_name}: Error generating aggregated figure: {e}")

    # Compute and print final Macro F1 if predictions are available.
    try:
        if preds.size and gts.size:
            final_f1 = f1_score(gts, preds, average="macro")
            print(f"{exp_name.title()} Final Test Macro F1: {final_f1:.4f}")
            return final_f1
    except Exception as e:
        print(f"{exp_name}: Error computing final F1: {e}")
    return None

def main():
    # Aggregated figures: only one final plot per experiment.
    # Baseline aggregated figure.
    baseline_file = ("experiment_results/experiment_1eeee893d4a04eb19b409b6b8319afca_proc_3158132/"
                     "experiment_data.npy")
    aggregate_experiment("baseline", baseline_file)

    # Research aggregated figure.
    research_file = ("experiment_results/experiment_f78078eecf3d4f7ba02088c528bfb85e_proc_3162419/"
                     "experiment_data.npy")
    aggregate_experiment("research", research_file)

    # Ablation: No Label Smoothing aggregated figure.
    ablation_ls_file = ("experiment_results/experiment_7497bc328b84462b993f6130609a670c_proc_3168672/"
                        "experiment_data.npy")
    aggregate_experiment("ablation no label smoothing", ablation_ls_file, key_path=["No_Label_Smoothing", "SPR_BENCH"])

    # Ablation: No Dropout aggregated figure.
    ablation_nd_file = ("experiment_results/experiment_0a6b083dfec54895ba3787d16f671c7d_proc_3168673/"
                        "experiment_data.npy")
    aggregate_experiment("ablation no dropout", ablation_nd_file, key_path=["no_dropout", "SPR_BENCH"])

    print("Figure generation complete. Aggregated final plots saved in 'figures/'.")

if __name__ == "__main__":
    main()
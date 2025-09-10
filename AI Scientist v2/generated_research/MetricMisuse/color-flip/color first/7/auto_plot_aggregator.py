#!/usr/bin/env python3
"""
Final Aggregator Script
This script loads precomputed experiment results (stored as .npy files) from various experiments:
  • Baseline
  • Research (with glyph clustering preprocessing)
  • Several Ablation studies:
      – Random Cluster Assignment
      – Cluster-Only Representation
      – Bag-of-Embeddings (No-RNN)
      – Shape-Based Clustering (Rule-Based Groups)
      – No-AE Clustering (Raw One-Hot K-Means)

For each experiment, we generate publication‐ready figures (saved in the “figures/” directory)
that illustrate key outcomes:
  • Training Loss curves
  • Complexity‐Weighted Accuracy (Cpx WA) tracks (train vs. validation)
  • Weighted Accuracy comparisons (Color-Weighted (CWA), Shape-Weighted (SWA))
  • Aggregated multi‐panel figures for ablation studies, and selected confusion matrices.

Each plotting routine is wrapped in its own try/except block so that failures in one do not affect the others.
All figures are saved with dpi=300 and employ larger fonts for publication readability.
No synthetic data is introduced – all data is loaded from the provided .npy files.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import itertools

# Use a larger font size for publication clarity
plt.rcParams.update({'font.size': 14, 'axes.spines.top': False, 'axes.spines.right': False})

def plot_baseline():
    # Baseline Experiment: File from baseline summary
    try:
        path = "experiment_results/experiment_58776df0d3084187852c5ee56edd0b03_proc_1730837/experiment_data.npy"
        data = np.load(path, allow_pickle=True).item()
        epochs = np.array(data.get("epochs", []))
        train_losses = np.array(data.get("losses", {}).get("train", []))
        train_metrics = data.get("metrics", {}).get("train", [])
        val_metrics = data.get("metrics", {}).get("val", [])
        # Extract validation weighted accuracies if available
        if val_metrics:
            val_cwa = np.array([m.get("cwa", np.nan) for m in val_metrics])
            val_swa = np.array([m.get("swa", np.nan) for m in val_metrics])
            val_cpx = np.array([m.get("cpx", np.nan) for m in val_metrics])
        else:
            val_cwa = val_swa = val_cpx = np.array([])
        if train_metrics:
            train_cpx = np.array([m.get("cpx", np.nan) for m in train_metrics])
        else:
            train_cpx = np.array([])

        # Plot 1: Training Loss Curve
        try:
            plt.figure()
            plt.plot(epochs, train_losses, marker="o", label="Train Loss")
            plt.title("SPR BENCH: Training Loss per Epoch", fontsize=16)
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig("figures/Baseline_Train_Loss.png", dpi=300)
            plt.close()
        except Exception as e:
            print("Error plotting Baseline Training Loss:", e)

        # Plot 2: Train vs. Validation Complexity-Weighted Accuracy
        try:
            plt.figure()
            plt.plot(epochs, train_cpx, marker="o", label="Train Cpx WA")
            plt.plot(epochs, val_cpx, marker="s", label="Val Cpx WA")
            plt.title("SPR BENCH: Complexity-Weighted Accuracy", fontsize=16)
            plt.xlabel("Epoch")
            plt.ylabel("Cpx WA")
            plt.legend()
            plt.tight_layout()
            plt.savefig("figures/Baseline_CpxWA_Train_Val.png", dpi=300)
            plt.close()
        except Exception as e:
            print("Error plotting Baseline Cpx WA:", e)

        # Plot 3: Validation Weighted Accuracy Comparison (CWA, SWA, CpxWA)
        try:
            plt.figure()
            plt.plot(epochs, val_cwa, marker="o", label="Val CWA")
            plt.plot(epochs, val_swa, marker="^", label="Val SWA")
            plt.plot(epochs, val_cpx, marker="s", label="Val Cpx WA")
            plt.title("SPR BENCH: Weighted Accuracy Comparison (Validation)", fontsize=16)
            plt.xlabel("Epoch")
            plt.ylabel("Weighted Accuracy")
            plt.legend()
            plt.tight_layout()
            plt.savefig("figures/Baseline_Val_Weighted_Accuracy.png", dpi=300)
            plt.close()
        except Exception as e:
            print("Error plotting Baseline Weighted Accuracy Comparison:", e)
    except Exception as e:
        print("Error in Baseline plotting:", e)

def plot_research():
    # Research Experiment: File from research summary
    try:
        path = "experiment_results/experiment_7b59a1d4a01046c38a17d5b3f7e1911d_proc_1743796/experiment_data.npy"
        data = np.load(path, allow_pickle=True).item()
        epochs = np.array(data.get("epochs", []))
        train_losses = np.array(data.get("losses", {}).get("train", []))
        train_metrics = data.get("metrics", {}).get("train", [])
        val_metrics = data.get("metrics", {}).get("val", [])
        # Prepare metrics
        train_cpx = np.array([m.get("cpx", np.nan) for m in train_metrics]) if train_metrics else np.array([])
        if val_metrics:
            val_cpx = np.array([m.get("cpx", np.nan) for m in val_metrics])
            val_cwa = np.array([m.get("cwa", np.nan) for m in val_metrics])
            val_swa = np.array([m.get("swa", np.nan) for m in val_metrics])
        else:
            val_cpx = val_cwa = val_swa = np.array([])

        # Aggregate three plots in one figure (1 row, 3 columns)
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        # Subplot 1: Training Loss
        axs[0].plot(epochs, train_losses, marker="o", label="Train Loss")
        axs[0].set_title("Research: Training Loss", fontsize=16)
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].legend()
        # Subplot 2: Complexity-Weighted Accuracy
        axs[1].plot(epochs, train_cpx, marker="o", label="Train Cpx WA")
        axs[1].plot(epochs, val_cpx, marker="s", label="Val Cpx WA")
        axs[1].set_title("Research: Complexity-Weighted Accuracy", fontsize=16)
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Cpx WA")
        axs[1].legend()
        # Subplot 3: Weighted Accuracy Comparison (CWA, SWA, CpxWA)
        axs[2].plot(epochs, val_cwa, marker="o", label="Val CWA")
        axs[2].plot(epochs, val_swa, marker="^", label="Val SWA")
        axs[2].plot(epochs, val_cpx, marker="s", label="Val Cpx WA")
        axs[2].set_title("Research: Weighted Accuracy Comparison", fontsize=16)
        axs[2].set_xlabel("Epoch")
        axs[2].set_ylabel("Weighted Accuracy")
        axs[2].legend()
        plt.tight_layout()
        plt.savefig("figures/Research_Aggregated.png", dpi=300)
        plt.close()
    except Exception as e:
        print("Error in Research plotting:", e)

def plot_ablation_aggregated(file_path, exp_name):
    """
    For an ablation experiment outlined by file_path (a .npy file),
    create an aggregated 2x2 figure with:
      - Training Loss
      - Complexity-Weighted Accuracy (Train vs. Val)
      - Color-Weighted Accuracy (Train vs. Val)
      - Shape-Weighted Accuracy (Train vs. Val)
    The figure is saved in figures/ with an exp_name in the file name.
    """
    try:
        data = np.load(file_path, allow_pickle=True).item()
        epochs = np.array(data.get("epochs", []))
        losses_dict = data.get("losses", {})
        train_losses = np.array(losses_dict.get("train", []))
        metrics = data.get("metrics", {})
        train_metrics = metrics.get("train", [])
        val_metrics = metrics.get("val", [])
        
        def get_metric(m_list, key):
            return np.array([m.get(key, np.nan) for m in m_list]) if m_list else np.array([])

        train_cpx = get_metric(train_metrics, "cpx")
        val_cpx   = get_metric(val_metrics, "cpx")
        train_cwa = get_metric(train_metrics, "cwa")
        val_cwa   = get_metric(val_metrics, "cwa")
        train_swa = get_metric(train_metrics, "swa")
        val_swa   = get_metric(val_metrics, "swa")

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        # Top-left: Training Loss
        axs[0, 0].plot(epochs, train_losses, marker="o", label="Train Loss")
        axs[0, 0].set_title(f"{exp_name}: Training Loss", fontsize=16)
        axs[0, 0].set_xlabel("Epoch")
        axs[0, 0].set_ylabel("Loss")
        axs[0, 0].legend()
        # Top-right: Complexity-Weighted Accuracy
        axs[0, 1].plot(epochs, train_cpx, marker="o", label="Train Cpx WA")
        axs[0, 1].plot(epochs, val_cpx, marker="s", label="Val Cpx WA")
        axs[0, 1].set_title(f"{exp_name}: Complexity-Weighted Accuracy", fontsize=16)
        axs[0, 1].set_xlabel("Epoch")
        axs[0, 1].set_ylabel("Cpx WA")
        axs[0, 1].legend()
        # Bottom-left: Color-Weighted Accuracy
        axs[1, 0].plot(epochs, train_cwa, marker="o", label="Train CWA")
        axs[1, 0].plot(epochs, val_cwa, marker="s", label="Val CWA")
        axs[1, 0].set_title(f"{exp_name}: Color-Weighted Accuracy", fontsize=16)
        axs[1, 0].set_xlabel("Epoch")
        axs[1, 0].set_ylabel("CWA")
        axs[1, 0].legend()
        # Bottom-right: Shape-Weighted Accuracy
        axs[1, 1].plot(epochs, train_swa, marker="o", label="Train SWA")
        axs[1, 1].plot(epochs, val_swa, marker="s", label="Val SWA")
        axs[1, 1].set_title(f"{exp_name}: Shape-Weighted Accuracy", fontsize=16)
        axs[1, 1].set_xlabel("Epoch")
        axs[1, 1].set_ylabel("SWA")
        axs[1, 1].legend()
        plt.tight_layout()
        plt.savefig(f"figures/{exp_name}_Aggregated.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error in aggregated plot for {exp_name}:", e)

def plot_ablation_confusion(file_path, exp_name):
    """
    For the given ablation experiment (file_path), if predictions and ground-truth arrays are present,
    produce a confusion matrix plot.
    """
    try:
        data = np.load(file_path, allow_pickle=True).item()
        preds = np.array(data.get("predictions", []))
        gts = np.array(data.get("ground_truth", []))
        if preds.size and gts.size:
            labels = sorted(list(set(gts) | set(preds)))
            cm = np.zeros((len(labels), len(labels)), dtype=int)
            for t, p in zip(gts, preds):
                i = labels.index(t)
                j = labels.index(p)
                cm[i, j] += 1
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            plt.title(f"{exp_name}: Confusion Matrix", fontsize=16)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.colorbar()
            # Annotate each cell in the confusion matrix
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12)
            plt.tight_layout()
            plt.savefig(f"figures/{exp_name}_Confusion_Matrix.png", dpi=300)
            plt.close()
    except Exception as e:
        print(f"Error in confusion matrix plot for {exp_name}:", e)

def main():
    os.makedirs("figures", exist_ok=True)

    # Generate Baseline Figures
    plot_baseline()

    # Generate Research Figures (aggregated in one 1x3 figure)
    plot_research()

    # Ablation experiments: file paths and experiment names (use underscored names for filenames)
    ablation_experiments = {
        "Random_Cluster_Assignment": "experiment_results/experiment_f7c97ea16ed244f88b7cb40e654cd258_proc_1764218/experiment_data.npy",
        "Cluster_Only_Representation": "experiment_results/experiment_c99eec957c2f47fda470482595864055_proc_1764219/experiment_data.npy",
        "Bag_of_Embeddings_No_RNN": "experiment_results/experiment_9fa14b7c62e94430b7c3c15da251e97e_proc_1764220/experiment_data.npy",
        "Shape_Based_Clustering": "experiment_results/experiment_33e7f87e67ae4f73a143a6902712cc6f_proc_1764217/experiment_data.npy",
        "No_AE_Clustering": "experiment_results/experiment_eab794439a5a46f0a5e096438600d0c1_proc_1764220/experiment_data.npy"
    }
    # For each ablation, produce an aggregated panel of metrics.
    for exp_name, file_path in ablation_experiments.items():
        plot_ablation_aggregated(file_path, exp_name)

    # For select ablation experiments, also produce confusion matrix plots (3 chosen here)
    for exp_name in ["Random_Cluster_Assignment", "Bag_of_Embeddings_No_RNN", "No_AE_Clustering"]:
        file_path = ablation_experiments[exp_name]
        plot_ablation_confusion(file_path, exp_name)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Final Aggregator Script for Symbolic Glyph Clustering Final Figures

This script loads experiment .npy files from the baseline, research and ablation studies,
generates scientifically informative and unique plots directly from the stored data,
and saves the final figures into the "figures/" folder.
Each figure is saved in its own try-except block.
All plot texts use descriptive names without underscores and have increased font sizes.
The total number of figures is less than or equal to 12.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Update matplotlib parameters for publication quality
plt.rcParams.update({
    'font.size': 14,
    'axes.spines.top': False,
    'axes.spines.right': False
})

# Create figures folder
os.makedirs("figures", exist_ok=True)

##########################################
# Helper Functions
##########################################
def save_and_close(fig, fname):
    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print("Saved:", fname)

def plot_bar(ax, metrics, keys, title):
    # keys should be a list such as ["acc", "cwa", "swa", "ccwa"]
    values = [metrics.get(k, np.nan) for k in keys]
    ax.bar([k.upper() for k in keys], values, color="skyblue")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    for i, v in enumerate(values):
        if not np.isnan(v):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center")

##########################################
# Baseline Figures (3 Figures)
##########################################
try:
    # Load baseline experiment data from .npy (ngram tuning)
    baseline_path = "experiment_results/experiment_f38befd1f3664b459f80fd03120fb8e1_proc_1634126/experiment_data.npy"
    baseline_data = np.load(baseline_path, allow_pickle=True).item()
    base_exp = baseline_data["ngram_range_tuning"]["SPR_BENCH"]
    runs = base_exp.get("runs", [])
    best_ngram = base_exp.get("best_ngram", "N/A")
    
    # Figure 1: Baseline Loss Curves (Train vs Val across n-gram settings)
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in runs:
        ngram = r.get("ngram", "NA")
        epochs = np.arange(1, len(r["losses"]["train"]) + 1)
        ax.plot(epochs, r["losses"]["train"], label=f"{ngram} train")
        ax.plot(epochs, r["losses"]["val"], linestyle="--", label=f"{ngram} val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross Entropy Loss")
    ax.set_title("Baseline Loss Curves Across n-gram Settings")
    ax.legend(title="n-gram Range")
    save_and_close(fig, os.path.join("figures", "Baseline Loss Curves.png"))
except Exception as e:
    print("Error in Baseline Loss Curves plot:", e)

try:
    # Figure 2: Baseline Validation Accuracy Curves
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in runs:
        ngram = r.get("ngram", "NA")
        accs = [m.get("acc") for m in r["metrics"]["val"]]
        epochs = np.arange(1, len(accs) + 1)
        ax.plot(epochs, accs, label=f"{ngram}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Baseline Validation Accuracy per n-gram")
    ax.legend(title="n-gram Range")
    save_and_close(fig, os.path.join("figures", "Baseline Validation Accuracy.png"))
except Exception as e:
    print("Error in Baseline Validation Accuracy plot:", e)

try:
    # Figure 3: Baseline Test Metrics Bar Chart
    test_metrics = base_exp.get("metrics", {}).get("test", {})
    fig, ax = plt.subplots(figsize=(6, 4))
    keys = ["acc", "cwa", "swa", "compwa"]
    plot_bar(ax, test_metrics, keys, f"Baseline Test Metrics (Best n-gram {best_ngram})")
    ax.set_ylabel("Score")
    save_and_close(fig, os.path.join("figures", "Baseline Test Metrics.png"))
except Exception as e:
    print("Error in Baseline Test Metrics plot:", e)

##########################################
# Research Figures (1 Figure)
##########################################
try:
    # Load research experiment data
    research_path = "experiment_results/experiment_7c8c9cc926b14c96968d28e479b6cc03_proc_1653756/experiment_data.npy"
    research_data = np.load(research_path, allow_pickle=True).item()
    test_accs = {}
    for ds_key, ds_data in research_data.items():
        test_metrics = ds_data.get("metrics", {}).get("test", {})
        test_accs[ds_key] = test_metrics.get("acc", np.nan)
    # Figure 4: Research Dataset Test Accuracy Comparison Across Datasets
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(test_accs.keys())
    vals = [test_accs[n] for n in names]
    ax.bar(names, vals, color="lightgreen")
    ax.set_ylim(0, 1)
    ax.set_title("Research Test Accuracy Comparison Across Datasets")
    for i, v in enumerate(vals):
        if not np.isnan(v):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    ax.set_ylabel("Accuracy")
    save_and_close(fig, os.path.join("figures", "Research Dataset Comparison.png"))
except Exception as e:
    print("Error in Research dataset comparison plot:", e)

##########################################
# Ablation Studies Aggregated (1 Figure)
##########################################
# We aggregate the test metrics bar charts from all ablation experiments into a single figure.
# This avoids duplication and limits the total number of figures.
ablation_experiments = {
    "Remove Cluster Feature": ("RemoveClusterFeat", "experiment_results/experiment_4cdddbe6194f4bc4a6f325cb11da9fb4_proc_1691878/experiment_data.npy"),
    "Random Cluster Assignment": ("RandomClusterAssignment", "experiment_results/experiment_092512c7cbf549629232fe4003797d13_proc_1691879/experiment_data.npy"),
    "Remove Token Feature": ("remove_token_feature", "experiment_results/experiment_fdb79f2c205a4224b439cb5c4df5067c_proc_1691881/experiment_data.npy"),
    "ORD Embedding Cluster": ("ORD_EMB_CLUSTER_ABLATION", "experiment_results/experiment_b301fa4fca314b3d874b295d1dd1f751_proc_1691880/experiment_data.npy"),
    "No Bigram Feature": ("NoBigramFeatureAblation", "experiment_results/experiment_fb7c7069c061416fa2dd060dad24e11e_proc_1691879/experiment_data.npy"),
    "Token Order Shuffle": ("TokenOrderShuffle", "experiment_results/experiment_b8926c4f7ec3486680728520c5787b70_proc_1691878/experiment_data.npy"),
    "Two Cluster Granularity": ("TwoClusterGranularity", "experiment_results/experiment_9eab43b2d7b34868a2ad4b8d6ae3d839_proc_1691881/experiment_data.npy"),
    "Binary Count Feature": ("BinaryCountAblation", "experiment_results/experiment_479284fee8714e2a94499310cdd4833b_proc_1691880/experiment_data.npy")
}

try:
    # Create one aggregated figure with subplots (2 rows x 4 columns)
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs = axs.flatten()
    keys = ["acc", "cwa", "swa", "ccwa"]
    for i, (exp_label, (exp_key, file_path)) in enumerate(ablation_experiments.items()):
        try:
            ablation_data = np.load(file_path, allow_pickle=True).item()
            exp_dict = ablation_data.get(exp_key, {}).get("SPR_BENCH", {})
            test_metrics = exp_dict.get("metrics", {}).get("test", {})
            title = f"{exp_label}: Test Metrics"
            plot_bar(axs[i], test_metrics, keys, title)
            axs[i].set_ylabel("Score")
        except Exception as inner_e:
            print(f"Error in ablation {exp_label}:", inner_e)
            axs[i].axis('off')
    # Remove any unused subplots if total experiments < subplots
    for j in range(i+1, len(axs)):
        axs[j].axis('off')
    fig.suptitle("Ablation Studies: Aggregated Test Metrics", fontsize=16)
    save_and_close(fig, os.path.join("figures", "Ablation_Aggregated_Test_Metrics.png"))
except Exception as e:
    print("Error creating aggregated ablation test metrics plot:", e)

print("All plots generated and saved in the 'figures/' directory.")
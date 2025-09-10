#!/usr/bin/env python3
"""
Final Aggregator Script for Scientific Paper Figures

This script loads final experiment results (stored as .npy files) from baseline, research, 
and ablation runs. It creates a comprehensive set of final, publishable plots (saved in "figures/")
and a summary comparison plot (in Figures/Aggregate_Abations_Comparison.png).

Each plot is wrapped in its own try/except block so that an error in one does not stop
the entire aggregation.

All plots use increased font sizes and professional styling.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, matthews_corrcoef, f1_score

# Set a larger font size for publication quality
plt.rcParams.update({'font.size': 14})

# Create the output directory for final figures
os.makedirs("figures", exist_ok=True)

#########################################
# Helper functions
#########################################
def save_and_close(fig, fname):
    try:
        fig.tight_layout()
        fig.savefig(fname, dpi=300)
        plt.close(fig)
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error saving figure {fname}: {e}")
        plt.close(fig)

def plot_conf_matrix(ax, y_true, y_pred, title):
    try:
        cm = confusion_matrix(y_true, y_pred)
        im = ax.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")
        ax.set_title(title)
        # Add a colorbar on the side
        plt.colorbar(im, ax=ax)
    except Exception as e:
        print(f"Error in plotting confusion matrix: {e}")

#########################################
# 1. Baseline Aggregated Plot (from Lightweight RNN sweep)
#########################################
try:
    # Load baseline experiment data (baseline best node weight-decay sweep)
    baseline_file = "experiment_results/experiment_4ae98398efcc4757abe9e72139c5bba9_proc_3330952/experiment_data.npy"
    baseline_data = np.load(baseline_file, allow_pickle=True).item()
    
    # The baseline data is stored under a key like "weight_decay"
    # Sort the weight decay keys numerically (as strings)
    wd_dict = baseline_data["weight_decay"]
    wds = sorted(wd_dict.keys(), key=lambda x: float(x))
    
    # We will aggregate two plots in one figure: left: Loss Curves, right: Macro-F1 curves.
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Loss curves: for each weight decay, plot training and validation loss curves.
    for wd in wds:
        epochs = wd_dict[wd]["epochs"]
        train_losses = wd_dict[wd]["losses"]["train"]
        val_losses   = wd_dict[wd]["losses"]["val"]
        axs[0].plot(epochs, train_losses, label=f"Train wd={wd}")
        axs[0].plot(epochs, val_losses, linestyle="--", label=f"Val wd={wd}")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Baseline: Train vs Validation Loss")
    axs[0].legend()
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    
    # Plot F1 curves: for each weight decay, plot training and validation macro-F1 curves.
    for wd in wds:
        epochs = wd_dict[wd]["epochs"]
        train_f1 = wd_dict[wd]["metrics"]["train"]
        val_f1   = wd_dict[wd]["metrics"]["val"]
        axs[1].plot(epochs, train_f1, label=f"Train wd={wd}")
        axs[1].plot(epochs, val_f1, linestyle="--", label=f"Val wd={wd}")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Macro-F1")
    axs[1].set_title("Baseline: Train vs Validation Macro-F1")
    axs[1].legend()
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    
    save_and_close(fig, os.path.join("figures", "Baseline_Aggregated.png"))
except Exception as e:
    print(f"Error creating Baseline Aggregated plot: {e}")

#########################################
# 2. Research Aggregated Plot (from hybrid architecture experiments)
#########################################
try:
    research_file = "experiment_results/experiment_909fe2658add469dbb8e91530d25d9e8_proc_3335814/experiment_data.npy"
    research_data = np.load(research_file, allow_pickle=True).item()
    # research summary: assume one key, e.g. "SPR_BENCH"
    research_key = list(research_data.keys())[0]
    rd = research_data[research_key]
    
    epochs = list(range(1, len(rd.get("losses", {}).get("train", [])) + 1))
    train_loss = rd.get("losses", {}).get("train", [])
    val_loss   = rd.get("losses", {}).get("val", [])
    
    # Assume metrics are stored under keys e.g. "train_MCC" and "val_MCC" if available, else fallback to "metrics"
    train_metric = rd.get("metrics", {}).get("train_MCC", rd.get("metrics", {}).get("train", []))
    val_metric   = rd.get("metrics", {}).get("val_MCC", rd.get("metrics", {}).get("val", []))
    
    # Generate aggregated figure with two subplots: Loss and MCC curves.
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(epochs, train_loss, label="Train Loss")
    axs[0].plot(epochs, val_loss, linestyle="--", label="Val Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Research: Train vs Validation Loss")
    axs[0].legend()
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    
    axs[1].plot(epochs, train_metric, label="Train MCC")
    axs[1].plot(epochs, val_metric, linestyle="--", label="Val MCC")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("MCC")
    axs[1].set_title("Research: Train vs Validation MCC")
    axs[1].legend()
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    
    save_and_close(fig, os.path.join("figures", "Research_Aggregated.png"))
except Exception as e:
    print(f"Error creating Research Aggregated plot: {e}")

#########################################
# 3. Function to Plot Ablation Experiments (three subplots: Loss, MCC, Confusion Matrix)
#########################################
def plot_ablation(ablation_name, npy_path, out_filename):
    try:
        data = np.load(npy_path, allow_pickle=True).item()
        # Ablation experiments are stored under key [ablation_name]["SPR_BENCH"]
        exp = data[ablation_name]["SPR_BENCH"]
        
        epochs = np.arange(1, len(exp["losses"]["train"]) + 1)
        train_loss = exp["losses"]["train"]
        val_loss = exp["losses"]["val"]
        train_mcc = exp["metrics"].get("train_MCC", exp["metrics"].get("train", []))
        val_mcc   = exp["metrics"].get("val_MCC", exp["metrics"].get("val", []))
        preds = np.array(exp.get("predictions", []))
        gts   = np.array(exp.get("ground_truth", []))
        
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        
        # Subplot 1: Loss curves
        axs[0].plot(epochs, train_loss, label="Train Loss")
        axs[0].plot(epochs, val_loss, linestyle="--", label="Val Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title(f"{ablation_name}: Loss Curves")
        axs[0].legend()
        axs[0].spines['top'].set_visible(False)
        axs[0].spines['right'].set_visible(False)
        
        # Subplot 2: MCC curves
        axs[1].plot(epochs, train_mcc, label="Train MCC")
        axs[1].plot(epochs, val_mcc, linestyle="--", label="Val MCC")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("MCC")
        axs[1].set_title(f"{ablation_name}: MCC Curves")
        axs[1].legend()
        axs[1].spines['top'].set_visible(False)
        axs[1].spines['right'].set_visible(False)
        
        # Subplot 3: Confusion Matrix (test set)
        if preds.size and gts.size:
            plot_conf_matrix(axs[2], gts, preds, f"{ablation_name}: Confusion Matrix")
        else:
            axs[2].text(0.5, 0.5, "No test predictions", horizontalalignment="center", verticalalignment="center")
            axs[2].set_title(f"{ablation_name}: Confusion Matrix")
        
        save_and_close(fig, os.path.join("figures", out_filename))
    except Exception as e:
        print(f"Error plotting ablation {ablation_name}: {e}")

#########################################
# 4. Main Paper Ablation Plots (selected ablations)
#########################################
# Plot for RemovePositionalEmbedding, NoDropout, SingleHeadAttention
plot_ablation("RemovePositionalEmbedding", 
              "experiment_results/experiment_36902de587c44bc3b9f0087b7234a0bd_proc_3345783/experiment_data.npy",
              "RemovePositionalEmbedding.png")
plot_ablation("NoDropout", 
              "experiment_results/experiment_f554a19a3fd94ae188322bc22b82f251_proc_3345786/experiment_data.npy",
              "NoDropout.png")
plot_ablation("SingleHeadAttention", 
              "experiment_results/experiment_d6023b2e616f429180d925fc63383620_proc_3345786/experiment_data.npy",
              "SingleHeadAttention.png")

#########################################
# 5. Appendix Ablation Plots (remaining experiments)
#########################################
plot_ablation("NoCLS_MeanPooling", 
              "experiment_results/experiment_909fe2658add469dbb8e91530d25d9e8_proc_3335814/experiment_data.npy",
              "NoCLS_MeanPooling.png")
plot_ablation("NoPaddingMask", 
              "experiment_results/experiment_3168ed8f2a6d4f5ebb6ab0b8d2770e8a_proc_3345785/experiment_data.npy",
              "NoPaddingMask.png")
plot_ablation("NoFeedForwardLayer", 
              "experiment_results/experiment_b680285c088745ef8d338acb364ee956_proc_3345783/experiment_data.npy",
              "NoFeedForwardLayer.png")
plot_ablation("SinusoidalPositionalEmbedding", 
              "experiment_results/experiment_38e3f739fd314986b409ca8f227f458f_proc_3345784/experiment_data.npy",
              "SinusoidalPositionalEmbedding.png")
plot_ablation("FrozenEmbeddingLayer", 
              "experiment_results/experiment_781ed71a5fb34d769c80ac705621d5c1_proc_3345785/experiment_data.npy",
              "FrozenEmbeddingLayer.png")

#########################################
# 6. Aggregate Ablation Comparison Plot (Bar chart of Test MCC & F1)
#########################################
try:
    # List of ablation names and their corresponding npy file paths
    ablations = {
        "RemovePositionalEmbedding": "experiment_results/experiment_36902de587c44bc3b9f0087b7234a0bd_proc_3345783/experiment_data.npy",
        "NoCLS_MeanPooling": "experiment_results/experiment_909fe2658add469dbb8e91530d25d9e8_proc_3335814/experiment_data.npy",
        "NoPaddingMask": "experiment_results/experiment_3168ed8f2a6d4f5ebb6ab0b8d2770e8a_proc_3345785/experiment_data.npy",
        "NoDropout": "experiment_results/experiment_f554a19a3fd94ae188322bc22b82f251_proc_3345786/experiment_data.npy",
        "NoFeedForwardLayer": "experiment_results/experiment_b680285c088745ef8d338acb364ee956_proc_3345783/experiment_data.npy",
        "SinusoidalPositionalEmbedding": "experiment_results/experiment_38e3f739fd314986b409ca8f227f458f_proc_3345784/experiment_data.npy",
        "FrozenEmbeddingLayer": "experiment_results/experiment_781ed71a5fb34d769c80ac705621d5c1_proc_3345785/experiment_data.npy",
        "SingleHeadAttention": "experiment_results/experiment_d6023b2e616f429180d925fc63383620_proc_3345786/experiment_data.npy"
    }
    
    test_mccs = []
    test_f1s = []
    labels = []
    
    for ablate, fpath in ablations.items():
        try:
            d = np.load(fpath, allow_pickle=True).item()[ablate]["SPR_BENCH"]
            preds = np.array(d.get("predictions", []))
            gts = np.array(d.get("ground_truth", []))
            if preds.size and gts.size:
                mcc_val = matthews_corrcoef(gts, preds)
                f1_val = f1_score(gts, preds, average="macro")
            else:
                mcc_val, f1_val = 0, 0
            labels.append(ablate)
            test_mccs.append(mcc_val)
            test_f1s.append(f1_val)
        except Exception as e:
            print(f"Error processing ablation {ablate} for aggregate plot: {e}")
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(labels))
    ax[0].bar(x, test_mccs, color="skyblue")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels, rotation=45, ha="right")
    ax[0].set_ylabel("Test MCC")
    ax[0].set_title("Ablation Comparison: Test MCC")
    
    ax[1].bar(x, test_f1s, color="lightgreen")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels, rotation=45, ha="right")
    ax[1].set_ylabel("Test Macro-F1")
    ax[1].set_title("Ablation Comparison: Test Macro-F1")
    
    save_and_close(fig, os.path.join("figures", "Aggregate_Ablations_Comparison.png"))
except Exception as e:
    print(f"Error creating Aggregate Ablation Comparison plot: {e}")
    
print("All plots have been generated in the 'figures/' directory.")
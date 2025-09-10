#!/usr/bin/env python3
"""
Final Aggregator Script for Scientific Paper Figures

This script loads experiment data from pre‐saved .npy files, aggregates plots across
the baseline, research, and ablation experiments, and writes final publication‐quality
figures to the "figures/" folder. Each figure is created in a try‐except block so that a failure
in one plot does not stop production of the others.

Data are loaded exactly from the configured file paths in the JSON summaries.

Note:
• All figures use a higher font size for publication clarity.
• Ablation results are aggregated into grouped figures where applicable.
• Only the most unique and informative figures are produced.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

# Set global font size for publication-quality plots
plt.rcParams.update({'font.size': 14})

# Create output directory for final figures
os.makedirs("figures", exist_ok=True)

############################
# Helper plotting functions
############################

def plot_loss_curve(rec, title, save_path):
    """Plot training and validation loss curve from record dict."""
    epochs = rec.get("epochs", [])
    losses = rec.get("losses", {})
    train_loss = losses.get("train", [])
    val_loss = losses.get("val", [])
    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss", marker="o")
    plt.plot(epochs, val_loss, label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_macroF1_curve(rec, title, save_path):
    """Plot training and validation Macro-F1 curve from record dict."""
    epochs = rec.get("epochs", [])
    metrics = rec.get("metrics", {})
    train_f1 = metrics.get("train", [])
    val_f1 = metrics.get("val", [])
    plt.figure()
    plt.plot(epochs, train_f1, label="Train Macro-F1", marker="o")
    plt.plot(epochs, val_f1, label="Validation Macro-F1", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_confusion_matrix(rec, title, save_path):
    """Plot a normalized confusion matrix using predictions and ground_truth."""
    preds = rec.get("predictions", [])
    gts = rec.get("ground_truth", [])
    if preds and gts:
        cmatrix = confusion_matrix(gts, preds, normalize="true")
        plt.figure()
        im = plt.imshow(cmatrix, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

###########################
# Main plotting aggregations
###########################
def main():
    ############################################
    # 1. Baseline Experiment (SPR_BENCH)
    baseline_path = "experiment_results/experiment_179195b55c1b49c89b65435ec46aee6a_proc_3458581/experiment_data.npy"
    try:
        baseline_data = np.load(baseline_path, allow_pickle=True).item()
        baseline_rec = baseline_data.get("SPR_BENCH", {})
    except Exception as e:
        print(f"Error loading baseline data: {e}")
        baseline_rec = {}
        
    # Baseline Loss Curve
    try:
        plot_loss_curve(baseline_rec, "Baseline: Training vs Validation Loss (SPR BENCH)", 
                        os.path.join("figures", "baseline_loss_curve.png"))
    except Exception as e:
        print(f"Error plotting baseline loss curve: {e}")
        
    # Baseline Macro-F1 Curve
    try:
        plot_macroF1_curve(baseline_rec, "Baseline: Training vs Validation Macro-F1 (SPR BENCH)", 
                           os.path.join("figures", "baseline_macroF1_curve.png"))
    except Exception as e:
        print(f"Error plotting baseline Macro-F1 curve: {e}")
        
    # Baseline Confusion Matrix
    try:
        plot_confusion_matrix(baseline_rec, "Baseline: Normalized Confusion Matrix (SPR BENCH)", 
                              os.path.join("figures", "baseline_confusion_matrix.png"))
    except Exception as e:
        print(f"Error plotting baseline confusion matrix: {e}")
        
    ############################################
    # 2. Research Experiment (SPR_BENCH)
    research_path = "experiment_results/experiment_65321ac8ff4547189fa5ca65071deef2_proc_3469031/experiment_data.npy"
    try:
        research_data = np.load(research_path, allow_pickle=True).item()
        research_rec = research_data.get("SPR_BENCH", {})
    except Exception as e:
        print(f"Error loading research data: {e}")
        research_rec = {}
    
    # Research Loss Curve
    try:
        plot_loss_curve(research_rec, "Research: Training vs Validation Loss (SPR BENCH)", 
                        os.path.join("figures", "research_loss_curve.png"))
    except Exception as e:
        print(f"Error plotting research loss curve: {e}")
        
    # Research Macro-F1 Curve
    try:
        plot_macroF1_curve(research_rec, "Research: Training vs Validation Macro-F1 (SPR BENCH)", 
                           os.path.join("figures", "research_macroF1_curve.png"))
    except Exception as e:
        print(f"Error plotting research Macro-F1 curve: {e}")
        
    # Research: Combined Confusion Matrix and Label Distribution
    try:
        preds = research_rec.get("predictions", [])
        gts = research_rec.get("ground_truth", [])
        # Create a two-subplot figure
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        # Confusion Matrix
        if preds and gts:
            cmatrix = confusion_matrix(gts, preds, normalize="true")
            im0 = axs[0].imshow(cmatrix, cmap="Blues")
            axs[0].set_title("Normalized Confusion Matrix")
            axs[0].set_xlabel("Predicted Label")
            axs[0].set_ylabel("True Label")
            fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
        # Label Distribution (Bar Plot)
        if gts:
            labels, counts = np.unique(gts, return_counts=True)
            axs[1].bar(labels, counts, color="gray")
            axs[1].set_title("Ground Truth Label Distribution")
            axs[1].set_xlabel("Label")
            axs[1].set_ylabel("Frequency")
        plt.suptitle("Research Experiment (SPR BENCH)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join("figures", "research_combined_confusion_label.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error creating research combined figure: {e}")
        
    ############################################
    # 3. Ablation Experiments Aggregated Plots
    # For ablation experiments, the structure is:
    #   data = {experiment_name: { "SPR_BENCH": record } }
    
    # Define a helper to load an ablation record given experiment key and file path.
    def load_ablation(exp_key, file_path):
        try:
            data = np.load(file_path, allow_pickle=True).item()
            rec = data.get(exp_key, {}).get("SPR_BENCH", {})
            return rec
        except Exception as e:
            print(f"Error loading {exp_key}: {e}")
            return {}
    
    # (a) Remove_Gating_Mechanism
    gating_path = "experiment_results/experiment_50a2be79119b457db7bf4002c4564740_proc_3475347/experiment_data.npy"
    gating_rec = load_ablation("Remove_Gating_Mechanism", gating_path)
    try:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        # Loss curve
        epochs = gating_rec.get("epochs", [])
        axs[0].plot(epochs, gating_rec.get("losses", {}).get("train", []), label="Train", marker="o")
        axs[0].plot(epochs, gating_rec.get("losses", {}).get("val", []), label="Validation", marker="o")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("Gating: Loss Curves")
        axs[0].legend()
        # Macro-F1 curve
        axs[1].plot(epochs, gating_rec.get("metrics", {}).get("train", []), label="Train", marker="o")
        axs[1].plot(epochs, gating_rec.get("metrics", {}).get("val", []), label="Validation", marker="o")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Macro-F1")
        axs[1].set_title("Gating: Macro-F1 Curves")
        axs[1].legend()
        # Confusion Matrix
        preds = gating_rec.get("predictions", [])
        gts = gating_rec.get("ground_truth", [])
        if preds and gts:
            cmatrix = confusion_matrix(gts, preds)
            im = axs[2].imshow(cmatrix, cmap="Blues")
            axs[2].set_title("Gating: Confusion Matrix")
            axs[2].set_xlabel("Predicted")
            axs[2].set_ylabel("True")
            fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
        plt.suptitle("Ablation: Remove Gating Mechanism (SPR BENCH)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join("figures", "ablation_remove_gating.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error plotting Remove_Gating_Mechanism ablation: {e}")
        
    # (b) Remove_Positional_Encoding
    posenc_path = "experiment_results/experiment_810c7a54116f4d18ad9535cd3824c8b1_proc_3475349/experiment_data.npy"
    posenc_rec = load_ablation("Remove_Positional_Encoding", posenc_path)
    try:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        epochs = posenc_rec.get("epochs", [])
        axs[0].plot(epochs, posenc_rec.get("losses", {}).get("train", []), label="Train", marker="o")
        axs[0].plot(epochs, posenc_rec.get("losses", {}).get("val", []), label="Validation", marker="o")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("PosEnc Removal: Loss Curves")
        axs[0].legend()
        axs[1].plot(epochs, posenc_rec.get("metrics", {}).get("train", []), label="Train", marker="o")
        axs[1].plot(epochs, posenc_rec.get("metrics", {}).get("val", []), label="Validation", marker="o")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Macro-F1")
        axs[1].set_title("PosEnc Removal: Macro-F1 Curves")
        axs[1].legend()
        preds = posenc_rec.get("predictions", [])
        gts = posenc_rec.get("ground_truth", [])
        if preds and gts:
            cmatrix = confusion_matrix(gts, preds)
            im = axs[2].imshow(cmatrix, cmap="Blues")
            axs[2].set_title("PosEnc Removal: Confusion Matrix")
            axs[2].set_xlabel("Predicted")
            axs[2].set_ylabel("True")
            fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
        plt.suptitle("Ablation: Remove Positional Encoding (SPR BENCH)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join("figures", "ablation_remove_positional_encoding.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error plotting Remove_Positional_Encoding ablation: {e}")
        
    # (c) Remove_Transformer_Encoder
    transenc_path = "experiment_results/experiment_d431c8105f1e4c47bbf288d937ec82d1_proc_3475350/experiment_data.npy"
    transenc_rec = load_ablation("Remove_Transformer_Encoder", transenc_path)
    try:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        epochs = transenc_rec.get("epochs", [])
        axs[0].plot(epochs, transenc_rec.get("losses", {}).get("train", []), label="Train", marker="o")
        axs[0].plot(epochs, transenc_rec.get("losses", {}).get("val", []), label="Validation", marker="o")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("Transformer-Encoder Removal: Loss Curves")
        axs[0].legend()
        axs[1].plot(epochs, transenc_rec.get("metrics", {}).get("train", []), label="Train", marker="o")
        axs[1].plot(epochs, transenc_rec.get("metrics", {}).get("val", []), label="Validation", marker="o")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Macro-F1")
        axs[1].set_title("Transformer-Encoder Removal: Macro-F1 Curves")
        axs[1].legend()
        preds = transenc_rec.get("predictions", [])
        gts = transenc_rec.get("ground_truth", [])
        if preds and gts:
            cmatrix = confusion_matrix(gts, preds)
            im = axs[2].imshow(cmatrix, cmap="Blues")
            axs[2].set_title("Transformer-Encoder Removal: Confusion Matrix")
            axs[2].set_xlabel("Predicted")
            axs[2].set_ylabel("True")
            fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
        plt.suptitle("Ablation: Remove Transformer Encoder (SPR BENCH)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join("figures", "ablation_remove_transformer_encoder.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error plotting Remove_Transformer_Encoder ablation: {e}")
    
    ############################################
    # 4. Cross-Ablation Comparison: Final Validation Macro-F1 Scores
    # For remaining ablation experiments, we extract the final validation Macro-F1
    # from their records, and compare them in a horizontal bar plot.
    ablation_experiments = {
        "Remove_Symbolic_Feature_Pathway": "experiment_results/experiment_06b6e30139f24245a681af92f496ac7f_proc_3475348/experiment_data.npy",
        "Remove_Bigram_Hashed_Features": "experiment_results/experiment_f8f2780ee057416d97a0615af96cfaef_proc_3475350/experiment_data.npy",
        "Remove_Unigram_Count_Features": "experiment_results/experiment_f00da480006e45aba494926217801d47_proc_3475349/experiment_data.npy",
        "Remove_Label_Smoothing_Loss": "experiment_results/experiment_b3902c60a0e14c5ab38d1bf75831e7c4_proc_3475347/experiment_data.npy"
    }
    final_val_f1 = {}
    for exp_key, path in ablation_experiments.items():
        try:
            data = np.load(path, allow_pickle=True).item()
            rec = data.get(exp_key, {}).get("SPR_BENCH", {})
            metrics = rec.get("metrics", {})
            if metrics.get("val", []):
                final_val_f1[exp_key] = metrics["val"][-1]
        except Exception as e:
            print(f"Error extracting final Macro-F1 for {exp_key}: {e}")

    try:
        if final_val_f1:
            names = list(final_val_f1.keys())
            scores = [final_val_f1[n] for n in names]
            plt.figure(figsize=(8, 5))
            plt.barh(names, scores, color="teal")
            plt.xlabel("Final Validation Macro-F1")
            plt.title("Ablation Comparison: Final Validation Macro-F1")
            plt.tight_layout()
            plt.savefig(os.path.join("figures", "ablation_final_val_macroF1_comparison.png"), dpi=300)
            plt.close()
    except Exception as e:
        print(f"Error plotting cross-ablation final Macro-F1 comparison: {e}")
        
    print("All final figures have been saved in the 'figures/' directory.")

if __name__ == "__main__":
    main()
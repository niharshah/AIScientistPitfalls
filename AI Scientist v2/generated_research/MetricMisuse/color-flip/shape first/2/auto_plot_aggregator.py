#!/usr/bin/env python3
"""
Aggregated final plots for "Context-Aware Contrastive Learning for Enhanced Symbolic Pattern Recognition"

This script loads experiment result data from the given .npy files (using full, exact file paths)
and produces a comprehensive set of publishable figures saved in the "figures/" directory.
Each plot is generated in a try–except block so that failure in one does not prevent the others.
We produce 12 final figures (3 each for Baseline, Research, No‐Contrastive Ablation, and Dual‐Encoder Ablation)
for inclusion in the main paper. (Other ablation plots can be saved in the appendix separately.)

All text elements use an increased font size for readability.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import itertools

# Set global aesthetics
plt.rcParams.update({'font.size': 14})
# Remove top/right spines for a professional look in most cases later (we will set this for each axis)
def style_axes(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return ax

# Ensure final figures directory
os.makedirs("figures", exist_ok=True)

def load_experiment_data(file_path):
    try:
        data = np.load(file_path, allow_pickle=True).item()
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

##############################################
# Baseline Plots (Using Num_Epochs Tuning results)
##############################################
def plot_baseline():
    # File from BASELINE_SUMMARY "best node"
    base_file = "experiment_results/experiment_c70a6808b9a045d7a11aee55dc386cd5_proc_2991852/experiment_data.npy"
    exp_data = load_experiment_data(base_file)
    if exp_data is None: 
        return
    # The data is stored under key "num_epochs_tuning" -> "SPR_BENCH"
    spr_runs = exp_data.get("num_epochs_tuning", {}).get("SPR_BENCH", {})
    
    # Plot 1: Training & Validation loss curves (side-by-side subplots)
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for epochs_str, logs in spr_runs.items():
            tr_loss = logs["losses"]["train"]
            val_loss = logs["losses"]["val"]
            epochs = range(1, len(tr_loss)+1)
            axes[0].plot(epochs, tr_loss, label=f"{epochs_str} epochs")
            axes[1].plot(epochs, val_loss, label=f"{epochs_str} epochs")
        axes[0].set_title("Train Loss (SPR BENCH)")
        axes[1].set_title("Validation Loss (SPR BENCH)")
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            style_axes(ax)
            ax.legend()
        fig.suptitle("SPR BENCH Loss Curves Across Epoch Settings", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join("figures", "baseline_loss_curves.png"), dpi=300)
        plt.close()
    except Exception as e:
        print("Error creating baseline loss curves:", e)
        plt.close()
    
    # Plot 2: Validation Harmonic Weighted Accuracy (HWA) curves
    try:
        plt.figure(figsize=(6,5))
        for epochs_str, logs in spr_runs.items():
            # Assuming logs["metrics"]["val"] is a list of dicts with key "hwa"
            hwa = [m["hwa"] for m in logs["metrics"]["val"]]
            epochs = range(1, len(hwa)+1)
            plt.plot(epochs, hwa, label=f"{epochs_str} epochs")
        plt.title("SPR BENCH Validation HWA vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        style_axes(plt.gca())
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "baseline_HWA_curves.png"), dpi=300)
        plt.close()
    except Exception as e:
        print("Error creating baseline HWA curves:", e)
        plt.close()
        
    # Plot 3: Best HWA Bar Chart
    try:
        best_hwa = {int(k): max(m["hwa"] for m in v["metrics"]["val"]) for k, v in spr_runs.items()}
        xs, ys = zip(*sorted(best_hwa.items()))
        plt.figure(figsize=(6,5))
        plt.bar([str(x) for x in xs], ys, color="skyblue", edgecolor="black")
        plt.title("Best HWA vs Maximum Epochs (Baseline)")
        plt.xlabel("Max Epochs")
        plt.ylabel("Best HWA")
        style_axes(plt.gca())
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "baseline_best_HWA.png"), dpi=300)
        plt.close()
    except Exception as e:
        print("Error creating baseline best HWA bar chart:", e)
        plt.close()

##############################################
# Research Plots (Joint Training)
##############################################
def plot_research():
    # File from RESEARCH_SUMMARY "best node"
    res_file = "experiment_results/experiment_201dcc6bd7f64e02b2b6efda481739a2_proc_2999655/experiment_data.npy"
    exp_data = load_experiment_data(res_file)
    if exp_data is None:
        return
    jt = exp_data.get("joint_training", {})
    loss_tr = jt.get("losses", {}).get("train", [])
    loss_val = jt.get("losses", {}).get("val", [])
    metrics_val = jt.get("metrics", {}).get("val", [])
    preds = jt.get("predictions", [])
    gts = jt.get("ground_truth", [])
    
    # Plot 4: Joint-Training Loss Curves (Train vs Val)
    try:
        epochs = range(1, len(loss_tr)+1)
        fig, axes = plt.subplots(1,2, figsize=(12,5))
        axes[0].plot(epochs, loss_tr, label="Train")
        axes[1].plot(epochs, loss_val, label="Validation", color="orange")
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            style_axes(ax)
            ax.legend()
        axes[0].set_title("Train Loss (Joint Training)")
        axes[1].set_title("Validation Loss (Joint Training)")
        fig.suptitle("SPR BENCH Joint-Training Loss Curves", fontsize=16)
        plt.tight_layout(rect=[0,0.03,1,0.95])
        plt.savefig(os.path.join("figures", "research_joint_loss_curves.png"), dpi=300)
        plt.close()
    except Exception as e:
        print("Error creating research loss curves:", e)
        plt.close()
    
    # Plot 5: Validation Metrics Curves (SWA, CWA, CCWA)
    try:
        if metrics_val:
            epochs = [m["epoch"] for m in metrics_val]
            swa = [m["swa"] for m in metrics_val]
            cwa = [m["cwa"] for m in metrics_val]
            ccwa = [m["ccwa"] for m in metrics_val]
            plt.figure(figsize=(6,5))
            plt.plot(epochs, swa, label="SWA")
            plt.plot(epochs, cwa, label="CWA")
            plt.plot(epochs, ccwa, label="CCWA")
            plt.xlabel("Epoch")
            plt.ylabel("Metric Value")
            style_axes(plt.gca())
            plt.title("Validation Metrics (Joint Training)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join("figures", "research_metric_curves.png"), dpi=300)
            plt.close()
    except Exception as e:
        print("Error creating research metric curves:", e)
        plt.close()
    
    # Plot 6: Confusion Matrix 
    try:
        if preds and gts:
            classes = sorted(set(gts))
            n = len(classes)
            cm = np.zeros((n, n), dtype=int)
            for t, p in zip(gts, preds):
                cm[t][p] += 1
            plt.figure(figsize=(5,5))
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.title("Confusion Matrix (Joint Training)")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.xticks(ticks=range(n), labels=classes)
            plt.yticks(ticks=range(n), labels=classes)
            for i, j in itertools.product(range(n), range(n)):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plt.tight_layout()
            plt.savefig(os.path.join("figures", "research_confusion_matrix.png"), dpi=300)
            plt.close()
    except Exception as e:
        print("Error creating research confusion matrix:", e)
        plt.close()

##############################################
# Ablation: No-Contrastive Training (α = 0)
##############################################
def plot_ablation_no_contrastive():
    # File from ABLATION_SUMMARY: "No-Contrastive Training (α = 0)"
    abl_file = "experiment_results/experiment_4c6b59fd0d114d6994f652f0e740e598_proc_3006673/experiment_data.npy"
    exp_data = load_experiment_data(abl_file)
    if exp_data is None:
        return
    node = exp_data.get("no_contrastive", {}).get("spr_bench", {})
    if not node:
        print("No data for no_contrastive ablation")
        return
    train_loss = np.array(node["losses"]["train"])
    val_loss = np.array(node["losses"]["val"])
    epochs = np.arange(1, len(train_loss)+1)
    # Plot 7: Loss Curves
    try:
        plt.figure(figsize=(6,5))
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        style_axes(plt.gca())
        plt.title("Loss Curves (No-Contrastive)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "abl_no_contrastive_loss_curves.png"), dpi=300)
        plt.close()
    except Exception as e:
        print("Error creating ablation no-contrastive loss curves:", e)
        plt.close()
    
    # Plot 8: Weighted Accuracies Curves (SWA, CWA, CCWA)
    try:
        metrics_val = node["metrics"]["val"]
        swa_vals = np.array([m["swa"] for m in metrics_val])
        cwa_vals = np.array([m["cwa"] for m in metrics_val])
        ccwa_vals = np.array([m["ccwa"] for m in metrics_val])
        plt.figure(figsize=(6,5))
        plt.plot(epochs, swa_vals, label="SWA")
        plt.plot(epochs, cwa_vals, label="CWA")
        plt.plot(epochs, ccwa_vals, label="CCWA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        style_axes(plt.gca())
        plt.title("Weighted Accuracies (No-Contrastive)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "abl_no_contrastive_metrics.png"), dpi=300)
        plt.close()
    except Exception as e:
        print("Error creating ablation no-contrastive metrics curves:", e)
        plt.close()
    
    # Plot 9: Confusion Matrix for Ablation No-Contrastive
    try:
        preds = node.get("predictions", [])
        trues = node.get("ground_truth", [])
        if preds and trues:
            # assume labels are numeric
            labels = sorted(set(trues))
            n = len(labels)
            cm = np.zeros((n, n), dtype=int)
            for t, p in zip(trues, preds):
                cm[t, p] += 1
            plt.figure(figsize=(5,5))
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.title("Confusion Matrix (No-Contrastive)")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.xticks(range(n), labels)
            plt.yticks(range(n), labels)
            for i, j in itertools.product(range(n), range(n)):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plt.tight_layout()
            plt.savefig(os.path.join("figures", "abl_no_contrastive_confusion_matrix.png"), dpi=300)
            plt.close()
    except Exception as e:
        print("Error creating ablation no-contrastive confusion matrix:", e)
        plt.close()

##############################################
# Ablation: Dual-Encoder Contrastive (No Weight Sharing)
##############################################
def plot_ablation_dual_encoder():
    # File from ABLATION_SUMMARY: "Dual-Encoder Contrastive (No Weight Sharing)"
    dual_file = "experiment_results/experiment_40181c141cc04b2ab7da4102228fd577_proc_3006676/experiment_data.npy"
    exp_data = load_experiment_data(dual_file)
    if exp_data is None:
        return
    node = exp_data.get("dual_encoder_no_share", {}).get("spr_bench", {})
    if not node:
        print("No data for dual_encoder_no_share ablation")
        return
    train_loss = node["losses"]["train"]
    val_loss = node["losses"]["val"]
    epochs = range(1, len(train_loss)+1)
    metrics_val = node.get("metrics", {}).get("val", [])
    
    # Plot 10: Loss Curves
    try:
        plt.figure(figsize=(6,5))
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        style_axes(plt.gca())
        plt.title("Loss Curves (Dual-Encoder)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "abl_dual_encoder_loss_curves.png"), dpi=300)
        plt.close()
    except Exception as e:
        print("Error creating dual encoder loss curves:", e)
        plt.close()
    
    # Plot 11: Metric Curves (SWA, CWA, CCWA)
    try:
        if metrics_val:
            epochs_metrics = [m["epoch"] for m in metrics_val]
            swa = [m["swa"] for m in metrics_val]
            cwa = [m["cwa"] for m in metrics_val]
            ccwa = [m["ccwa"] for m in metrics_val]
            plt.figure(figsize=(6,5))
            plt.plot(epochs_metrics, swa, label="SWA")
            plt.plot(epochs_metrics, cwa, label="CWA")
            plt.plot(epochs_metrics, ccwa, label="CCWA")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            style_axes(plt.gca())
            plt.title("Validation Metrics (Dual-Encoder)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join("figures", "abl_dual_encoder_metrics.png"), dpi=300)
            plt.close()
    except Exception as e:
        print("Error creating dual encoder metric curves:", e)
        plt.close()
    
    # Plot 12: Confusion Matrix
    try:
        preds = node.get("predictions", [])
        gts = node.get("ground_truth", [])
        if preds and gts:
            num_lab = max(max(preds), max(gts)) + 1
            cm = np.zeros((num_lab, num_lab), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure(figsize=(5,5))
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.title("Confusion Matrix (Dual-Encoder)")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.xticks(range(num_lab), list(range(num_lab)))
            plt.yticks(range(num_lab), list(range(num_lab)))
            for i, j in itertools.product(range(num_lab), range(num_lab)):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plt.tight_layout()
            plt.savefig(os.path.join("figures", "abl_dual_encoder_confusion_matrix.png"), dpi=300)
            plt.close()
    except Exception as e:
        print("Error creating dual encoder confusion matrix:", e)
        plt.close()

def main():
    print("Generating Baseline plots ...")
    plot_baseline()
    print("Generating Research plots ...")
    plot_research()
    print("Generating Ablation (No-Contrastive) plots ...")
    plot_ablation_no_contrastive()
    print("Generating Ablation (Dual-Encoder) plots ...")
    plot_ablation_dual_encoder()
    print("All plots generated (saved under 'figures/').")

if __name__ == "__main__":
    main()
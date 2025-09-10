#!/usr/bin/env python3
"""
Final Aggregator Script for Contextual Embedding-Based SPR Research Figures
This script loads experimental .npy results from baseline, research and ablation studies,
aggregates them into a comprehensive set of publication‐quality figures, and saves all plots
into the "figures/" directory.
Each plot (or aggregated group of plots) is wrapped in its own try/except block so that a failure
in one section does not stop the entire script.
All numerical data are loaded from existing .npy files (using full and exact paths provided in the summaries).

Author: AI Researcher
Date: 2023-10
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Increase font sizes for publication quality
plt.rcParams.update({'font.size': 14, 'figure.dpi': 300})
# Remove top/right spines for a clean look in all plots later on
for spine in ['top', 'right']:
    plt.rcParams["axes.spines." + spine] = False

# Create output figures directory (only this folder is used in final paper)
os.makedirs("figures", exist_ok=True)

###############################################################################
# Helper functions
###############################################################################
def compute_macro_f1(preds, labels, num_cls):
    f1s = []
    for c in range(num_cls):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
        else:
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1s.append(0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))

###############################################################################
# 1. Baseline Results (from first JSON: BASELINE_SUMMARY)
###############################################################################
baseline_file = "experiment_results/experiment_89abcd2ab84b4fb89db44a3c173d28b4_proc_3154414/experiment_data.npy"
try:
    base_all = np.load(baseline_file, allow_pickle=True).item()
    # Expecting key "SPR_BENCH" inside
    base = base_all["SPR_BENCH"]
    # Expect keys: losses: {"train_loss", "val_loss"}, metrics: {"train_acc", "val_acc"}, predictions and ground_truth.
    epochs = np.arange(1, len(base["losses"]["train_loss"]) + 1)
    
    # Figure 1: Loss & Accuracy curves (aggregated in one figure, two subplots)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # Loss curves
    axs[0].plot(epochs, base["losses"]["train_loss"], label="Train Loss", marker="o")
    axs[0].plot(epochs, base["losses"]["val_loss"], label="Validation Loss", marker="o")
    axs[0].set_title("Baseline SPR: Loss vs Epoch")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Cross-Entropy Loss")
    axs[0].legend(loc="best")
    # Accuracy curves
    axs[1].plot(epochs, base["metrics"]["train_acc"], label="Train Accuracy", marker="o")
    axs[1].plot(epochs, base["metrics"]["val_acc"], label="Validation Accuracy", marker="o")
    axs[1].set_title("Baseline SPR: Accuracy vs Epoch")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend(loc="best")
    plt.tight_layout()
    fig_path = os.path.join("figures", "Baseline_Loss_and_Accuracy.png")
    plt.savefig(fig_path)
    plt.close(fig)
    print("Saved Baseline Loss & Accuracy curves:", fig_path)
except Exception as e:
    print("Error in Baseline Loss/Acc plots:", e)
    plt.close()

# Figure 2: Baseline Confusion Matrix
try:
    preds = np.array(base["predictions"])
    gts = np.array(base["ground_truth"])
    num_classes = int(max(preds.max(), gts.max()) + 1)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for gt, pr in zip(gts, preds):
        cm[gt, pr] += 1
    plt.figure(figsize=(5, 5))
    im = plt.imshow(cm, cmap="Blues")
    plt.title("Baseline SPR: Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.xticks(range(num_classes))
    plt.yticks(range(num_classes))
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig_path = os.path.join("figures", "Baseline_Confusion_Matrix.png")
    plt.savefig(fig_path)
    plt.close()
    test_acc = (preds == gts).mean()
    print(f"Baseline Test Accuracy: {test_acc*100:.2f}% (Confusion Matrix saved at {fig_path})")
except Exception as e:
    print("Error in Baseline Confusion Matrix plot:", e)
    plt.close()

###############################################################################
# 2. Research Results (from RESEARCH_SUMMARY)
###############################################################################
research_file = "experiment_results/experiment_e1175791daf84894abd198fece9b2a3b_proc_3165651/experiment_data.npy"
try:
    res_all = np.load(research_file, allow_pickle=True).item()
    research = res_all["SPR_BENCH"]
    epochs = np.arange(1, len(research["losses"]["train"]) + 1)
    # Figure 3: Aggregated Learning Curves from Research (Loss, Accuracy, Macro-F1)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    # Loss curves
    axs[0].plot(epochs, research["losses"]["train"], label="Train Loss", marker="o")
    axs[0].plot(epochs, research["losses"]["val"], label="Validation Loss", marker="o")
    axs[0].set_title("Research SPR: Loss vs Epoch")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend(loc="best")
    # Accuracy curves
    axs[1].plot(epochs, research["metrics"]["train_acc"], label="Train Accuracy", marker="o")
    axs[1].plot(epochs, research["metrics"]["val_acc"], label="Validation Accuracy", marker="o")
    axs[1].set_title("Research SPR: Accuracy vs Epoch")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend(loc="best")
    # Macro-F1 curves
    axs[2].plot(epochs, research["metrics"]["train_f1"], label="Train Macro-F1", marker="o")
    axs[2].plot(epochs, research["metrics"]["val_f1"], label="Validation Macro-F1", marker="o")
    axs[2].set_title("Research SPR: Macro-F1 vs Epoch")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Macro-F1")
    axs[2].legend(loc="best")
    plt.tight_layout()
    fig_path = os.path.join("figures", "Research_Learning_Curves.png")
    plt.savefig(fig_path)
    plt.close(fig)
    print("Saved Research Learning Curves:", fig_path)
except Exception as e:
    print("Error in Research Learning Curves plot:", e)
    plt.close()

# Figure 4: Research Confusion Matrix
try:
    preds = np.array(research.get("predictions", []))
    gts = np.array(research.get("ground_truth", []))
    if preds.size and gts.size:
        num_classes = int(max(preds.max(), gts.max()) + 1)
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for gt, pr in zip(gts, preds):
            cm[gt, pr] += 1
        plt.figure(figsize=(5, 5))
        im = plt.imshow(cm, cmap="Blues")
        plt.title("Research SPR: Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.xticks(range(num_classes))
        plt.yticks(range(num_classes))
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        fig_path = os.path.join("figures", "Research_Confusion_Matrix.png")
        plt.savefig(fig_path)
        plt.close()
        test_acc = (preds == gts).mean()
        print(f"Research Test Accuracy: {test_acc*100:.2f}% (Confusion Matrix saved at {fig_path})")
    else:
        print("No predictions/ground-truth in Research data.")
except Exception as e:
    print("Error in Research Confusion Matrix plot:", e)
    plt.close()

###############################################################################
# 3. Ablation Studies
###############################################################################
# 3.1 Remove Positional Encoding (from ABLATION_SUMMARY)
remove_pe_file = "experiment_results/experiment_d3ac1e5060024140bd05e1c41e4086a6_proc_3173675/experiment_data.npy"
try:
    ablation_all = np.load(remove_pe_file, allow_pickle=True).item()
    # Expecting structure: ablation_all["SPR_BENCH"] inside "Remove Positional Encoding"
    remove_pe = ablation_all.get("no_positional_encoding", {}).get("SPR_BENCH", {})
    if remove_pe:
        epochs = np.arange(1, len(remove_pe.get("losses", {}).get("train", [])) + 1)
        # Figure 5: Remove Positional Encoding: Loss, Accuracy, Macro-F1 aggregated (3 subplots)
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        # Loss curves
        axs[0].plot(epochs, remove_pe["losses"]["train"], label="Train Loss", marker="o")
        axs[0].plot(epochs, remove_pe["losses"]["val"], label="Validation Loss", marker="o")
        axs[0].set_title("Ablation (No Positional Encoding): Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].legend(loc="best")
        # Accuracy curves
        axs[1].plot(epochs, remove_pe["metrics"]["train_acc"], label="Train Acc", marker="o")
        axs[1].plot(epochs, remove_pe["metrics"]["val_acc"], label="Val Acc", marker="o")
        axs[1].set_title("Ablation (No Positional Encoding): Accuracy")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Accuracy")
        axs[1].legend(loc="best")
        # Macro-F1 curves
        axs[2].plot(epochs, remove_pe["metrics"]["train_f1"], label="Train Macro-F1", marker="o")
        axs[2].plot(epochs, remove_pe["metrics"]["val_f1"], label="Val Macro-F1", marker="o")
        axs[2].set_title("Ablation (No Positional Encoding): Macro-F1")
        axs[2].set_xlabel("Epoch")
        axs[2].set_ylabel("Macro-F1")
        axs[2].legend(loc="best")
        plt.tight_layout()
        fig_path = os.path.join("figures", "Ablation_No_PositionalEncoding_Curves.png")
        plt.savefig(fig_path)
        plt.close(fig)
        print("Saved Ablation (No Positional Encoding) learning curves:", fig_path)
    else:
        print("No data found for Remove Positional Encoding ablation.")
except Exception as e:
    print("Error in Remove Positional Encoding aggregated plot:", e)
    plt.close()

# Figure 6: Remove Positional Encoding: Confusion Matrix
try:
    if remove_pe:
        preds = np.array(remove_pe.get("predictions", []))
        gts = np.array(remove_pe.get("ground_truth", []))
        if preds.size and gts.size:
            num_classes = int(max(preds.max(), gts.max()) + 1)
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for gt, pr in zip(gts, preds):
                cm[gt, pr] += 1
            plt.figure(figsize=(5, 5))
            im = plt.imshow(cm, cmap="Blues")
            plt.title("Ablation (No Positional Encoding): Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.xticks(range(num_classes))
            plt.yticks(range(num_classes))
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.tight_layout()
            fig_path = os.path.join("figures", "Ablation_No_PositionalEncoding_Confusion.png")
            plt.savefig(fig_path)
            plt.close()
            print("Saved Ablation (No Positional Encoding) confusion matrix:", fig_path)
except Exception as e:
    print("Error in Remove Positional Encoding confusion matrix plot:", e)
    plt.close()

# 3.2 No-CLS Mean Pooling (from ABLATION_SUMMARY)
nocls_file = "experiment_results/experiment_b6e6287e6c0e458da22bf4c0fcb3f2d3_proc_3173676/experiment_data.npy"
try:
    nocls_all = np.load(nocls_file, allow_pickle=True).item()
    nocls = nocls_all["SPR_BENCH"]
    epochs = np.arange(1, len(nocls["losses"]["train"]) + 1)
    # Figure 7: No-CLS Mean Pooling aggregated curves: Loss, Accuracy, Macro-F1
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].plot(epochs, nocls["losses"]["train"], label="Train Loss", marker="o")
    axs[0].plot(epochs, nocls["losses"]["val"], label="Validation Loss", marker="o")
    axs[0].set_title("No-CLS Mean Pooling: Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend(loc="best")
    
    axs[1].plot(epochs, nocls["metrics"]["train_acc"], label="Train Accuracy", marker="o")
    axs[1].plot(epochs, nocls["metrics"]["val_acc"], label="Validation Accuracy", marker="o")
    axs[1].set_title("No-CLS Mean Pooling: Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend(loc="best")
    
    axs[2].plot(epochs, nocls["metrics"]["train_f1"], label="Train Macro-F1", marker="o")
    axs[2].plot(epochs, nocls["metrics"]["val_f1"], label="Validation Macro-F1", marker="o")
    axs[2].set_title("No-CLS Mean Pooling: Macro-F1")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Macro-F1")
    axs[2].legend(loc="best")
    plt.tight_layout()
    fig_path = os.path.join("figures", "Ablation_NoCLS_MeanPooling_Curves.png")
    plt.savefig(fig_path)
    plt.close(fig)
    print("Saved No-CLS Mean Pooling learning curves:", fig_path)
    
    # Figure 8: No-CLS Mean Pooling Confusion Matrix
    preds = np.array(nocls.get("predictions", []))
    gts = np.array(nocls.get("ground_truth", []))
    if preds.size and gts.size:
        num_classes = int(max(preds.max(), gts.max()) + 1)
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for gt, pr in zip(gts, preds):
            cm[gt, pr] += 1
        plt.figure(figsize=(5, 5))
        im = plt.imshow(cm, cmap="Blues")
        plt.title("No-CLS Mean Pooling: Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.xticks(range(num_classes))
        plt.yticks(range(num_classes))
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        fig_path = os.path.join("figures", "Ablation_NoCLS_MeanPooling_Confusion.png")
        plt.savefig(fig_path)
        plt.close()
        print("Saved No-CLS Mean Pooling confusion matrix:", fig_path)
except Exception as e:
    print("Error in No-CLS Mean Pooling plots:", e)
    plt.close()

# 3.3 Token Order Shuffling (Bag-of-Words) (from ABLATION_SUMMARY)
token_shuffle_file = "experiment_results/experiment_4827a969888d455faf2d60e8f91aa7b5_proc_3173674/experiment_data.npy"
try:
    tok_all = np.load(token_shuffle_file, allow_pickle=True).item()
    # For Token Order Shuffling, assume structure similar to other ablations under key "SPR_BENCH"
    token_ablation = tok_all["SPR_BENCH"]
    epochs = np.arange(1, len(token_ablation["losses"]["train"]) + 1)
    # Figure 9: Token Order Shuffling aggregated curves: Loss, Accuracy, Macro-F1 in one figure
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].plot(epochs, token_ablation["losses"]["train"], label="Train Loss", marker="o")
    axs[0].plot(epochs, token_ablation["losses"]["val"], label="Validation Loss", marker="o")
    axs[0].set_title("Token Order Shuffling: Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend(loc="best")
    
    axs[1].plot(epochs, token_ablation["metrics"]["train_acc"], label="Train Accuracy", marker="o")
    axs[1].plot(epochs, token_ablation["metrics"]["val_acc"], label="Validation Accuracy", marker="o")
    axs[1].set_title("Token Order Shuffling: Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend(loc="best")
    
    axs[2].plot(epochs, token_ablation["metrics"]["train_f1"], label="Train Macro-F1", marker="o")
    axs[2].plot(epochs, token_ablation["metrics"]["val_f1"], label="Validation Macro-F1", marker="o")
    axs[2].set_title("Token Order Shuffling: Macro-F1")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Macro-F1")
    axs[2].legend(loc="best")
    plt.tight_layout()
    fig_path = os.path.join("figures", "Ablation_TokenOrderShuffling_Curves.png")
    plt.savefig(fig_path)
    plt.close(fig)
    print("Saved Token Order Shuffling learning curves:", fig_path)
    
    # Figure 10: Token Order Shuffling confusion matrix
    preds = np.array(token_ablation.get("predictions", []))
    gts = np.array(token_ablation.get("ground_truth", []))
    if preds.size and gts.size:
        num_classes = int(max(preds.max(), gts.max()) + 1)
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for gt, pr in zip(gts, preds):
            cm[gt, pr] += 1
        plt.figure(figsize=(5, 5))
        im = plt.imshow(cm, cmap="Blues")
        plt.title("Token Order Shuffling: Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.xticks(range(num_classes))
        plt.yticks(range(num_classes))
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        fig_path = os.path.join("figures", "Ablation_TokenOrderShuffling_Confusion.png")
        plt.savefig(fig_path)
        plt.close()
        print("Saved Token Order Shuffling confusion matrix:", fig_path)
except Exception as e:
    print("Error in Token Order Shuffling plots:", e)
    plt.close()

# 3.4 Learned Positional Embeddings (from ABLATION_SUMMARY)
learned_pe_file = "experiment_results/experiment_3c56282f781b4aefbeec99f7a5fc85fb_proc_3173677/experiment_data.npy"
try:
    lp_all = np.load(learned_pe_file, allow_pickle=True).item()
    lp = lp_all["SPR_BENCH"]
    epochs = np.arange(1, len(lp["losses"]["train"]) + 1)
    # Figure 11: Learned Positional Embeddings aggregated curves: Loss, Accuracy, Macro-F1, and Confusion Matrix (4 subplots)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    # Loss curves
    axs[0, 0].plot(epochs, lp["losses"]["train"], label="Train Loss", marker="o")
    axs[0, 0].plot(epochs, lp["losses"]["val"], label="Validation Loss", marker="o")
    axs[0, 0].set_title("Learned PE: Loss")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend(loc="best")
    # Accuracy curves
    axs[0, 1].plot(epochs, lp["metrics"]["train_acc"], label="Train Accuracy", marker="o")
    axs[0, 1].plot(epochs, lp["metrics"]["val_acc"], label="Validation Accuracy", marker="o")
    axs[0, 1].set_title("Learned PE: Accuracy")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Accuracy")
    axs[0, 1].legend(loc="best")
    # Macro-F1 curves
    axs[1, 0].plot(epochs, lp["metrics"]["train_f1"], label="Train Macro-F1", marker="o")
    axs[1, 0].plot(epochs, lp["metrics"]["val_f1"], label="Validation Macro-F1", marker="o")
    axs[1, 0].set_title("Learned PE: Macro-F1")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("Macro-F1")
    axs[1, 0].legend(loc="best")
    # Confusion Matrix
    preds = np.array(lp.get("predictions", []))
    labels = np.array(lp.get("ground_truth", []))
    if preds.size and labels.size:
        num_classes = int(max(preds.max(), labels.max()) + 1)
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for gt, pr in zip(labels, preds):
            cm[gt, pr] += 1
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = axs[1, 1].imshow(cm_norm, cmap="Blues")
        axs[1, 1].set_title("Learned PE: Norm Confusion Matrix")
        axs[1, 1].set_xlabel("Predicted")
        axs[1, 1].set_ylabel("Ground Truth")
        axs[1, 1].set_xticks(range(num_classes))
        axs[1, 1].set_yticks(range(num_classes))
        plt.colorbar(im, ax=axs[1, 1], fraction=0.046, pad=0.04)
    else:
        axs[1, 1].text(0.5, 0.5, "No predictions", ha="center", va="center")
    plt.tight_layout()
    fig_path = os.path.join("figures", "Ablation_Learned_PositionalEmbeddings_Curves.png")
    plt.savefig(fig_path)
    plt.close(fig)
    print("Saved Learned Positional Embeddings aggregated curves:", fig_path)
except Exception as e:
    print("Error in Learned Positional Embeddings plot:", e)
    plt.close()

###############################################################################
# End of Aggregated Plots Script
###############################################################################
print("Final aggregated plots have been saved in the 'figures/' directory.")
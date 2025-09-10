#!/usr/bin/env python3
"""
Final Aggregator Script for Zero-Shot Synthetic PolyRule Reasoning with Neural Symbolic Integration

This script aggregates experimental results from baseline/research and several ablation studies.
Each figure groups related plots into one figure (with up to 3 subplots) and is saved in the 'figures/' directory.
All data are loaded from existing .npy files using the full paths as specified in the experiment summaries.
Each figure is wrapped in its own try/except block so that failures in one do not affect the others.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Set a larger font size for readability in the final PDF paper
plt.rcParams.update({'font.size': 14})

# Create the output figures directory
fig_dir = "figures"
os.makedirs(fig_dir, exist_ok=True)

# Helper function to remove top and right spines from an axis
def style_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ------------------------
# 1. Baseline / Research Results
#    File path from the summary:
#    "experiment_results/experiment_5032b21fc69c4c59b341004095b32966_proc_2700561/experiment_data.npy"
#    Key: ["EPOCH_TUNING"]["SPR_BENCH"]
# ------------------------
try:
    baseline_file = "experiment_results/experiment_5032b21fc69c4c59b341004095b32966_proc_2700561/experiment_data.npy"
    baseline_data = np.load(baseline_file, allow_pickle=True).item()
    rec = baseline_data["EPOCH_TUNING"]["SPR_BENCH"]
    epochs = list(range(1, len(rec["losses"]["train"]) + 1))
    
    # Figure: Baseline Part 1 - Loss Curves & Validation Accuracy (2 subplots)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: Loss Curves (Train vs Validation)
    axs[0].plot(epochs, rec["losses"]["train"], label="Train")
    axs[0].plot(epochs, rec["losses"]["val"], label="Validation")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Cross-Entropy Loss")
    axs[0].set_title("SPR BENCH Loss Curves")
    axs[0].legend()
    style_axis(axs[0])
    
    # Subplot 2: Validation Accuracy over Epochs
    # Assuming each element in rec["metrics"]["val"] is a dict with key "acc"
    val_acc = [m["acc"] for m in rec["metrics"]["val"]]
    axs[1].plot(epochs, val_acc, marker="o", color="green", label="Validation Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_ylim(0, 1)
    axs[1].set_title("SPR BENCH Validation Accuracy")
    axs[1].legend()
    style_axis(axs[1])
    
    plt.tight_layout()
    baseline_fig1_path = os.path.join(fig_dir, "baseline_loss_and_val_accuracy.png")
    plt.savefig(baseline_fig1_path, dpi=300)
    plt.close()
    print(f"Saved Baseline Figure Part 1: {baseline_fig1_path}")
except Exception as e:
    print(f"Error creating Baseline Figure Part 1: {e}")
    plt.close()


try:
    # Figure: Baseline Part 2 - Test Metrics Bar Chart & Confusion Matrix (2 subplots)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: Test Metrics Bar Chart
    test_metrics = rec["metrics"]["test"]
    metric_labels = ["Acc", "SWA", "CWA", "NRGS"]
    metric_values = [test_metrics[k.lower()] if k.lower() in test_metrics else test_metrics[k] for k in [label.lower() for label in metric_labels]]
    axs[0].bar(metric_labels, metric_values, color="skyblue")
    axs[0].set_ylim(0, 1)
    axs[0].set_title("SPR BENCH Test Metrics")
    axs[0].set_ylabel("Metric Value")
    style_axis(axs[0])
    
    # Subplot 2: Confusion Matrix
    y_true = np.array(rec["ground_truth"])
    y_pred = np.array(rec["predictions"])
    classes = sorted(set(y_true) | set(y_pred))
    num_classes = len(classes)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    im = axs[1].imshow(cm, cmap="Blues")
    axs[1].set_title("SPR BENCH Confusion Matrix")
    axs[1].set_xlabel("Predicted")
    axs[1].set_ylabel("True")
    axs[1].set_xticks(range(num_classes))
    axs[1].set_yticks(range(num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            axs[1].text(j, i, cm[i,j], ha="center", va="center", color="black")
    plt.colorbar(im, ax=axs[1])
    style_axis(axs[1])
    
    plt.tight_layout()
    baseline_fig2_path = os.path.join(fig_dir, "baseline_test_metrics_and_confusion.png")
    plt.savefig(baseline_fig2_path, dpi=300)
    plt.close()
    print(f"Saved Baseline Figure Part 2: {baseline_fig2_path}")
except Exception as e:
    print(f"Error creating Baseline Figure Part 2: {e}")
    plt.close()


# ------------------------
# 2. Ablation Study: Multi-Dataset Generalisation (3× Synthetic SPR Variants)
#    File: "experiment_results/experiment_e2b1e083178f40da9f26f13531fb5fa3_proc_2705534/experiment_data.npy"
#    Key: ["MultiDataset"]["SPR_BENCH_HELDOUT"]
# ------------------------
try:
    multi_file = "experiment_results/experiment_e2b1e083178f40da9f26f13531fb5fa3_proc_2705534/experiment_data.npy"
    multi_data = np.load(multi_file, allow_pickle=True).item()
    rec_multi = multi_data["MultiDataset"]["SPR_BENCH_HELDOUT"]
    epochs_multi = list(range(1, len(rec_multi["losses"]["train"]) + 1))
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Subplot 1: Loss Curves
    axs[0].plot(epochs_multi, rec_multi["losses"]["train"], label="Train")
    axs[0].plot(epochs_multi, rec_multi["losses"]["val"], label="Validation")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("SPR BENCH HELDOUT Loss Curves")
    axs[0].legend()
    style_axis(axs[0])
    
    # Subplot 2: Validation SWA Curve
    val_swa_multi = rec_multi["SWA"]["val"]
    axs[1].plot(epochs_multi, val_swa_multi, marker="o", color="purple", label="Validation SWA")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("SWA")
    axs[1].set_title("SPR BENCH HELDOUT Validation SWA")
    axs[1].legend()
    style_axis(axs[1])
    
    # Subplot 3: Confusion Matrix
    y_true_multi = np.array(rec_multi["ground_truth"])
    y_pred_multi = np.array(rec_multi["predictions"])
    classes_multi = sorted(set(y_true_multi) | set(y_pred_multi))
    n_cls_multi = len(classes_multi)
    cm_multi = np.zeros((n_cls_multi, n_cls_multi), dtype=int)
    for t, p in zip(y_true_multi, y_pred_multi):
        cm_multi[t, p] += 1
    im = axs[2].imshow(cm_multi, cmap="Blues")
    axs[2].set_title("SPR BENCH HELDOUT Confusion Matrix")
    axs[2].set_xlabel("Predicted")
    axs[2].set_ylabel("True")
    axs[2].set_xticks(range(n_cls_multi))
    axs[2].set_yticks(range(n_cls_multi))
    for i in range(n_cls_multi):
        for j in range(n_cls_multi):
            axs[2].text(j, i, cm_multi[i,j], ha="center", va="center", color="black")
    plt.colorbar(im, ax=axs[2])
    style_axis(axs[2])
    
    plt.tight_layout()
    multi_fig_path = os.path.join(fig_dir, "ablation_multidataset.png")
    plt.savefig(multi_fig_path, dpi=300)
    plt.close()
    print(f"Saved Ablation (Multi-Dataset) Figure: {multi_fig_path}")
except Exception as e:
    print(f"Error creating Ablation (Multi-Dataset) Figure: {e}")
    plt.close()


# ------------------------
# 3. Ablation Study: CLS-Token Pooling
#    File: "experiment_results/experiment_bda3ae3819dc4248b321b28b654c6cc2_proc_2705536/experiment_data.npy"
#    Key: ["CLS_Pooling"]["SPR_BENCH"]
# ------------------------
try:
    cls_file = "experiment_results/experiment_bda3ae3819dc4248b321b28b654c6cc2_proc_2705536/experiment_data.npy"
    cls_data = np.load(cls_file, allow_pickle=True).item()
    rec_cls = cls_data["CLS_Pooling"]["SPR_BENCH"]
    epochs_cls = list(range(1, len(rec_cls["losses"]["train"]) + 1))
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Subplot 1: Loss Curves
    axs[0].plot(epochs_cls, rec_cls["losses"]["train"], label="Train Loss")
    axs[0].plot(epochs_cls, rec_cls["losses"]["val"], label="Validation Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Cross-Entropy Loss")
    axs[0].set_title("CLS-Token Pooling: Loss Curves")
    axs[0].legend()
    style_axis(axs[0])
    
    # Subplot 2: Validation SWA
    val_swa_cls = rec_cls["SWA"]["val"]
    axs[1].plot(epochs_cls, val_swa_cls, marker="o", label="Validation SWA", color="darkorange")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("SWA")
    axs[1].set_title("CLS-Token Pooling: Validation SWA")
    axs[1].legend()
    style_axis(axs[1])
    
    # Subplot 3: Confusion Matrix
    y_true_cls = np.array(rec_cls["ground_truth"])
    y_pred_cls = np.array(rec_cls["predictions"])
    classes_cls = sorted(set(y_true_cls) | set(y_pred_cls))
    n_cls_cls = len(classes_cls)
    cm_cls = np.zeros((n_cls_cls, n_cls_cls), dtype=int)
    for t, p in zip(y_true_cls, y_pred_cls):
        cm_cls[t, p] += 1
    im = axs[2].imshow(cm_cls, cmap="Blues")
    axs[2].set_title("CLS-Token Pooling: Confusion Matrix")
    axs[2].set_xlabel("Predicted")
    axs[2].set_ylabel("True")
    axs[2].set_xticks(range(n_cls_cls))
    axs[2].set_yticks(range(n_cls_cls))
    for i in range(n_cls_cls):
        for j in range(n_cls_cls):
            axs[2].text(j, i, cm_cls[i,j], ha="center", va="center", color="black")
    plt.colorbar(im, ax=axs[2])
    style_axis(axs[2])
    
    plt.tight_layout()
    cls_fig_path = os.path.join(fig_dir, "ablation_cls_token_pooling.png")
    plt.savefig(cls_fig_path, dpi=300)
    plt.close()
    print(f"Saved Ablation (CLS-Token Pooling) Figure: {cls_fig_path}")
except Exception as e:
    print(f"Error creating Ablation (CLS-Token Pooling) Figure: {e}")
    plt.close()


# ------------------------
# 4. Ablation Study: No-Positional-Embedding
#    File: "experiment_results/experiment_f32622bc155c47da9ddf2fd2a24f5608_proc_2705537/experiment_data.npy"
#    Key: ["NoPositionalEmbedding"]["SPR_BENCH"]
# ------------------------
try:
    npe_file = "experiment_results/experiment_f32622bc155c47da9ddf2fd2a24f5608_proc_2705537/experiment_data.npy"
    npe_data = np.load(npe_file, allow_pickle=True).item()
    rec_npe = npe_data["NoPositionalEmbedding"]["SPR_BENCH"]
    epochs_npe = list(range(1, len(rec_npe.get("losses", {}).get("train", [])) + 1))
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Subplot 1: Loss Curve
    train_loss_npe = rec_npe.get("losses", {}).get("train", [])
    val_loss_npe   = rec_npe.get("losses", {}).get("val", [])
    if train_loss_npe and val_loss_npe:
        axs[0].plot(epochs_npe, train_loss_npe, label="Train Loss")
        axs[0].plot(epochs_npe, val_loss_npe, label="Validation Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Cross-Entropy Loss")
    axs[0].set_title("No-Positional-Embedding: Loss Curve")
    axs[0].legend()
    style_axis(axs[0])
    
    # Subplot 2: Validation SWA Curve
    val_swa_npe = rec_npe.get("SWA", {}).get("val", [])
    if val_swa_npe:
        axs[1].plot(epochs_npe, val_swa_npe, marker="o", label="Validation SWA", color="teal")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("SWA")
    axs[1].set_title("No-Positional-Embedding: Validation SWA")
    axs[1].legend()
    style_axis(axs[1])
    
    # Subplot 3: Confusion Matrix
    truths_npe = rec_npe.get("ground_truth", [])
    preds_npe  = rec_npe.get("predictions", [])
    if truths_npe and preds_npe:
        truths_npe = np.array(truths_npe)
        preds_npe = np.array(preds_npe)
        classes_npe = sorted(set(truths_npe) | set(preds_npe))
        n_cls_npe = len(classes_npe)
        cm_npe = np.zeros((n_cls_npe, n_cls_npe), dtype=int)
        for t, p in zip(truths_npe, preds_npe):
            cm_npe[t, p] += 1
        im = axs[2].imshow(cm_npe, cmap="Blues")
        axs[2].set_title("No-Positional-Embedding: Confusion Matrix")
        axs[2].set_xlabel("Predicted")
        axs[2].set_ylabel("True")
        axs[2].set_xticks(range(n_cls_npe))
        axs[2].set_yticks(range(n_cls_npe))
        for i in range(n_cls_npe):
            for j in range(n_cls_npe):
                axs[2].text(j, i, cm_npe[i,j], ha="center", va="center", color="black")
        plt.colorbar(im, ax=axs[2])
    style_axis(axs[2])
    
    plt.tight_layout()
    npe_fig_path = os.path.join(fig_dir, "ablation_no_positional_embedding.png")
    plt.savefig(npe_fig_path, dpi=300)
    plt.close()
    print(f"Saved Ablation (No-Positional-Embedding) Figure: {npe_fig_path}")
except Exception as e:
    print(f"Error creating Ablation (No-Positional-Embedding) Figure: {e}")
    plt.close()


# ------------------------
# 5. Ablation Study: No-Transformer (Bag-of-Embeddings)
#    File: "experiment_results/experiment_81bc56ad91344cd093b471adc55aa4a1_proc_2705534/experiment_data.npy"
#    Key: ["BagOfEmbeddings"]["SPR_BENCH"]
# ------------------------
try:
    boe_file = "experiment_results/experiment_81bc56ad91344cd093b471adc55aa4a1_proc_2705534/experiment_data.npy"
    boe_data = np.load(boe_file, allow_pickle=True).item()
    rec_boe = boe_data["BagOfEmbeddings"]["SPR_BENCH"]
    epochs_boe = list(range(1, len(rec_boe["losses"]["train"]) + 1))
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Subplot 1: Loss Curves
    axs[0].plot(epochs_boe, rec_boe["losses"]["train"], label="Train Loss")
    axs[0].plot(epochs_boe, rec_boe["losses"]["val"], label="Validation Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Cross-Entropy Loss")
    axs[0].set_title("Bag-of-Embeddings: Loss Curves")
    axs[0].legend()
    style_axis(axs[0])
    
    # Subplot 2: Validation SWA
    axs[1].plot(epochs_boe, rec_boe["SWA"]["val"], marker="o", color="brown", label="Validation SWA")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("SWA")
    axs[1].set_title("Bag-of-Embeddings: Validation SWA")
    axs[1].legend()
    style_axis(axs[1])
    
    # Subplot 3: Confusion Matrix
    truths_boe = np.array(rec_boe["ground_truth"])
    preds_boe = np.array(rec_boe["predictions"])
    classes_boe = sorted(set(truths_boe) | set(preds_boe))
    n_cls_boe = len(classes_boe)
    cm_boe = np.zeros((n_cls_boe, n_cls_boe), dtype=int)
    for t, p in zip(truths_boe, preds_boe):
        cm_boe[t, p] += 1
    im = axs[2].imshow(cm_boe, cmap="Blues")
    axs[2].set_title("Bag-of-Embeddings: Confusion Matrix")
    axs[2].set_xlabel("Predicted")
    axs[2].set_ylabel("True")
    axs[2].set_xticks(range(n_cls_boe))
    axs[2].set_yticks(range(n_cls_boe))
    for i in range(n_cls_boe):
        for j in range(n_cls_boe):
            axs[2].text(j, i, cm_boe[i,j], ha="center", va="center", color="black")
    plt.colorbar(im, ax=axs[2])
    style_axis(axs[2])
    
    plt.tight_layout()
    boe_fig_path = os.path.join(fig_dir, "ablation_bag_of_embeddings.png")
    plt.savefig(boe_fig_path, dpi=300)
    plt.close()
    print(f"Saved Ablation (Bag-of-Embeddings) Figure: {boe_fig_path}")
except Exception as e:
    print(f"Error creating Ablation (Bag-of-Embeddings) Figure: {e}")
    plt.close()


print("Aggregated all final plots successfully!")
#!/usr/bin/env python3
"""
Aggregator script to load npy experiment results and produce final scientific plots.
All figures are saved in the "figures/" directory.
Each plotting section is wrapped in a try-except block.
Fonts and spines are optionally adjusted for professional appearance.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Increase global font size for readability in final PDF.
plt.rcParams.update({'font.size': 12})

# Ensure figures directory exists.
os.makedirs("figures", exist_ok=True)

def plot_confusion(cm, ax, title="Confusion Matrix"):
    """Helper to plot a confusion matrix on axis ax."""
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

#########################################
# Figure 1: Baseline Loss Curves (Training & Validation)
#########################################
try:
    # Load baseline results from dropout tuning
    baseline_data = np.load(
        "experiment_results/experiment_1b26a93c3d094cee8a32cbed61d33fa7_proc_3160641/experiment_data.npy",
        allow_pickle=True
    ).item()
    dt = baseline_data.get("dropout_tuning", {})
    tags = list(dt.keys())[:5]  # use at most first 5 dropout settings

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for tag in tags:
        epochs = dt[tag]["epochs"]
        # Plot training loss
        axs[0].plot(epochs, np.array(dt[tag]["losses"]["train"]), label=tag)
        # Plot validation loss
        axs[1].plot(epochs, np.array(dt[tag]["losses"]["val"]), label=tag)
    axs[0].set_title("Baseline Training Loss")
    axs[1].set_title("Baseline Validation Loss")
    for ax in axs:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig("figures/baseline_loss_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figures/baseline_loss_curves.png")
except Exception as e:
    print("Error in Figure 1 (Baseline Loss Curves):", e)
    plt.close()

#########################################
# Figure 2: Baseline Test Macro-F1 Bar Plot
#########################################
try:
    # Using same baseline_data as above, compute test F1 per dropout setting
    test_f1s = [dt[tag]["metrics"]["test_f1"] for tag in tags]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(range(len(tags)), test_f1s, tick_label=[str(tag) for tag in tags])
    # If best overall test F1 is present in the baseline data dictionary:
    best_overall = baseline_data.get("best_overall", {})
    best_f1 = best_overall.get("test_f1", None)
    if best_f1 is not None:
        ax.axhline(best_f1, color="red", linestyle="--", label=f"Best Overall = {best_f1:.3f}")
    ax.set_title("Baseline Test Macro-F1 by Dropout")
    ax.set_xlabel("Dropout Setting")
    ax.set_ylabel("Macro-F1")
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig("figures/baseline_test_f1_bar.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figures/baseline_test_f1_bar.png")
except Exception as e:
    print("Error in Figure 2 (Baseline Test F1 Bar):", e)
    plt.close()

#########################################
# Figure 3: Research: Char+Bigram Count – Loss, F1, and Confusion Matrix
#########################################
try:
    research_data = np.load(
        "experiment_results/experiment_4867cf4077184200a6ae09a914d28ac5_proc_3167394/experiment_data.npy",
        allow_pickle=True
    ).item()
    cbc = research_data.get("char_bigram_count", {})
    epochs = np.array(cbc.get("epochs", []))
    train_loss = np.array(cbc.get("losses", {}).get("train", []))
    val_loss = np.array(cbc.get("losses", {}).get("val", []))
    train_f1 = np.array(cbc.get("metrics", {}).get("train_f1", []))
    val_f1 = np.array(cbc.get("metrics", {}).get("val_f1", []))
    preds = np.array(cbc.get("predictions", []))
    gts = np.array(cbc.get("ground_truth", []))

    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    # Subplot 1: Loss curves
    axs[0].plot(epochs, train_loss, label="Train")
    axs[0].plot(epochs, val_loss, label="Validation", color="orange")
    axs[0].set_title("Research Training vs Validation Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    # Subplot 2: F1 curves
    axs[1].plot(epochs, train_f1, label="Train")
    axs[1].plot(epochs, val_f1, label="Validation", color="orange")
    axs[1].set_title("Research Training vs Validation Macro-F1")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Macro-F1")
    axs[1].legend()
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    # Subplot 3: Confusion matrix (if preds exist)
    if preds.size and gts.size:
        num_cls = int(max(np.max(preds), np.max(gts))) + 1
        cm = np.zeros((num_cls, num_cls), dtype=int)
        for t, p in zip(gts, preds):
            cm[int(t), int(p)] += 1
        im = axs[2].imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
        axs[2].set_title("Research Confusion Matrix (Top-{} classes)".format(min(5, num_cls)))
        axs[2].set_xlabel("Predicted")
        axs[2].set_ylabel("True")
        axs[2].spines['right'].set_visible(False)
        axs[2].spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig("figures/research_summary_plots.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figures/research_summary_plots.png")
except Exception as e:
    print("Error in Figure 3 (Research Summary):", e)
    plt.close()

#########################################
# Figure 4: Ablation "No-Count-Vector Pathway" (key: "char_bigram_only"->"SPR_BENCH")
#########################################
try:
    ablation1 = np.load(
        "experiment_results/experiment_cf40ca240f1b42e9888209f021825e62_proc_3174204/experiment_data.npy",
        allow_pickle=True
    ).item()
    # The data is stored under key "char_bigram_only" for SPR_BENCH experiments.
    data1 = ablation1.get("char_bigram_only", {}).get("SPR_BENCH", {})
    epochs = np.array(data1.get("epochs", []))
    tr_loss = np.array(data1.get("losses", {}).get("train", []))
    val_loss = np.array(data1.get("losses", {}).get("val", []))
    tr_f1 = np.array(data1.get("metrics", {}).get("train_f1", []))
    val_f1 = np.array(data1.get("metrics", {}).get("val_f1", []))
    preds = np.array(data1.get("predictions", []))
    gts = np.array(data1.get("ground_truth", []))

    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    axs[0].plot(epochs, tr_loss, label="Train")
    axs[0].plot(epochs, val_loss, label="Validation", color="orange")
    axs[0].set_title("No-Count-Vector Loss")
    axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Loss"); axs[0].legend()
    axs[0].spines['right'].set_visible(False); axs[0].spines['top'].set_visible(False)
    axs[1].plot(epochs, tr_f1, label="Train")
    axs[1].plot(epochs, val_f1, label="Validation", color="orange")
    axs[1].set_title("No-Count-Vector Macro-F1")
    axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Macro-F1"); axs[1].legend()
    axs[1].spines['right'].set_visible(False); axs[1].spines['top'].set_visible(False)
    if preds.size and gts.size:
        num_cls = int(max(np.max(preds), np.max(gts))) + 1
        cm = np.zeros((num_cls, num_cls), dtype=int)
        for t, p in zip(gts, preds):
            cm[int(t), int(p)] += 1
        im = axs[2].imshow(cm, cmap="Blues")
        axs[2].set_title("No-Count-Vector Confusion Matrix")
        axs[2].set_xlabel("Predicted"); axs[2].set_ylabel("True")
        plt.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
        axs[2].spines['right'].set_visible(False); axs[2].spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig("figures/ablation_no_count_vector.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figures/ablation_no_count_vector.png")
except Exception as e:
    print("Error in Figure 4 (Ablation No-Count-Vector):", e)
    plt.close()

#########################################
# Figure 5: Ablation "No-Bigram-Embedding Pathway" (key: "no_bigram_char_count"->"spr_bench")
#########################################
try:
    ablation2 = np.load(
        "experiment_results/experiment_024186e2ce7b4e079d67796dcab573c9_proc_3174205/experiment_data.npy",
        allow_pickle=True
    ).item()
    data2 = ablation2.get("no_bigram_char_count", {}).get("spr_bench", {})
    epochs = np.array(data2.get("epochs", []))
    tr_loss = np.array(data2.get("losses", {}).get("train", []))
    val_loss = np.array(data2.get("losses", {}).get("val", []))
    tr_f1 = np.array(data2.get("metrics", {}).get("train_f1", []))
    val_f1 = np.array(data2.get("metrics", {}).get("val_f1", []))
    preds = np.array(data2.get("predictions", []))
    gts = np.array(data2.get("ground_truth", []))

    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    axs[0].plot(epochs, tr_loss, label="Train")
    axs[0].plot(epochs, val_loss, label="Validation", color="orange")
    axs[0].set_title("No-Bigram-Embedding Loss")
    axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Loss"); axs[0].legend()
    axs[0].spines['right'].set_visible(False); axs[0].spines['top'].set_visible(False)
    axs[1].plot(epochs, tr_f1, label="Train")
    axs[1].plot(epochs, val_f1, label="Validation", color="orange")
    axs[1].set_title("No-Bigram-Embedding Macro-F1")
    axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Macro-F1"); axs[1].legend()
    axs[1].spines['right'].set_visible(False); axs[1].spines['top'].set_visible(False)
    if preds.size and gts.size:
        num_cls = int(max(np.max(preds), np.max(gts))) + 1
        cm = np.zeros((num_cls, num_cls), dtype=int)
        for t, p in zip(gts, preds):
            cm[int(t), int(p)] += 1
        im = axs[2].imshow(cm, cmap="Blues")
        axs[2].set_title("No-Bigram-Embedding Confusion Matrix")
        axs[2].set_xlabel("Predicted"); axs[2].set_ylabel("True")
        plt.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
        axs[2].spines['right'].set_visible(False); axs[2].spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig("figures/ablation_no_bigram_embedding.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figures/ablation_no_bigram_embedding.png")
except Exception as e:
    print("Error in Figure 5 (Ablation No-Bigram-Embedding):", e)
    plt.close()

#########################################
# Figure 6: Ablation "No-Positional-Embedding" (key: "no_positional_embedding"->"spr_bench")
#########################################
try:
    ablation3 = np.load(
        "experiment_results/experiment_bc1e88e6947d44438acfbb28b43b04f3_proc_3174206/experiment_data.npy",
        allow_pickle=True
    ).item()
    data3 = ablation3.get("no_positional_embedding", {}).get("spr_bench", {})
    epochs = np.array(data3.get("epochs", []))
    tr_loss = np.array(data3.get("losses", {}).get("train", []))
    val_loss = np.array(data3.get("losses", {}).get("val", []))
    tr_f1 = np.array(data3.get("metrics", {}).get("train_f1", []))
    val_f1 = np.array(data3.get("metrics", {}).get("val_f1", []))
    preds = np.array(data3.get("predictions", []))
    gts = np.array(data3.get("ground_truth", []))
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    axs[0].plot(epochs, tr_loss, label="Train")
    axs[0].plot(epochs, val_loss, label="Validation", color="orange")
    axs[0].set_title("No-Positional-Embedding Loss")
    axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Loss"); axs[0].legend()
    axs[0].spines['right'].set_visible(False); axs[0].spines['top'].set_visible(False)
    axs[1].plot(epochs, tr_f1, label="Train")
    axs[1].plot(epochs, val_f1, label="Validation", color="orange")
    axs[1].set_title("No-Positional-Embedding Macro-F1")
    axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Macro-F1"); axs[1].legend()
    axs[1].spines['right'].set_visible(False); axs[1].spines['top'].set_visible(False)
    if preds.size and gts.size:
        num_cls = int(max(np.max(preds), np.max(gts))) + 1
        cm = np.zeros((num_cls, num_cls), dtype=int)
        for t, p in zip(gts, preds):
            cm[int(t), int(p)] += 1
        im = axs[2].imshow(cm, cmap="Blues")
        axs[2].set_title("No-Positional-Embedding Confusion Matrix")
        axs[2].set_xlabel("Predicted"); axs[2].set_ylabel("True")
        plt.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
        axs[2].spines['right'].set_visible(False); axs[2].spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig("figures/ablation_no_positional_embedding.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figures/ablation_no_positional_embedding.png")
except Exception as e:
    print("Error in Figure 6 (Ablation No-Positional-Embedding):", e)
    plt.close()

#########################################
# Figure 7: Ablation "No-Char-Embedding" (key: "no_char_bigram_count"->"spr_bench")
#########################################
try:
    ablation4 = np.load(
        "experiment_results/experiment_aa1ca0d0468d468e87a7a61e3f56074e_proc_3174207/experiment_data.npy",
        allow_pickle=True
    ).item()
    data4 = ablation4.get("no_char_bigram_count", {}).get("spr_bench", {})
    epochs = np.array(data4.get("epochs", []))
    tr_loss = np.array(data4.get("losses", {}).get("train", []))
    val_loss = np.array(data4.get("losses", {}).get("val", []))
    tr_f1 = np.array(data4.get("metrics", {}).get("train_f1", []))
    val_f1 = np.array(data4.get("metrics", {}).get("val_f1", []))
    preds = np.array(data4.get("predictions", []))
    gts = np.array(data4.get("ground_truth", []))
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    axs[0].plot(epochs, tr_loss, label="Train")
    axs[0].plot(epochs, val_loss, label="Validation", color="orange")
    axs[0].set_title("No-Char-Embedding Loss")
    axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Loss"); axs[0].legend()
    axs[0].spines['right'].set_visible(False); axs[0].spines['top'].set_visible(False)
    axs[1].plot(epochs, tr_f1, label="Train")
    axs[1].plot(epochs, val_f1, label="Validation", color="orange")
    axs[1].set_title("No-Char-Embedding Macro-F1")
    axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Macro-F1"); axs[1].legend()
    axs[1].spines['right'].set_visible(False); axs[1].spines['top'].set_visible(False)
    if preds.size and gts.size:
        num_cls = int(max(np.max(preds), np.max(gts))) + 1
        cm = np.zeros((num_cls, num_cls), dtype=int)
        for t, p in zip(gts, preds):
            cm[int(t), int(p)] += 1
        im = axs[2].imshow(cm, cmap="Blues")
        axs[2].set_title("No-Char-Embedding Confusion Matrix")
        axs[2].set_xlabel("Predicted"); axs[2].set_ylabel("True")
        plt.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
        axs[2].spines['right'].set_visible(False); axs[2].spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig("figures/ablation_no_char_embedding.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figures/ablation_no_char_embedding.png")
except Exception as e:
    print("Error in Figure 7 (Ablation No-Char-Embedding):", e)
    plt.close()

#########################################
# Figure 8: Ablation "Count-Only (Bag-of-Characters)" 
# Create a two-panel figure: (a) Loss and F1 curves; (b) Best Val vs Test F1 Bar and Confusion matrix (side-by-side)
#########################################
try:
    ablation5 = np.load(
        "experiment_results/experiment_d10b2936a8694b11a272f55e0bb1a677_proc_3174204/experiment_data.npy",
        allow_pickle=True
    ).item()
    data5 = ablation5.get("count_only", {}).get("spr_bench", {})
    epochs = np.array(data5.get("epochs", []))
    train_loss = np.array(data5.get("losses", {}).get("train", []))
    val_loss = np.array(data5.get("losses", {}).get("val", []))
    train_f1 = np.array(data5.get("metrics", {}).get("train_f1", []))
    val_f1 = np.array(data5.get("metrics", {}).get("val_f1", []))
    test_f1 = data5.get("metrics", {}).get("test_f1", None)
    preds = np.array(data5.get("predictions", []))
    gts = np.array(data5.get("ground_truth", []))
    
    # Panel (a): Loss and F1 curves in one figure (2 subplots)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(epochs, train_loss, label="Train")
    axs[0].plot(epochs, val_loss, label="Validation", color="orange")
    axs[0].set_title("Count-Only Loss")
    axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Loss"); axs[0].legend()
    axs[0].spines['right'].set_visible(False); axs[0].spines['top'].set_visible(False)
    axs[1].plot(epochs, train_f1, label="Train")
    axs[1].plot(epochs, val_f1, label="Validation", color="orange")
    axs[1].set_title("Count-Only Macro-F1")
    axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Macro-F1"); axs[1].legend()
    axs[1].spines['right'].set_visible(False); axs[1].spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig("figures/ablation_count_only_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Panel (b): Bar chart of Best Validation vs Test and Confusion matrix side-by-side
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    if len(val_f1) > 0 and test_f1 is not None:
        best_val = np.max(val_f1)
        axs[0].bar(["Best Val", "Test"], [best_val, test_f1],
                   color=["skyblue", "salmon"])
        for i, v in enumerate([best_val, test_f1]):
            axs[0].text(i, v + 0.01, f"{v:.3f}", ha="center")
        axs[0].set_ylim(0, 1)
        axs[0].set_title("Count-Only: Best Val vs Test Macro-F1")
        axs[0].set_ylabel("Macro-F1")
    if preds.size and gts.size:
        num_cls = int(max(np.max(preds), np.max(gts))) + 1
        cm = np.zeros((num_cls, num_cls), dtype=int)
        for t, p in zip(gts, preds):
            cm[int(t), int(p)] += 1
        im = axs[1].imshow(cm, cmap="Blues")
        axs[1].set_title("Count-Only Confusion Matrix")
        axs[1].set_xlabel("Predicted")
        axs[1].set_ylabel("True")
        plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig("figures/ablation_count_only_bar_conf.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figures/ablation_count_only (curves and bar/confusion) plots")
except Exception as e:
    print("Error in Figure 8 (Ablation Count-Only):", e)
    plt.close()

#########################################
# Figure 9: Ablation "Bigram-Only Transformer" (key: "bigram_only")
#########################################
try:
    ablation6 = np.load(
        "experiment_results/experiment_d3120e09a3cb4173be885a68aca9569e_proc_3174205/experiment_data.npy",
        allow_pickle=True
    ).item()
    data6 = ablation6.get("bigram_only", {})
    epochs = np.array(data6.get("epochs", []))
    tr_loss = np.array(data6.get("losses", {}).get("train", []))
    val_loss = np.array(data6.get("losses", {}).get("val", []))
    tr_f1 = np.array(data6.get("metrics", {}).get("train_f1", []))
    val_f1 = np.array(data6.get("metrics", {}).get("val_f1", []))
    preds = np.array(data6.get("predictions", []))
    gts = np.array(data6.get("ground_truth", []))
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    axs[0].plot(epochs, tr_loss, label="Train")
    axs[0].plot(epochs, val_loss, label="Validation", color="orange")
    axs[0].set_title("Bigram-Only Loss")
    axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Loss"); axs[0].legend()
    axs[0].spines['right'].set_visible(False); axs[0].spines['top'].set_visible(False)
    axs[1].plot(epochs, tr_f1, label="Train")
    axs[1].plot(epochs, val_f1, label="Validation", color="orange")
    axs[1].set_title("Bigram-Only Macro-F1")
    axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Macro-F1"); axs[1].legend()
    axs[1].spines['right'].set_visible(False); axs[1].spines['top'].set_visible(False)
    if preds.size and gts.size:
        num_cls = int(max(np.max(preds), np.max(gts))) + 1
        cm = np.zeros((num_cls, num_cls), dtype=int)
        for t, p in zip(gts, preds):
            cm[int(t), int(p)] += 1
        im = axs[2].imshow(cm, cmap="Blues")
        axs[2].set_title("Bigram-Only Confusion Matrix")
        axs[2].set_xlabel("Predicted"); axs[2].set_ylabel("True")
        plt.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
        axs[2].spines['right'].set_visible(False); axs[2].spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig("figures/ablation_bigram_only.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figures/ablation_bigram_only.png")
except Exception as e:
    print("Error in Figure 9 (Ablation Bigram-Only):", e)
    plt.close()

#########################################
# Figure 10: Ablation "No-Transformer-Context" (key: "no_transformer_context"->"spr_bench")
# Arrange in 2 rows and 2 columns: Loss & F1 curves; and Bar (Best Val vs Test) & Confusion Matrix.
#########################################
try:
    ablation7 = np.load(
        "experiment_results/experiment_ba1e3fd2501b472596f7245a82b88894_proc_3174206/experiment_data.npy",
        allow_pickle=True
    ).item()
    data7 = ablation7.get("no_transformer_context", {}).get("spr_bench", {})
    epochs = np.array(data7.get("epochs", []))
    tr_loss = np.array(data7.get("losses", {}).get("train", []))
    val_loss = np.array(data7.get("losses", {}).get("val", []))
    tr_f1 = np.array(data7.get("metrics", {}).get("train_f1", []))
    val_f1 = np.array(data7.get("metrics", {}).get("val_f1", []))
    test_f1 = data7.get("metrics", {}).get("test_f1", None)
    preds = np.array(data7.get("predictions", []))
    gts = np.array(data7.get("ground_truth", []))
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    # Top-left: Loss curves
    axs[0,0].plot(epochs, tr_loss, label="Train")
    axs[0,0].plot(epochs, val_loss, label="Validation", color="orange")
    axs[0,0].set_title("No-Transformer-Context Loss")
    axs[0,0].set_xlabel("Epoch"); axs[0,0].set_ylabel("Loss"); axs[0,0].legend()
    axs[0,0].spines['right'].set_visible(False); axs[0,0].spines['top'].set_visible(False)
    # Top-right: F1 curves
    axs[0,1].plot(epochs, tr_f1, label="Train")
    axs[0,1].plot(epochs, val_f1, label="Validation", color="orange")
    axs[0,1].set_title("No-Transformer-Context Macro-F1")
    axs[0,1].set_xlabel("Epoch"); axs[0,1].set_ylabel("Macro-F1"); axs[0,1].legend()
    axs[0,1].spines['right'].set_visible(False); axs[0,1].spines['top'].set_visible(False)
    # Bottom-left: Bar chart: Best Val vs Test
    if len(val_f1) > 0 and test_f1 is not None:
        best_val = np.max(val_f1)
        axs[1,0].bar(["Best Val", "Test"], [best_val, test_f1], color=["steelblue", "orange"])
        for i, v in enumerate([best_val, test_f1]):
            axs[1,0].text(i, v + 0.01, f"{v:.3f}", ha="center")
        axs[1,0].set_ylim(0, 1)
        axs[1,0].set_title("No-Transformer-Context: Val vs Test F1")
        axs[1,0].set_ylabel("Macro-F1")
    # Bottom-right: Confusion Matrix
    if preds.size and gts.size:
        cm = confusion_matrix(gts, preds)
        im = axs[1,1].imshow(cm, cmap="Blues")
        axs[1,1].set_title("No-Transformer-Context Confusion Matrix")
        axs[1,1].set_xlabel("Predicted"); axs[1,1].set_ylabel("True")
        plt.colorbar(im, ax=axs[1,1], fraction=0.046, pad=0.04)
        axs[1,1].spines['right'].set_visible(False); axs[1,1].spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig("figures/ablation_no_transformer_context.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figures/ablation_no_transformer_context.png")
except Exception as e:
    print("Error in Figure 10 (Ablation No-Transformer-Context):", e)
    plt.close()

#########################################
# Figure 11: Ablation "CBC-CLS (CLS-Token Pooling)" (key: "cls_pooling"->"spr_bench")
#########################################
try:
    ablation8 = np.load(
        "experiment_results/experiment_e580495aa632498eb543123e4a659111_proc_3174207/experiment_data.npy",
        allow_pickle=True
    ).item()
    data8 = ablation8.get("cls_pooling", {}).get("spr_bench", {})
    epochs = data8.get("epochs", [])
    tr_loss = data8.get("losses", {}).get("train", [])
    val_loss = data8.get("losses", {}).get("val", [])
    tr_f1 = data8.get("metrics", {}).get("train_f1", [])
    val_f1 = data8.get("metrics", {}).get("val_f1", [])
    test_f1 = data8.get("metrics", {}).get("test_f1", None)
    preds = data8.get("predictions", [])
    gts = data8.get("ground_truth", [])
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    axs[0].plot(epochs, tr_loss, label="Train")
    axs[0].plot(epochs, val_loss, label="Validation", color="orange")
    axs[0].set_title("CBC-CLS Loss")
    axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Loss"); axs[0].legend()
    axs[0].spines['right'].set_visible(False); axs[0].spines['top'].set_visible(False)
    axs[1].plot(epochs, tr_f1, label="Train")
    axs[1].plot(epochs, val_f1, label="Validation", color="orange")
    if test_f1 is not None:
        axs[1].hlines(test_f1, epochs[0], epochs[-1], colors="red", linestyles="dashed", label=f"Test={test_f1:.3f}")
    axs[1].set_title("CBC-CLS Macro-F1")
    axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Macro-F1"); axs[1].legend()
    axs[1].spines['right'].set_visible(False); axs[1].spines['top'].set_visible(False)
    if preds and gts:
        cm = confusion_matrix(gts, preds, normalize="true")
        im = axs[2].imshow(cm, cmap="Blues")
        axs[2].set_title("CBC-CLS Normalized Confusion Matrix")
        axs[2].set_xlabel("Predicted"); axs[2].set_ylabel("True")
        plt.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
        axs[2].spines['right'].set_visible(False); axs[2].spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig("figures/ablation_cbc_cls.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figures/ablation_cbc_cls.png")
except Exception as e:
    print("Error in Figure 11 (Ablation CBC-CLS):", e)
    plt.close()

print("All plots generated.")
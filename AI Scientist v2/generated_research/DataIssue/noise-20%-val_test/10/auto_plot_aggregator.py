#!/usr/bin/env python3
"""
Aggregate final experiment figures for the paper:
"Interpretable Neural Rule Learning for Synthetic PolyRule Reasoning"

This script loads the final experiment .npy files from Baseline, Research, and Ablation studies,
and generates a set of unique, publication-quality figures (no more than 12 in total) saved in the
"figures/" directory. Each figure aggregates related plots where appropriate. All axes, titles, and legends
use descriptive text with no underscores.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Use a style that is reliably available
plt.style.use('classic')
FONT_LABEL = 14
FONT_TITLE = 16
FONT_LEGEND = 12

def beautify_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=FONT_LABEL)

def macro_f1(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []
    for lb in labels:
        tp = np.sum((y_true == lb) & (y_pred == lb))
        fp = np.sum((y_true != lb) & (y_pred == lb))
        fn = np.sum((y_true == lb) & (y_pred != lb))
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 0.0 if (prec+rec)==0 else 2 * prec * rec / (prec+rec)
        f1s.append(f1)
    return np.mean(f1s)

os.makedirs("figures", exist_ok=True)

# Full file paths from summary JSON
baseline_path   = "experiment_results/experiment_f9117b0771a84de4b3e64104db87c556_proc_3301990/experiment_data.npy"
research_path   = "experiment_results/experiment_77518c893aff4bd4ad649b2e6558fda5_proc_3305857/experiment_data.npy"
remove_l1_path  = "experiment_results/experiment_b9b1ef646aeb44cb9c9919dad83c488f_proc_3309945/experiment_data.npy"
remove_boc_path = "experiment_results/experiment_0527ddea95e446cb8b8229f1b79afb8b_proc_3309946/experiment_data.npy"
freeze_emb_path = "experiment_results/experiment_482156c03db54f31bb22dc10798e957d_proc_3309948/experiment_data.npy"

# Load experiment data dictionaries
try:
    baseline_data = np.load(baseline_path, allow_pickle=True).item()
except Exception:
    baseline_data = {}
try:
    research_data = np.load(research_path, allow_pickle=True).item()
except Exception:
    research_data = {}
try:
    remove_l1_data = np.load(remove_l1_path, allow_pickle=True).item()
except Exception:
    remove_l1_data = {}
try:
    remove_boc_data = np.load(remove_boc_path, allow_pickle=True).item()
except Exception:
    remove_boc_data = {}
try:
    freeze_emb_data = np.load(freeze_emb_path, allow_pickle=True).item()
except Exception:
    freeze_emb_data = {}

# ---------------------- Figure 1 ----------------------
# Baseline: Validation Macro F1 versus Epoch (all learning rates)
try:
    lr_exp = baseline_data.get("learning_rate", {}).get("SPR_BENCH", {})
    runs = lr_exp.get("runs", {})
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    for lr, run in runs.items():
        # Accept both "val f1" and "val_f1"
        val_f1 = run.get("metrics", {}).get("val f1", []) or run.get("metrics", {}).get("val_f1", [])
        epochs = range(1, len(val_f1) + 1)
        ax.plot(epochs, val_f1, label=f"Learning Rate = {lr}", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=FONT_LABEL)
    ax.set_ylabel("Validation Macro F1", fontsize=FONT_LABEL)
    ax.set_title("Baseline: Validation Macro F1 vs Epoch (All Rates)", fontsize=FONT_TITLE)
    ax.legend(fontsize=FONT_LEGEND)
    beautify_ax(ax)
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Baseline Validation Macro F1 vs Epoch.png"))
    plt.close()
except Exception as e:
    plt.close()

# ---------------------- Figure 2 ----------------------
# Baseline: Train and Validation Loss (Best Learning Rate)
try:
    best_lr = lr_exp.get("best_lr", None)
    if best_lr is not None:
        run = runs.get(str(best_lr), {})
        tr_loss = run.get("losses", {}).get("train", [])
        val_loss = run.get("losses", {}).get("val", [])
        epochs = range(1, len(tr_loss) + 1)
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        ax.plot(epochs, tr_loss, label="Train Loss", linewidth=2)
        ax.plot(epochs, val_loss, label="Validation Loss", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=FONT_LABEL)
        ax.set_ylabel("Cross Entropy Loss", fontsize=FONT_LABEL)
        ax.set_title(f"Baseline: Loss Curves (Best Rate = {best_lr})", fontsize=FONT_TITLE)
        ax.legend(fontsize=FONT_LEGEND)
        beautify_ax(ax)
        plt.tight_layout()
        plt.savefig(os.path.join("figures", f"Baseline Loss Curves (Best Rate = {best_lr}).png"))
        plt.close()
except Exception as e:
    plt.close()

# ---------------------- Figure 3 ----------------------
# Baseline: Confusion Matrix (Test Set)
try:
    preds = np.array(lr_exp.get("predictions", []))
    gts   = np.array(lr_exp.get("ground_truth", []))
    if preds.size and gts.size:
        labels = np.unique(np.concatenate([gts, preds]))
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(gts, preds):
            cm[int(t), int(p)] += 1
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("Predicted", fontsize=FONT_LABEL)
        ax.set_ylabel("True", fontsize=FONT_LABEL)
        ax.set_title("Baseline: Confusion Matrix (Test Set)", fontsize=FONT_TITLE)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10)
        beautify_ax(ax)
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "Baseline Confusion Matrix (Test Set).png"))
        plt.close()
except Exception as e:
    plt.close()

# ---------------------- Figure 4 ----------------------
# Research: Combined Loss and Macro F1 Curves in one figure (2 subplots)
try:
    tr_loss = research_data.get("losses", {}).get("train", [])
    val_loss = research_data.get("losses", {}).get("val", [])
    tr_f1 = research_data.get("metrics", {}).get("train_f1", [])
    val_f1 = research_data.get("metrics", {}).get("val_f1", [])
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
    epochs_loss = range(1, len(tr_loss)+1)
    epochs_f1 = range(1, len(tr_f1)+1)
    axs[0].plot(epochs_loss, tr_loss, label="Train Loss", linewidth=2)
    axs[0].plot(epochs_loss, val_loss, label="Validation Loss", linewidth=2)
    axs[0].set_xlabel("Epoch", fontsize=FONT_LABEL)
    axs[0].set_ylabel("Loss", fontsize=FONT_LABEL)
    axs[0].set_title("Research: Loss Curves", fontsize=FONT_TITLE)
    axs[0].legend(fontsize=FONT_LEGEND)
    beautify_ax(axs[0])
    axs[1].plot(epochs_f1, tr_f1, label="Train Macro F1", linewidth=2)
    axs[1].plot(epochs_f1, val_f1, label="Validation Macro F1", linewidth=2)
    axs[1].set_xlabel("Epoch", fontsize=FONT_LABEL)
    axs[1].set_ylabel("Macro F1", fontsize=FONT_LABEL)
    axs[1].set_title("Research: Macro F1 Curves", fontsize=FONT_TITLE)
    axs[1].legend(fontsize=FONT_LEGEND)
    beautify_ax(axs[1])
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Research Loss and Macro F1 Curves.png"))
    plt.close()
except Exception as e:
    plt.close()

# ---------------------- Figure 5 ----------------------
# Research: Confusion Matrix (Test Set)
try:
    preds = np.array(research_data.get("predictions", []))
    gts   = np.array(research_data.get("ground_truth", []))
    if preds.size and gts.size:
        labels = np.unique(np.concatenate([gts, preds]))
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(gts, preds):
            cm[int(t), int(p)] += 1
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("Predicted", fontsize=FONT_LABEL)
        ax.set_ylabel("True", fontsize=FONT_LABEL)
        ax.set_title("Research: Confusion Matrix (Test Set)", fontsize=FONT_TITLE)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10)
        beautify_ax(ax)
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "Research Confusion Matrix (Test Set).png"))
        plt.close()
except Exception as e:
    plt.close()

# ---------------------- Figure 6 ----------------------
# Ablation (Remove L1): Combined Loss and Macro F1 Curves
try:
    rl1_tr_loss = remove_l1_data.get("losses", {}).get("train", [])
    rl1_val_loss = remove_l1_data.get("losses", {}).get("val", [])
    rl1_tr_f1 = remove_l1_data.get("metrics", {}).get("train_f1", [])
    rl1_val_f1 = remove_l1_data.get("metrics", {}).get("val_f1", [])
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
    epochs_loss = range(1, len(rl1_tr_loss)+1)
    epochs_f1 = range(1, len(rl1_tr_f1)+1)
    axs[0].plot(epochs_loss, rl1_tr_loss, label="Train Loss", linewidth=2)
    axs[0].plot(epochs_loss, rl1_val_loss, label="Validation Loss", linewidth=2)
    axs[0].set_xlabel("Epoch", fontsize=FONT_LABEL)
    axs[0].set_ylabel("Loss", fontsize=FONT_LABEL)
    axs[0].set_title("Ablation Remove L1: Loss Curves", fontsize=FONT_TITLE)
    axs[0].legend(fontsize=FONT_LEGEND)
    beautify_ax(axs[0])
    axs[1].plot(epochs_f1, rl1_tr_f1, label="Train Macro F1", linewidth=2)
    axs[1].plot(epochs_f1, rl1_val_f1, label="Validation Macro F1", linewidth=2)
    axs[1].set_xlabel("Epoch", fontsize=FONT_LABEL)
    axs[1].set_ylabel("Macro F1", fontsize=FONT_LABEL)
    axs[1].set_title("Ablation Remove L1: Macro F1 Curves", fontsize=FONT_TITLE)
    axs[1].legend(fontsize=FONT_LEGEND)
    beautify_ax(axs[1])
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Ablation Remove L1 Loss and Macro F1 Curves.png"))
    plt.close()
except Exception as e:
    plt.close()

# ---------------------- Figure 7 ----------------------
# Ablation (Remove L1): Bar Chart of REA Versus Hybrid Test Macro F1
try:
    rea_dev = remove_l1_data.get("metrics", {}).get("REA_dev", None)
    rea_test = remove_l1_data.get("metrics", {}).get("REA_test", None)
    hybrid_f1_list = remove_l1_data.get("metrics", {}).get("val_f1", [])
    hybrid_f1 = hybrid_f1_list[-1] if hybrid_f1_list else None
    if (rea_dev is not None) and (rea_test is not None) and (hybrid_f1 is not None):
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        bars = [rea_dev, rea_test, hybrid_f1]
        labels = ["REA Dev", "REA Test", "Hybrid Test Macro F1"]
        ax.bar(labels, bars, color=["skyblue", "lightgreen", "salmon"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy / F1", fontsize=FONT_LABEL)
        ax.set_title("Ablation Remove L1: REA vs Hybrid Test Macro F1", fontsize=FONT_TITLE)
        for i, v in enumerate(bars):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=FONT_LEGEND)
        beautify_ax(ax)
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "Ablation Remove L1 REA vs Hybrid Test Macro F1.png"))
        plt.close()
except Exception as e:
    plt.close()

# ---------------------- Figure 8 ----------------------
# Ablation (Remove BoC): Combined Loss and Macro F1 Curves
try:
    rb_tr_loss = remove_boc_data.get("losses", {}).get("train", [])
    rb_val_loss = remove_boc_data.get("losses", {}).get("val", [])
    rb_tr_f1 = remove_boc_data.get("metrics", {}).get("train_f1", [])
    rb_val_f1 = remove_boc_data.get("metrics", {}).get("val_f1", [])
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
    epochs_loss = range(1, len(rb_tr_loss)+1)
    epochs_f1 = range(1, len(rb_tr_f1)+1)
    axs[0].plot(epochs_loss, rb_tr_loss, label="Train Loss", linewidth=2)
    axs[0].plot(epochs_loss, rb_val_loss, label="Validation Loss", linewidth=2)
    axs[0].set_xlabel("Epoch", fontsize=FONT_LABEL)
    axs[0].set_ylabel("Loss", fontsize=FONT_LABEL)
    axs[0].set_title("Ablation Remove BoC: Loss Curves", fontsize=FONT_TITLE)
    axs[0].legend(fontsize=FONT_LEGEND)
    beautify_ax(axs[0])
    axs[1].plot(epochs_f1, rb_tr_f1, label="Train Macro F1", linewidth=2)
    axs[1].plot(epochs_f1, rb_val_f1, label="Validation Macro F1", linewidth=2)
    axs[1].set_xlabel("Epoch", fontsize=FONT_LABEL)
    axs[1].set_ylabel("Macro F1", fontsize=FONT_LABEL)
    axs[1].set_title("Ablation Remove BoC: Macro F1 Curves", fontsize=FONT_TITLE)
    axs[1].legend(fontsize=FONT_LEGEND)
    beautify_ax(axs[1])
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Ablation Remove BoC Loss and Macro F1 Curves.png"))
    plt.close()
except Exception as e:
    plt.close()

# ---------------------- Figure 9 ----------------------
# Ablation (Freeze Character Embeddings): Aggregate Plot (Loss, Macro F1, Confusion Matrix)
try:
    fe_tr_loss = freeze_emb_data.get("losses", {}).get("train", [])
    fe_val_loss = freeze_emb_data.get("losses", {}).get("val", [])
    fe_tr_f1 = freeze_emb_data.get("metrics", {}).get("train_f1", [])
    fe_val_f1 = freeze_emb_data.get("metrics", {}).get("val_f1", [])
    fe_preds = np.array(freeze_emb_data.get("preds_test", []))
    fe_gts   = np.array(freeze_emb_data.get("gts_test", []))
    fig, axs = plt.subplots(1, 3, figsize=(18, 4), dpi=300)
    epochs_loss = range(1, len(fe_tr_loss)+1)
    epochs_f1 = range(1, len(fe_tr_f1)+1)
    # Subplot 1: Loss curves
    axs[0].plot(epochs_loss, fe_tr_loss, label="Train Loss", linewidth=2)
    axs[0].plot(epochs_loss, fe_val_loss, label="Validation Loss", linewidth=2)
    axs[0].set_xlabel("Epoch", fontsize=FONT_LABEL)
    axs[0].set_ylabel("Loss", fontsize=FONT_LABEL)
    axs[0].set_title("Freeze Embeddings: Loss Curves", fontsize=FONT_TITLE)
    axs[0].legend(fontsize=FONT_LEGEND)
    beautify_ax(axs[0])
    # Subplot 2: Macro F1 curves
    axs[1].plot(epochs_f1, fe_tr_f1, label="Train Macro F1", linewidth=2)
    axs[1].plot(epochs_f1, fe_val_f1, label="Validation Macro F1", linewidth=2)
    axs[1].set_xlabel("Epoch", fontsize=FONT_LABEL)
    axs[1].set_ylabel("Macro F1", fontsize=FONT_LABEL)
    axs[1].set_title("Freeze Embeddings: Macro F1 Curves", fontsize=FONT_TITLE)
    axs[1].legend(fontsize=FONT_LEGEND)
    beautify_ax(axs[1])
    # Subplot 3: Confusion Matrix (if available)
    if fe_preds.size and fe_gts.size:
        labels = np.unique(np.concatenate([fe_gts, fe_preds]))
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(fe_gts, fe_preds):
            cm[int(t), int(p)] += 1
        im = axs[2].imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=axs[2])
        axs[2].set_xlabel("Predicted", fontsize=FONT_LABEL)
        axs[2].set_ylabel("True", fontsize=FONT_LABEL)
        axs[2].set_title("Freeze Embeddings: Confusion Matrix", fontsize=FONT_TITLE)
        axs[2].set_xticks(range(len(labels)))
        axs[2].set_yticks(range(len(labels)))
        for i in range(len(labels)):
            for j in range(len(labels)):
                axs[2].text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10)
        beautify_ax(axs[2])
    else:
        axs[2].text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=FONT_LABEL)
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Ablation Freeze Embeddings Aggregate.png"))
    plt.close()
except Exception as e:
    plt.close()

# ---------------------- Figure 10 ----------------------
# Overall Test Macro F1 Comparison Across Experiments
try:
    results = {}
    # Baseline
    bl_preds = np.array(lr_exp.get("predictions", []))
    bl_gts   = np.array(lr_exp.get("ground_truth", []))
    if bl_preds.size and bl_gts.size:
        results["Baseline"] = macro_f1(bl_gts, bl_preds)
    # Research
    rs_preds = np.array(research_data.get("predictions", []))
    rs_gts   = np.array(research_data.get("ground_truth", []))
    if rs_preds.size and rs_gts.size:
        results["Research"] = macro_f1(rs_gts, rs_preds)
    # Remove L1
    rl1_preds = np.array(remove_l1_data.get("predictions", []))
    rl1_gts   = np.array(remove_l1_data.get("ground_truth", []))
    if rl1_preds.size and rl1_gts.size:
        results["Remove L1"] = macro_f1(rl1_gts, rl1_preds)
    # Remove BoC
    rb_preds = np.array(remove_boc_data.get("predictions", []))
    rb_gts   = np.array(remove_boc_data.get("ground_truth", []))
    if rb_preds.size and rb_gts.size:
        results["Remove BoC"] = macro_f1(rb_gts, rb_preds)
    # Freeze Embeddings
    fe_preds = np.array(freeze_emb_data.get("preds_test", []))
    fe_gts   = np.array(freeze_emb_data.get("gts_test", []))
    if fe_preds.size and fe_gts.size:
        results["Freeze Embeddings"] = macro_f1(fe_gts, fe_preds)
    if results:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        categories = list(results.keys())
        scores = [results[k] for k in categories]
        ax.bar(categories, scores, color=["skyblue", "lightgreen", "salmon", "orchid", "gold"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Test Macro F1", fontsize=FONT_LABEL)
        ax.set_title("Overall Test Macro F1 Comparison", fontsize=FONT_TITLE)
        for i, v in enumerate(scores):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=FONT_LEGEND)
        beautify_ax(ax)
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "Overall Test Macro F1 Comparison.png"))
        plt.close()
except Exception as e:
    plt.close()
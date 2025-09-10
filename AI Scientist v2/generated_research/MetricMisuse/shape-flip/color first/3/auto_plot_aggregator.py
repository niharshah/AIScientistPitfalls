#!/usr/bin/env python3
"""
Aggregator script for final research paper figures.
Loads experiment_data.npy files from baseline, research and ablation runs,
aggregates key plots and saves them into the "figures/" directory.
Each plot is wrapped in its own try-except block so that one failure
does not block the rest of the plotting.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Set global plotting parameters for a professional look.
plt.rcParams.update({'font.size': 14})
plt.rcParams['figure.dpi'] = 300

def style_ax(ax):
    """Remove top/right spines and add a light grid."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3)

# Ensure figures folder exists.
os.makedirs("figures", exist_ok=True)

###############################################
# 1. BASELINE PLOTS (from GCN hyperparam tuning)
###############################################
try:
    baseline_file = "experiment_results/experiment_b396019dcf9b4785902e93538f226955_proc_1445250/experiment_data.npy"
    baseline_data = np.load(baseline_file, allow_pickle=True).item()
    runs = baseline_data.get("num_epochs", {})
    run_keys = list(runs.keys())
    if not run_keys:
        print("No run keys found in baseline data.")
except Exception as e:
    print(f"Error loading baseline data: {e}")
    runs = {}

# (A) Plot BWA learning curves for runs (split into parts if >3)
try:
    # split run keys into groups of up to 3
    groups = [run_keys[i:i+3] for i in range(0, len(run_keys), 3)]
    for idx, group in enumerate(groups):
        fig, axs = plt.subplots(1, len(group), figsize=(6 * len(group), 5))
        if len(group) == 1:
            axs = [axs]
        for ax, rk in zip(axs, group):
            run_data = runs[rk]
            # Assume run_data["metrics"]["train"] and run_data["metrics"]["val"] are lists of numbers
            epochs = np.arange(1, len(run_data["metrics"]["train"]) + 1)
            train_bwa = run_data["metrics"]["train"]
            val_bwa = run_data["metrics"]["val"]
            ax.plot(epochs, train_bwa, marker='o', label="Train BWA")
            ax.plot(epochs, val_bwa, marker='s', label="Val BWA")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("BWA")
            ax.set_title(f"Run {rk}")
            ax.legend()
            style_ax(ax)
        fig.suptitle("Baseline: BWA Learning Curves")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fname = os.path.join("figures", f"baseline_bwa_curves_part{idx+1}.png")
        fig.savefig(fname)
        plt.close(fig)
        print(f"Saved {fname}")
except Exception as e:
    print(f"Error plotting baseline BWA curves: {e}")

# (B) Bar chart comparing test BWA across runs
try:
    run_names = []
    test_bwa = []
    for rk, run_data in runs.items():
        tm = run_data.get("test_metrics", {})
        if "BWA" in tm:
            run_names.append(rk)
            test_bwa.append(tm["BWA"])
    if run_names:
        fig, ax = plt.subplots(figsize=(7, 5))
        x = np.arange(len(run_names))
        ax.bar(x, test_bwa, color="skyblue")
        ax.set_xticks(x)
        ax.set_xticklabels(run_names, rotation=45, ha="right")
        ax.set_ylabel("Test BWA")
        ax.set_title("Baseline: Test BWA Comparison")
        style_ax(ax)
        fig.tight_layout()
        fname = os.path.join("figures", "baseline_test_bwa_comparison.png")
        fig.savefig(fname)
        plt.close(fig)
        print(f"Saved {fname}")
    else:
        print("No test BWA values available in baseline.")
except Exception as e:
    print(f"Error plotting baseline test BWA comparison: {e}")

# (C) Confusion matrix for best run (highest Test BWA)
try:
    if runs:
        best_run_key = max(runs.items(), key=lambda item: item[1].get("test_metrics", {}).get("BWA", -np.inf))[0]
        best_run = runs[best_run_key]
        preds = np.array(best_run["predictions"])
        gts = np.array(best_run["ground_truth"])
        num_classes = int(max(preds.max(), gts.max()) + 1)
        conf_mat = np.zeros((num_classes, num_classes), dtype=int)
        for gt, pr in zip(gts, preds):
            conf_mat[gt, pr] += 1

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(conf_mat, cmap="Blues")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"Baseline: Confusion Matrix (Best Run {best_run_key})")
        style_ax(ax)
        # Annotate cells
        for (i, j), v in np.ndenumerate(conf_mat):
            ax.text(j, i, str(v), ha="center", va="center", color="black", fontsize=10)
        fig.tight_layout()
        fname = os.path.join("figures", f"baseline_confusion_matrix_{best_run_key}.png")
        fig.savefig(fname)
        plt.close(fig)
        print(f"Saved {fname}")
    else:
        print("No run data for confusion matrix in baseline.")
except Exception as e:
    print(f"Error plotting baseline confusion matrix: {e}")

###############################################
# 2. RESEARCH PLOTS (RGCN multi-relational model)
###############################################
try:
    research_file = "experiment_results/experiment_7172764bf12543dd9eee225dc58c4a2d_proc_1458730/experiment_data.npy"
    research_data = np.load(research_file, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading research data: {e}")
    research_data = {}

# (A) Combined Loss and BWA curves in one figure (side by side)
try:
    # Expect research_data keys: "losses", "metrics", where losses contains "train" and "val"
    losses = research_data.get("losses", {})
    metrics = research_data.get("metrics", {})
    if losses and metrics:
        epochs = np.arange(1, len(losses.get("train", [])) + 1)
        train_loss = losses.get("train", [])
        val_loss = losses.get("val", [])
        train_bwa = [m["BWA"] for m in metrics.get("train", [])]
        val_bwa = [m["BWA"] for m in metrics.get("val", [])]

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].plot(epochs, train_loss, marker='o', label="Train Loss")
        axs[0].plot(epochs, val_loss, marker='s', label="Val Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Cross-Entropy Loss")
        axs[0].set_title("Research: Loss Curve")
        axs[0].legend()
        style_ax(axs[0])

        axs[1].plot(epochs, train_bwa, marker='o', label="Train BWA")
        axs[1].plot(epochs, val_bwa, marker='s', label="Val BWA")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("BWA")
        axs[1].set_title("Research: BWA Curve")
        axs[1].legend()
        style_ax(axs[1])

        fig.suptitle("Research Experiment: Loss and BWA Curves")
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fname = os.path.join("figures", "research_loss_bwa.png")
        fig.savefig(fname)
        plt.close(fig)
        print(f"Saved {fname}")
    else:
        print("Research data missing loss or metric info.")
except Exception as e:
    print(f"Error plotting research loss/BWA curves: {e}")

# (B) Research Test Metrics Bar Chart
try:
    tm = research_data.get("test_metrics", {})
    metrics_labels = ["BWA", "CWA", "SWA", "StrWA"]
    values = [tm.get(m, np.nan) for m in metrics_labels]
    if any(values):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(metrics_labels, values, color="salmon")
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
        ax.set_ylabel("Score")
        ax.set_title("Research: Test Metrics")
        style_ax(ax)
        fig.tight_layout()
        fname = os.path.join("figures", "research_test_metrics.png")
        fig.savefig(fname)
        plt.close(fig)
        print(f"Saved {fname}")
    else:
        print("No test metrics found in research data.")
except Exception as e:
    print(f"Error plotting research test metrics: {e}")

# (C) Research Confusion Matrix
try:
    preds = np.array(research_data.get("predictions", []))
    gts = np.array(research_data.get("ground_truth", []))
    if preds.size and gts.size:
        num_classes = int(max(preds.max(), gts.max()) + 1)
        conf = np.zeros((num_classes, num_classes), dtype=int)
        for gt, pr in zip(gts, preds):
            conf[gt, pr] += 1
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(conf, cmap="Blues")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Research: Confusion Matrix")
        style_ax(ax)
        for (i, j), v in np.ndenumerate(conf):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=10)
        fig.tight_layout()
        fname = os.path.join("figures", "research_confusion_matrix.png")
        fig.savefig(fname)
        plt.close(fig)
        print(f"Saved {fname}")
    else:
        print("No predictions/ground_truth in research data for confusion matrix.")
except Exception as e:
    print(f"Error plotting research confusion matrix: {e}")

###############################################
# 3. ABLATION PLOTS
###############################################
# (A) Attribute-Only Graph Ablation (removes sequential edges)
try:
    attr_file = "experiment_results/experiment_b8794e84d56b48dd94fa4f96d32f3931_proc_1463605/experiment_data.npy"
    attr_data = np.load(attr_file, allow_pickle=True).item()
    # For attr-only, data is under "attr_only" -> "spr_bench"
    attr_exp = attr_data.get("attr_only", {}).get("spr_bench", {})
except Exception as e:
    print(f"Error loading attribute-only ablation data: {e}")
    attr_exp = {}

if attr_exp:
    try:
        # Combined Loss and BWA curves in one figure
        epochs = np.arange(1, len(attr_exp.get("losses", {}).get("train", [])) + 1)
        train_loss = attr_exp["losses"]["train"]
        val_loss = attr_exp["losses"]["val"]
        train_bwa = [m["BWA"] for m in attr_exp["metrics"]["train"]]
        val_bwa = [m["BWA"] for m in attr_exp["metrics"]["val"]]
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].plot(epochs, train_loss, marker='o', label="Train Loss")
        axs[0].plot(epochs, val_loss, marker='s', label="Val Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("Attr-Only: Loss Curve")
        axs[0].legend()
        style_ax(axs[0])
        
        axs[1].plot(epochs, train_bwa, marker='o', label="Train BWA")
        axs[1].plot(epochs, val_bwa, marker='s', label="Val BWA")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("BWA")
        axs[1].set_title("Attr-Only: BWA Curve")
        axs[1].legend()
        style_ax(axs[1])
        
        fig.suptitle("Attribute-Only Ablation: Loss and BWA")
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fname = os.path.join("figures", "attronly_loss_bwa.png")
        fig.savefig(fname)
        plt.close(fig)
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error plotting attr-only loss/BWA curves: {e}")

    # Test metrics bar chart for attribute-only
    try:
        tm = attr_exp.get("test_metrics", {})
        labels_list = ["BWA", "CWA", "SWA", "StrWA"]
        vals = [tm.get(k, 0) for k in labels_list]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(labels_list, vals, color="mediumseagreen")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title("Attr-Only: Test Metrics")
        style_ax(ax)
        fig.tight_layout()
        fname = os.path.join("figures", "attronly_test_metrics.png")
        fig.savefig(fname)
        plt.close(fig)
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error plotting attr-only test metrics: {e}")
else:
    print("No attribute-only ablation data available.")

# (B) Single-Layer RGCN (Depth Ablation)
try:
    single_file = "experiment_results/experiment_ea280893de2b446d9b22e954366e1dc4_proc_1463605/experiment_data.npy"
    single_data = np.load(single_file, allow_pickle=True).item()
    single_exp = single_data.get("single_layer_rgcn", {}).get("spr_bench", {})
except Exception as e:
    print(f"Error loading single-layer RGCN data: {e}")
    single_exp = {}

if single_exp:
    # Loss curve
    try:
        epochs = list(range(1, len(single_exp.get("losses", {}).get("train", [])) + 1))
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(epochs, single_exp["losses"]["train"], marker='o', label="Train Loss")
        ax.plot(epochs, single_exp["losses"]["val"], marker='s', label="Val Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Single-Layer RGCN: Loss Curve")
        ax.legend()
        style_ax(ax)
        fig.tight_layout()
        fname = os.path.join("figures", "single_layer_loss.png")
        fig.savefig(fname)
        plt.close(fig)
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error plotting single-layer loss curve: {e}")
    # BWA curve
    try:
        tr_metrics = single_exp.get("metrics", {}).get("train", [])
        val_metrics = single_exp.get("metrics", {}).get("val", [])
        epochs = list(range(1, len(tr_metrics)+1))
        train_bwa = [m.get("BWA", np.nan) for m in tr_metrics]
        val_bwa = [m.get("BWA", np.nan) for m in val_metrics]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(epochs, train_bwa, marker='o', label="Train BWA")
        ax.plot(epochs, val_bwa, marker='s', label="Val BWA")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BWA")
        ax.set_title("Single-Layer RGCN: BWA Curve")
        ax.legend()
        style_ax(ax)
        fig.tight_layout()
        fname = os.path.join("figures", "single_layer_bwa.png")
        fig.savefig(fname)
        plt.close(fig)
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error plotting single-layer BWA curve: {e}")
    # CWA & SWA curves
    try:
        val_cwa = [m.get("CWA", np.nan) for m in val_metrics]
        val_swa = [m.get("SWA", np.nan) for m in val_metrics]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(epochs, val_cwa, marker='o', label="Val CWA")
        ax.plot(epochs, val_swa, marker='s', label="Val SWA")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Single-Layer RGCN: CWA & SWA (Validation)")
        ax.legend()
        style_ax(ax)
        fig.tight_layout()
        fname = os.path.join("figures", "single_layer_cwa_swa.png")
        fig.savefig(fname)
        plt.close(fig)
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error plotting single-layer CWA/SWA curves: {e}")
    # Label distribution bar chart
    try:
        preds = np.array(single_exp.get("predictions", []))
        gts = np.array(single_exp.get("ground_truth", []))
        if preds.size and gts.size:
            labels_unique = sorted(set(gts) | set(preds))
            pred_counts = [np.sum(preds == lab) for lab in labels_unique]
            gt_counts = [np.sum(gts == lab) for lab in labels_unique]
            x = np.arange(len(labels_unique))
            width = 0.35
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.bar(x - width/2, gt_counts, width, label="Ground Truth")
            ax.bar(x + width/2, pred_counts, width, label="Predictions")
            ax.set_xlabel("Label")
            ax.set_ylabel("Count")
            ax.set_title("Single-Layer RGCN: Label Distribution")
            ax.set_xticks(x)
            ax.set_xticklabels(labels_unique)
            ax.legend()
            style_ax(ax)
            fig.tight_layout()
            fname = os.path.join("figures", "single_layer_label_distribution.png")
            fig.savefig(fname)
            plt.close(fig)
            print(f"Saved {fname}")
        else:
            print("No predictions or ground_truth data for single-layer label distribution.")
    except Exception as e:
        print(f"Error plotting single-layer label distribution: {e}")
else:
    print("No single-layer RGCN data available.")

# (C) Unidirectional-Edges Graph Ablation
try:
    uni_file = "experiment_results/experiment_7545bfa89c9e4b4199bfeb795ca70b44_proc_1463604/experiment_data.npy"
    uni_data = np.load(uni_file, allow_pickle=True).item()
    uni_exp = uni_data.get("unidirectional_edges", {}).get("spr_bench", {})
except Exception as e:
    print(f"Error loading unidirectional-edges data: {e}")
    uni_exp = {}

if uni_exp:
    try:
        epochs = np.arange(1, len(uni_exp.get("losses", {}).get("train", [])) + 1)
        # Prepare metric arrays from training and validation for: Loss, BWA, CWA, SWA, StrWA
        train_loss = uni_exp["losses"]["train"]
        val_loss = uni_exp["losses"]["val"]
        def get_metric(metric_name):
            tr = [m.get(metric_name, np.nan) for m in uni_exp["metrics"]["train"]]
            va = [m.get(metric_name, np.nan) for m in uni_exp["metrics"]["val"]]
            return tr, va

        train_bwa, val_bwa = get_metric("BWA")
        train_cwa, val_cwa = get_metric("CWA")
        train_swa, val_swa = get_metric("SWA")
        train_strwa, val_strwa = get_metric("StrWA")

        fig, axs = plt.subplots(2, 3, figsize=(16, 10))
        axs = axs.flatten()
        # Plot 1: Loss
        axs[0].plot(epochs, train_loss, marker='o', label="Train")
        axs[0].plot(epochs, val_loss, marker='s', label="Val")
        axs[0].set_title("Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].legend()
        style_ax(axs[0])
        # Plot 2: BWA
        axs[1].plot(epochs, train_bwa, marker='o', label="Train")
        axs[1].plot(epochs, val_bwa, marker='s', label="Val")
        axs[1].set_title("BWA")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("BWA")
        axs[1].legend()
        style_ax(axs[1])
        # Plot 3: CWA
        axs[2].plot(epochs, train_cwa, marker='o', label="Train")
        axs[2].plot(epochs, val_cwa, marker='s', label="Val")
        axs[2].set_title("CWA")
        axs[2].set_xlabel("Epoch")
        axs[2].set_ylabel("CWA")
        axs[2].legend()
        style_ax(axs[2])
        # Plot 4: SWA
        axs[3].plot(epochs, train_swa, marker='o', label="Train")
        axs[3].plot(epochs, val_swa, marker='s', label="Val")
        axs[3].set_title("SWA")
        axs[3].set_xlabel("Epoch")
        axs[3].set_ylabel("SWA")
        axs[3].legend()
        style_ax(axs[3])
        # Plot 5: StrWA
        axs[4].plot(epochs, train_strwa, marker='o', label="Train")
        axs[4].plot(epochs, val_strwa, marker='s', label="Val")
        axs[4].set_title("StrWA")
        axs[4].set_xlabel("Epoch")
        axs[4].set_ylabel("StrWA")
        axs[4].legend()
        style_ax(axs[4])
        # Remove the extra empty subplot
        axs[5].axis('off')
        fig.suptitle("Unidirectional-Edges Ablation: Metrics Over Epochs", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fname = os.path.join("figures", "unidirectional_metrics.png")
        fig.savefig(fname)
        plt.close(fig)
        print(f"Saved {fname}")
        # Optionally, print final test metrics if available.
        print("Unidirectional Final Test Metrics:", uni_exp.get("test_metrics", {}))
    except Exception as e:
        print(f"Error plotting unidirectional-edges metrics: {e}")
else:
    print("No unidirectional-edges ablation data available.")

print("Figure aggregation complete.")
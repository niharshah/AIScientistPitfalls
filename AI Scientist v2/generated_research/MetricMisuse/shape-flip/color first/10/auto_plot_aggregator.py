#!/usr/bin/env python3
"""
Final Aggregator Script for Comprehensive Research Paper Figures

This script aggregates experimental results from three experiment categories:
  • BASELINE (GraphSAGE-related experiments with epoch tuning)
  • RESEARCH (Enhanced RGCN experiments)
  • ABLATION (A series of ablation studies)

It loads data from the provided .npy files (using full and exact paths),
creates publication‐ready figures in "figures/" with professional styling,
and wraps each individual plot in its own try/except block so that a failure
in one plot does not stop the rest.

All figures are produced with increased font sizes, clear labels/titles,
and without top/right spines. Figures are saved at dpi=300.
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import itertools

# Increase font size for publication modes
plt.rcParams.update({'font.size': 14})

# Create output directory for final paper figures
os.makedirs("figures", exist_ok=True)

def remove_top_right(ax):
    # Remove top and right spines for professional look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def save_and_close(fig, filename):
    fig.savefig(os.path.join("figures", filename), dpi=300, bbox_inches="tight")
    plt.close(fig)

def load_npy(file_path):
    try:
        data = np.load(file_path, allow_pickle=True).item()
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

########################################
# 1. BASELINE EXPERIMENT FIGURES
########################################

# Path from BASELINE_SUMMARY
baseline_file = "experiment_results/experiment_00c4ea7b14aa49f080d6935866182eaa_proc_1544297/experiment_data.npy"
baseline_data = load_npy(baseline_file)

if baseline_data is not None and "EPOCHS" in baseline_data and "SPR_BENCH" in baseline_data["EPOCHS"]:
    runs_dict = baseline_data["EPOCHS"]["SPR_BENCH"]
    run_keys = sorted([k for k in runs_dict.keys() if k.startswith("run_")],
                      key=lambda s: int(s.split("_")[-1]))

    # Plot 1A: Aggregated Loss Curves for Baseline runs.
    try:
        n = len(run_keys)
        cols = 3
        rows = math.ceil(n/cols)
        fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        axs = axs.flatten() if n>1 else [axs]
        for i, rk in enumerate(run_keys):
            losses = runs_dict[rk].get("losses", {})
            train_losses = losses.get("train", [])
            val_losses = losses.get("val", [])
            ax = axs[i]
            epochs = list(range(1, len(train_losses)+1))
            ax.plot(epochs, train_losses, label="Train Loss", marker="o")
            ax.plot(epochs, val_losses, label="Val Loss", marker="o")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(f"{rk}: Loss Curves")
            remove_top_right(ax)
            ax.legend()
        # Hide unused subplots
        for j in range(i+1, len(axs)):
            axs[j].axis("off")
        save_and_close(fig, "baseline_loss_curves.png")
    except Exception as e:
        print(f"Error in Baseline Loss Curves: {e}")

    # Plot 1B: Aggregated Harmonically Weighted Accuracy (HWA) Curves for Baseline runs.
    try:
        n = len(run_keys)
        cols = 3
        rows = math.ceil(n/cols)
        fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        axs = axs.flatten() if n>1 else [axs]
        for i, rk in enumerate(run_keys):
            metrics_list = runs_dict[rk].get("metrics", {}).get("val", [])
            if metrics_list:
                hwa = [m.get("hwa", 0) for m in metrics_list]
                epochs = list(range(1, len(hwa)+1))
                ax = axs[i]
                ax.plot(epochs, hwa, marker="o", color="tab:purple")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Harmonic Weighted Acc")
                ax.set_title(f"{rk}: Val HWA")
                remove_top_right(ax)
            else:
                axs[i].text(0.5, 0.5, "No metrics", ha="center", va="center")
        for j in range(i+1, len(axs)):
            axs[j].axis("off")
        save_and_close(fig, "baseline_hwa_curves.png")
    except Exception as e:
        print(f"Error in Baseline HWA Curves: {e}")

    # Plot 1C: Baseline Best-Run Confusion Matrix
    try:
        best_run_key = runs_dict.get("best_run", None)
        if best_run_key is None:
            # Try to use a default run if "best_run" not provided.
            best_run_key = run_keys[0] if run_keys else None
        if best_run_key:
            preds = runs_dict.get("predictions", [])
            golds = runs_dict.get("ground_truth", [])
            if not preds or not golds:
                # Sometimes the best run data is stored under the specific run key
                preds = runs_dict[best_run_key].get("predictions", [])
                golds = runs_dict[best_run_key].get("ground_truth", [])
            labels = sorted(list(set(golds) | set(preds)))
            lbl2idx = {l: i for i, l in enumerate(labels)}
            cm = np.zeros((len(labels), len(labels)), dtype=int)
            for g, p in zip(golds, preds):
                cm[lbl2idx[g], lbl2idx[p]] += 1
            fig, ax = plt.subplots(figsize=(6,5))
            im = ax.imshow(cm, cmap="Blues")
            remove_top_right(ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Baseline Test Confusion Matrix")
            plt.colorbar(im, ax=ax)
            # Annotate cells
            for i in range(len(labels)):
                for j in range(len(labels)):
                    color = "white" if cm[i,j] > (cm.max()/2.) else "black"
                    ax.text(j, i, cm[i,j], ha="center", va="center", color=color)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            save_and_close(fig, "baseline_confusion_matrix.png")
        else:
            print("No best run information found for Baseline confusion matrix.")
    except Exception as e:
        print(f"Error in Baseline Confusion Matrix: {e}")
else:
    print("Baseline data not found or improperly formatted.")


########################################
# 2. RESEARCH EXPERIMENT FIGURES
########################################

# Path from RESEARCH_SUMMARY
research_file = "experiment_results/experiment_af5219d2109c4f8a978c4dde5ead197c_proc_1551993/experiment_data.npy"
research_data = load_npy(research_file)

if research_data is not None and "SPR_BENCH" in research_data:
    spr = research_data["SPR_BENCH"]
    # Plot 2A: Research Loss Curves (Train vs Validation)
    try:
        train_losses = spr.get("losses", {}).get("train", [])
        val_losses = spr.get("losses", {}).get("val", [])
        epochs = list(range(1, len(train_losses)+1))
        fig, ax = plt.subplots(figsize=(6,5))
        ax.plot(epochs, train_losses, label="Train Loss", marker="o")
        ax.plot(epochs, val_losses, label="Val Loss", marker="o")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.set_title("Research: Loss Curves (Train vs Validation)")
        remove_top_right(ax)
        ax.legend()
        save_and_close(fig, "research_loss_curves.png")
    except Exception as e:
        print(f"Error in Research Loss Curves: {e}")

    # Plot 2B: Research Validation Metrics (CWA, SWA, CpxWA)
    try:
        val_metrics = spr.get("metrics", {}).get("val", [])
        if val_metrics:
            epochs = list(range(1, len(val_metrics)+1))
            cwa = [m.get("cwa", 0) for m in val_metrics]
            swa = [m.get("swa", 0) for m in val_metrics]
            cpx = [m.get("cpxwa", 0) for m in val_metrics]
            fig, ax = plt.subplots(figsize=(6,5))
            ax.plot(epochs, cwa, label="CWA", marker="o")
            ax.plot(epochs, swa, label="SWA", marker="o")
            ax.plot(epochs, cpx, label="CpxWA", marker="o")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Weighted Accuracy")
            ax.set_title("Research: Validation Accuracy Curves")
            remove_top_right(ax)
            ax.legend()
            save_and_close(fig, "research_val_metrics.png")
        else:
            print("No research validation metric data available.")
    except Exception as e:
        print(f"Error in Research validation metrics plot: {e}")

    # Plot 2C: Research Test Metrics Bar Chart
    try:
        test_m = spr.get("metrics", {}).get("test", {})
        if test_m:
            metrics = ["cwa", "swa", "cpxwa"]
            values = [test_m.get(m, 0) for m in metrics]
            fig, ax = plt.subplots(figsize=(6,5))
            ax.bar(metrics, values, color=["tab:blue", "tab:orange", "tab:green"])
            for i, v in enumerate(values):
                ax.text(i, v+0.01, f"{v:.2f}", ha="center")
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Score")
            ax.set_title("Research: Test Metrics (CWA/SWA/CpxWA)")
            remove_top_right(ax)
            save_and_close(fig, "research_test_metrics.png")
        else:
            print("No research test metrics found.")
    except Exception as e:
        print(f"Error in Research test metrics plot: {e}")

    # Plot 2D: Research Confusion Matrix on Test Set
    try:
        preds = spr.get("predictions", [])
        golds = spr.get("ground_truth", [])
        if preds and golds:
            labels = sorted(list(set(golds) | set(preds)))
            lbl2idx = {l: i for i, l in enumerate(labels)}
            cm = np.zeros((len(labels), len(labels)), dtype=int)
            for g, p in zip(golds, preds):
                cm[lbl2idx[g], lbl2idx[p]] += 1
            fig, ax = plt.subplots(figsize=(6,5))
            im = ax.imshow(cm, cmap="Blues")
            remove_top_right(ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Research: Test Confusion Matrix")
            plt.colorbar(im, ax=ax)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    color = "white" if cm[i,j] > (cm.max()/2.) else "black"
                    ax.text(j, i, cm[i,j], ha="center", va="center", color=color)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            save_and_close(fig, "research_confusion_matrix.png")
        else:
            print("Research predictions or ground truth not available for confusion matrix.")
    except Exception as e:
        print(f"Error in Research Confusion Matrix plot: {e}")
else:
    print("Research experiment data not found or improperly formatted.")

########################################
# 3. ABLATION STUDIES (Aggregated)
########################################

# List of ablation experiments with their file paths and keys.
ablation_experiments = [
    {"name": "No-Shape/Color Edges", "file": "experiment_results/experiment_350faea3424f45c3922ae66299332127_proc_1557385/experiment_data.npy", "key": "SPR_BENCH"},
    {"name": "Single-Relation GCN", "file": "experiment_results/experiment_8ddf2dcf1a584be5b4b695c20bef45eb_proc_1557386/experiment_data.npy", "key": "SPR_BENCH"},
    {"name": "Remove Pos", "file": "experiment_results/experiment_d390036463814de9823c3c6a4ec04710_proc_1557387/experiment_data.npy", "key": "SPR_BENCH"},
    {"name": "No-Shape Embedding", "file": "experiment_results/experiment_27da17102d124859b96de3da69c5b2c4_proc_1557388/experiment_data.npy", "key": "SPR_BENCH"},
    {"name": "No-GNN", "file": "experiment_results/experiment_b922ad84465d457ab1289a42a0b937b0_proc_1557386/experiment_data.npy", "key": "SPR_BENCH"},
    {"name": "No-Sequential-Edges", "file": "experiment_results/experiment_558ab6f6a35640eebe368f9b69f7b42e_proc_1557387/experiment_data.npy", "key": "NoSeqEdge"},
    {"name": "Concat-Embeddings", "file": "experiment_results/experiment_e5109803cc414a90a62558591c468ace_proc_1557385/experiment_data.npy", "key": "SPR_BENCH"}
]

# Containers for aggregated ablation plots data
abl_loss_data = []
abl_val_metrics = []
abl_test_metrics = []  # List of dicts with keys: name, cwa, swa, cpxwa

for exp in ablation_experiments:
    data = load_npy(exp["file"])
    if data is None:
        print(f"Skipping {exp['name']} due to load failure.")
        continue
    # Use provided key if exists, otherwise use the entire data
    d = data.get(exp["key"], data)
    # Ensure required fields exist
    losses = d.get("losses", {})
    train = losses.get("train", None)
    val = losses.get("val", None)
    metrics_val = d.get("metrics", {}).get("val", None)
    metrics_test = d.get("metrics", {}).get("test", None)
    if train is not None and val is not None:
        abl_loss_data.append({
            "name": exp["name"],
            "epochs": list(range(1, len(train)+1)),
            "train": train,
            "val": val
        })
    if metrics_val is not None:
        # For simplicity, store first epoch index from the metric dict if available,
        # otherwise use index order.
        epochs = list(range(1, len(metrics_val)+1))
        abl_val_metrics.append({
            "name": exp["name"],
            "epochs": epochs,
            "cwa": [m.get("cwa", 0) for m in metrics_val],
            "swa": [m.get("swa", 0) for m in metrics_val],
            "cpx": [m.get("cpxwa", 0) for m in metrics_val],
        })
    if metrics_test is not None:
        abl_test_metrics.append({
            "name": exp["name"],
            "cwa": metrics_test.get("cwa", 0),
            "swa": metrics_test.get("swa", 0),
            "cpx": metrics_test.get("cpxwa", 0)
        })

# Plot 3A: Aggregated Ablation Loss Curves (one subplot per ablation)
try:
    n = len(abl_loss_data)
    if n > 0:
        cols = 3
        rows = math.ceil(n/cols)
        fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        axs = axs.flatten() if n>1 else [axs]
        for i, d in enumerate(abl_loss_data):
            ax = axs[i]
            ax.plot(d["epochs"], d["train"], label="Train Loss", marker="o")
            ax.plot(d["epochs"], d["val"], label="Val Loss", marker="o")
            ax.set_title(d["name"])
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            remove_top_right(ax)
            ax.legend()
        for j in range(i+1, len(axs)):
            axs[j].axis("off")
        save_and_close(fig, "ablation_loss_curves.png")
    else:
        print("No ablation loss data to plot.")
except Exception as e:
    print(f"Error in Ablation Loss Curves: {e}")

# Plot 3B: Aggregated Ablation Validation Metrics Curves
try:
    n = len(abl_val_metrics)
    if n > 0:
        cols = 3
        rows = math.ceil(n/cols)
        fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        axs = axs.flatten() if n>1 else [axs]
        for i, d in enumerate(abl_val_metrics):
            ax = axs[i]
            ax.plot(d["epochs"], d["cwa"], label="CWA", marker="o")
            ax.plot(d["epochs"], d["swa"], label="SWA", marker="o")
            ax.plot(d["epochs"], d["cpx"], label="CpxWA", marker="o")
            ax.set_title(d["name"])
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            remove_top_right(ax)
            ax.legend()
        for j in range(i+1, len(axs)):
            axs[j].axis("off")
        save_and_close(fig, "ablation_validation_metrics.png")
    else:
        print("No ablation validation metrics data to plot.")
except Exception as e:
    print(f"Error in Ablation Validation Metrics Curves: {e}")

# Plot 3C: Aggregated Ablation Test Metrics Bar Chart (grouped bar chart)
try:
    if len(abl_test_metrics) > 0:
        labels = [d["name"] for d in abl_test_metrics]
        cwa_vals = [d["cwa"] for d in abl_test_metrics]
        swa_vals = [d["swa"] for d in abl_test_metrics]
        cpx_vals = [d["cpx"] for d in abl_test_metrics]
        x = np.arange(len(labels))
        width = 0.25
        fig, ax = plt.subplots(figsize=(max(6, len(labels)*1.2),5))
        rects1 = ax.bar(x - width, cwa_vals, width, label="CWA", color="tab:blue")
        rects2 = ax.bar(x, swa_vals, width, label="SWA", color="tab:orange")
        rects3 = ax.bar(x + width, cpx_vals, width, label="CpxWA", color="tab:green")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylim(0,1)
        ax.set_ylabel("Accuracy")
        ax.set_title("Ablation Study: Test Weighted Accuracies")
        remove_top_right(ax)
        ax.legend()
        save_and_close(fig, "ablation_test_metrics.png")
    else:
        print("No ablation test metrics data to plot.")
except Exception as e:
    print(f"Error in Ablation Test Metrics Bar Chart: {e}")

print("All figures have been generated and saved in the 'figures/' directory.")
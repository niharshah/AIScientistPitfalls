#!/usr/bin/env python3
"""
Final Aggregator Script for GNN for SPR Experiments
This script loads experiment_data from published .npy files and aggregates final scientific plots.
All final figures are saved under the "figures/" directory.
Each plot is wrapped in a try/except block so failure in one does not block others.
Font sizes and spines are adjusted for professional publication.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Set global plotting parameters for publication quality
plt.rcParams.update({
    "font.size": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Make sure the final plots directory exists
os.makedirs("figures", exist_ok=True)

# ---------------------------
# Helper Functions
# ---------------------------
def load_experiment_data(path):
    try:
        data = np.load(path, allow_pickle=True).item()
        return data
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def save_fig(fig, filename):
    out_path = os.path.join("figures", filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")

# ---------------------------
# 1. BASELINE (from GNN for SPR baseline)
# npy file from baseline summary:
# "experiment_results/experiment_bff4641a5ff54195af51ebb5e46d03f2_proc_1513229/experiment_data.npy"
baseline_path = "experiment_results/experiment_bff4641a5ff54195af51ebb5e46d03f2_proc_1513229/experiment_data.npy"
baseline_data = load_experiment_data(baseline_path)
if baseline_data:
    ed = baseline_data.get("SPR", {})
    epochs = ed.get("epochs", [])
    losses = ed.get("losses", {})
    metrics = ed.get("metrics", {})
    lr_vals = ed.get("lr_values", [])
    best_lr = ed.get("best_lr", None)
    # For the baseline we assume losses["train"] and losses["val"] are lists of lists (per lr index)
    try:
        # Figure A: Combined baseline overview with 3 subplots: Train/Val Loss, HM curves, LR Sweep Bar Chart
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        # Subplot 1: Loss curves for best LR
        if best_lr in lr_vals:
            best_idx = lr_vals.index(best_lr)
        else:
            best_idx = 0
        axs[0].plot(epochs, losses.get("train", [[]])[best_idx], label="Train Loss")
        axs[0].plot(epochs, losses.get("val", [[]])[best_idx], label="Validation Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Cross-Entropy Loss")
        axs[0].set_title(f"Loss Curves (Best LR = {best_lr})")
        axs[0].legend()
        # Subplot 2: HM curves for best LR
        tr_hm = [m.get("HM", np.nan) for m in metrics.get("train", [[]])[best_idx]]
        val_hm = [m.get("HM", np.nan) for m in metrics.get("val", [[]])[best_idx]]
        axs[1].plot(epochs, tr_hm, label="Train HM")
        axs[1].plot(epochs, val_hm, label="Validation HM")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Harmonic Mean (HM)")
        axs[1].set_title("HM Curves")
        axs[1].legend()
        # Subplot 3: Bar chart of final validation HM vs LR
        final_hms = [metrics.get("val", [[]])[i][-1].get("HM", np.nan) for i in range(len(lr_vals)) if len(metrics.get("val", [[]])[i])>0]
        axs[2].bar([str(lr) for lr in lr_vals], final_hms, color="skyblue")
        axs[2].set_xlabel("Learning Rate")
        axs[2].set_ylabel("Final Val HM")
        axs[2].set_title("LR Sweep: Final Val HM")
        save_fig(fig, "baseline_overview.png")
    except Exception as e:
        print(f"Error in baseline overview plot: {e}")
    
    try:
        # Figure B: Baseline Confusion Matrix (test)
        preds = np.array(ed.get("predictions", []), dtype=int)
        gts = np.array(ed.get("ground_truth", []), dtype=int)
        if preds.size and gts.size:
            n_cls = int(max(preds.max(), gts.max())) + 1
            cm = np.zeros((n_cls, n_cls), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Ground Truth")
            ax.set_title("Baseline: Confusion Matrix (Test)")
            fig.colorbar(im, ax=ax)
            for i in range(n_cls):
                for j in range(n_cls):
                    ax.text(j, i, cm[i, j], ha="center", va="center",
                            color="white" if cm[i, j] > cm.max()/2 else "black")
            save_fig(fig, "baseline_confusion_matrix.png")
        else:
            print("Baseline: No predictions/ground truth available for confusion matrix.")
    except Exception as e:
        print(f"Error in baseline confusion matrix plot: {e}")

# ---------------------------
# 2. RESEARCH (from upgraded GNN, multi-head GAT)
# npy file from research summary:
# "experiment_results/experiment_8b962da3fdf44a4d9a34cf0e2e4c5922_proc_1517534/experiment_data.npy"
research_path = "experiment_results/experiment_8b962da3fdf44a4d9a34cf0e2e4c5922_proc_1517534/experiment_data.npy"
research_data = load_experiment_data(research_path)
if research_data:
    ed = research_data.get("SPR", {})
    epochs = ed.get("epochs", [])
    loss_tr = ed.get("losses", {}).get("train", [])
    loss_val = ed.get("losses", {}).get("val", [])
    # Assume HWA is under metrics and stored per epoch
    hwa_tr = [m.get("HWA", np.nan) for m in ed.get("metrics", {}).get("train", [])]
    hwa_val = [m.get("HWA", np.nan) for m in ed.get("metrics", {}).get("val", [])]
    preds = np.array(ed.get("predictions", []), dtype=int)
    gts = np.array(ed.get("ground_truth", []), dtype=int)
    try:
        # Figure C: Combined research overview in 3 subplots: Loss, HWA, Confusion Matrix
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        # Loss curves
        axs[0].plot(epochs, loss_tr, label="Train Loss")
        axs[0].plot(epochs, loss_val, label="Val Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("Research: Train vs Val Loss")
        axs[0].legend()
        # HWA curves
        axs[1].plot(epochs, hwa_tr, label="Train HWA")
        axs[1].plot(epochs, hwa_val, label="Val HWA")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("HWA")
        axs[1].set_title("Research: HWA Curves")
        axs[1].legend()
        # Confusion Matrix if available
        if preds.size and gts.size:
            n_cls = int(max(preds.max(), gts.max())) + 1
            cm = np.zeros((n_cls, n_cls), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            im = axs[2].imshow(cm, cmap="Blues")
            axs[2].set_xlabel("Predicted")
            axs[2].set_ylabel("Ground Truth")
            axs[2].set_title("Research: Confusion Matrix")
            fig.colorbar(im, ax=axs[2])
            for i in range(n_cls):
                for j in range(n_cls):
                    axs[2].text(j, i, cm[i, j], ha="center", va="center",
                               color="white" if cm[i, j] > cm.max()/2 else "black")
        else:
            axs[2].text(0.5, 0.5, "No Data", ha="center", va="center")
        save_fig(fig, "research_overview.png")
    except Exception as e:
        print(f"Error in research overview plot: {e}")

# ---------------------------
# 3. ABLATION STUDIES
# We produce several figures from selected ablation experiments.
# For clarity, each ablation variant is processed separately.

# A. Remove Shape/Color Similarity Edges
# npy: "experiment_results/experiment_562571aba5894ebe8c71b875e71fc68c_proc_1520778/experiment_data.npy"
sim_edges_path = "experiment_results/experiment_562571aba5894ebe8c71b875e71fc68c_proc_1520778/experiment_data.npy"
sim_data = load_experiment_data(sim_edges_path)
if sim_data:
    ed = sim_data.get("SPR", {})
    epochs = ed.get("epochs", [])
    loss_tr = ed.get("losses", {}).get("train", [])
    loss_val = ed.get("losses", {}).get("val", [])
    hwa_tr = [m.get("HWA", np.nan) for m in ed.get("metrics", {}).get("train", [])]
    hwa_val = [m.get("HWA", np.nan) for m in ed.get("metrics", {}).get("val", [])]
    preds = ed.get("predictions", [])
    gts = ed.get("ground_truth", [])
    try:
        # Figure D: 3 subplots: Loss curves, HWA curves, Confusion Matrix for similarity edges ablation
        fig, axs = plt.subplots(1,3, figsize=(18,5))
        axs[0].plot(epochs, loss_tr, label="Train Loss")
        axs[0].plot(epochs, loss_val, label="Val Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("Ablation (No Similarity Edges): Loss")
        axs[0].legend()
        axs[1].plot(epochs, hwa_tr, label="Train HWA")
        axs[1].plot(epochs, hwa_val, label="Val HWA")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("HWA")
        axs[1].set_title("Ablation: HWA Curves")
        axs[1].legend()
        # Confusion Matrix
        preds_arr = np.array(preds, dtype=int)
        gts_arr = np.array(gts, dtype=int)
        if preds_arr.size and gts_arr.size:
            n_cls = int(max(preds_arr.max(), gts_arr.max())) + 1
            cm = np.zeros((n_cls, n_cls), dtype=int)
            for t, p in zip(gts_arr, preds_arr):
                cm[t, p] += 1
            im = axs[2].imshow(cm, cmap="Blues")
            axs[2].set_xlabel("Predicted")
            axs[2].set_ylabel("Ground Truth")
            axs[2].set_title("Ablation: Confusion Matrix")
            fig.colorbar(im, ax=axs[2])
            for i in range(n_cls):
                for j in range(n_cls):
                    axs[2].text(j, i, cm[i, j], ha="center", va="center",
                                color="white" if cm[i, j] > cm.max()/2 else "black")
        else:
            axs[2].text(0.5,0.5,"No Data", ha="center", va="center")
        save_fig(fig, "ablation_remove_similarity_edges.png")
    except Exception as e:
        print(f"Error in ablation (Remove Similarity Edges) plot: {e}")

# B. Remove Positional Feature (No-POS)
# npy: "experiment_results/experiment_e9e95618dbe04048a21e1c8181e012a2_proc_1520779/experiment_data.npy"
no_pos_path = "experiment_results/experiment_e9e95618dbe04048a21e1c8181e012a2_proc_1520779/experiment_data.npy"
no_pos_data = load_experiment_data(no_pos_path)
if no_pos_data:
    ed = no_pos_data.get("SPR", {})
    epochs = np.array(ed.get("epochs", []))
    loss_tr = np.array(ed.get("losses", {}).get("train", []))
    loss_val = np.array(ed.get("losses", {}).get("val", []))
    # Assuming metrics "CWA", "SWA", "HWA" exists
    cwa_tr = np.array([m.get("CWA", np.nan) for m in ed.get("metrics", {}).get("train", [])])
    cwa_val = np.array([m.get("CWA", np.nan) for m in ed.get("metrics", {}).get("val", [])])
    swa_tr = np.array([m.get("SWA", np.nan) for m in ed.get("metrics", {}).get("train", [])])
    swa_val = np.array([m.get("SWA", np.nan) for m in ed.get("metrics", {}).get("val", [])])
    hwa_tr = np.array([m.get("HWA", np.nan) for m in ed.get("metrics", {}).get("train", [])])
    hwa_val = np.array([m.get("HWA", np.nan) for m in ed.get("metrics", {}).get("val", [])])
    preds = np.array(ed.get("predictions", []), dtype=int)
    gts = np.array(ed.get("ground_truth", []), dtype=int)
    try:
        # Figure E: 3 subplots: Loss curves, HWA curves, Confusion Matrix (No-POS)
        fig, axs = plt.subplots(1,3, figsize=(18,5))
        axs[0].plot(epochs, loss_tr, label="Train Loss")
        axs[0].plot(epochs, loss_val, label="Validation Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("No-POS: Loss Curves")
        axs[0].legend()
        axs[1].plot(epochs, hwa_tr, label="Train HWA")
        axs[1].plot(epochs, hwa_val, label="Validation HWA")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("HWA")
        axs[1].set_title("No-POS: HWA Curves")
        axs[1].legend()
        # Confusion Matrix
        if preds.size and gts.size:
            n_cls = int(max(preds.max(), gts.max()))+1
            cm = np.zeros((n_cls, n_cls), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            im = axs[2].imshow(cm, cmap="Blues")
            axs[2].set_xlabel("Predicted")
            axs[2].set_ylabel("Ground Truth")
            axs[2].set_title("No-POS: Confusion Matrix")
            fig.colorbar(im, ax=axs[2])
            for i in range(n_cls):
                for j in range(n_cls):
                    axs[2].text(j, i, cm[i, j], ha="center", va="center",
                                color="white" if cm[i, j]>cm.max()/2 else "black")
        else:
            axs[2].text(0.5,0.5,"No Data", ha="center", va="center")
        save_fig(fig, "ablation_no_pos.png")
    except Exception as e:
        print(f"Error in ablation (No-POS) plot: {e}")

# C. Single-Head Attention (No-MultiHead)
# npy: "experiment_results/experiment_18a61f2d7e7b4af586add0b913645ccd_proc_1520780/experiment_data.npy"
no_multihead_path = "experiment_results/experiment_18a61f2d7e7b4af586add0b913645ccd_proc_1520780/experiment_data.npy"
no_multi_data = load_experiment_data(no_multihead_path)
if no_multi_data:
    ed = no_multi_data.get("NoMultiHead", {}).get("SPR", None)
    if ed:
        epochs = ed.get("epochs", [])
        loss_tr = ed.get("losses", {}).get("train", [])
        loss_val = ed.get("losses", {}).get("val", [])
        metrics_val = ed.get("metrics", {}).get("val", [])
        # Extract validation metrics for CWA, SWA, HWA
        cwa = [m.get("CWA", np.nan) for m in metrics_val]
        swa = [m.get("SWA", np.nan) for m in metrics_val]
        hwa = [m.get("HWA", np.nan) for m in metrics_val]
        preds = np.array(ed.get("predictions", []), dtype=int)
        gts = np.array(ed.get("ground_truth", []), dtype=int)
        try:
            # Figure F: 3 subplots: Loss curves, Metric curves, Confusion Matrix (No-MultiHead)
            fig, axs = plt.subplots(1,3, figsize=(18,5))
            axs[0].plot(epochs, loss_tr, label="Train Loss")
            axs[0].plot(epochs, loss_val, label="Validation Loss")
            axs[0].set_xlabel("Epoch")
            axs[0].set_ylabel("Loss")
            axs[0].set_title("No-MultiHead: Loss Curves")
            axs[0].legend()
            axs[1].plot(epochs, cwa, label="CWA")
            axs[1].plot(epochs, swa, label="SWA")
            axs[1].plot(epochs, hwa, label="HWA")
            axs[1].set_xlabel("Epoch")
            axs[1].set_ylabel("Metric Value")
            axs[1].set_title("No-MultiHead: Validation Metrics")
            axs[1].legend()
            if preds.size and gts.size:
                n_cls = int(max(preds.max(), gts.max()))+1
                cm = np.zeros((n_cls, n_cls), dtype=int)
                for t, p in zip(gts, preds):
                    cm[t, p] += 1
                im = axs[2].imshow(cm, cmap="Blues")
                axs[2].set_xlabel("Predicted")
                axs[2].set_ylabel("Ground Truth")
                axs[2].set_title("No-MultiHead: Confusion Matrix")
                fig.colorbar(im, ax=axs[2])
                for i in range(n_cls):
                    for j in range(n_cls):
                        axs[2].text(j, i, cm[i, j], ha="center", va="center",
                                    color="white" if cm[i, j]>cm.max()/2 else "black")
            else:
                axs[2].text(0.5, 0.5, "No Data", ha="center", va="center")
            save_fig(fig, "ablation_no_multihead.png")
        except Exception as e:
            print(f"Error in ablation (No-MultiHead) plot: {e}")
    else:
        print("NoMultiHead data not found in the loaded file.")

# D. No-Bidirectional-Edges
# npy: "experiment_results/experiment_bcd63e121ae841b18f5010309cc1eb87_proc_1520779/experiment_data.npy"
nobidir_path = "experiment_results/experiment_bcd63e121ae841b18f5010309cc1eb87_proc_1520779/experiment_data.npy"
nobidir_data = load_experiment_data(nobidir_path)
if nobidir_data:
    ed = nobidir_data.get("NoBiDir", {}).get("SPR", {})
    epochs = ed.get("epochs", [])
    tr_loss = ed.get("losses", {}).get("train", [])
    val_loss = ed.get("losses", {}).get("val", [])
    # Use HWA from metrics for brevity
    tr_met = [m.get("HWA", np.nan) for m in ed.get("metrics", {}).get("train", [])]
    val_met = [m.get("HWA", np.nan) for m in ed.get("metrics", {}).get("val", [])]
    preds = np.array(ed.get("predictions", []), dtype=int)
    gts = np.array(ed.get("ground_truth", []), dtype=int)
    try:
        # Figure G: 3 subplots: Loss curve, HWA curve, Confusion Matrix (NoBiDir)
        fig, axs = plt.subplots(1, 3, figsize=(18,5))
        axs[0].plot(epochs, tr_loss, label="Train Loss")
        axs[0].plot(epochs, val_loss, label="Validation Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("No-Bidirectional Edges: Loss")
        axs[0].legend()
        axs[1].plot(epochs, tr_met, label="Train HWA")
        axs[1].plot(epochs, val_met, label="Validation HWA")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("HWA")
        axs[1].set_title("No-Bidirectional Edges: HWA")
        axs[1].legend()
        if preds.size and gts.size:
            n_cls = int(max(preds.max(), gts.max()))+1
            cm = np.zeros((n_cls, n_cls), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            im = axs[2].imshow(cm, cmap="Blues")
            axs[2].set_xlabel("Predicted")
            axs[2].set_ylabel("Ground Truth")
            axs[2].set_title("No-Bidirectional Edges: Confusion Matrix")
            fig.colorbar(im, ax=axs[2])
            for i in range(n_cls):
                for j in range(n_cls):
                    axs[2].text(j, i, cm[i, j], ha="center", va="center",
                                color="white" if cm[i, j] > cm.max()/2 else "black")
        else:
            axs[2].text(0.5, 0.5, "No Data", ha="center", va="center")
        save_fig(fig, "ablation_no_bidirectional_edges.png")
    except Exception as e:
        print(f"Error in ablation (No-Bidirectional-Edges) plot: {e}")

# E. No-Node-Features (Edge-Only Graph)
# npy: "experiment_results/experiment_5ab6eed00cd3481ab33abe944963a4bc_proc_1520781/experiment_data.npy"
no_node_path = "experiment_results/experiment_5ab6eed00cd3481ab33abe944963a4bc_proc_1520781/experiment_data.npy"
no_node_data = load_experiment_data(no_node_path)
if no_node_data:
    # For this ablation, we also produce a scatter plot and class distribution bar chart.
    ed = no_node_data.get("SPR", {})
    epochs = np.array(ed.get("epochs", []))
    loss_tr = np.array(ed.get("losses", {}).get("train", []))
    loss_val = np.array(ed.get("losses", {}).get("val", []))
    hwa_tr = np.array([m.get("HWA", np.nan) for m in ed.get("metrics", {}).get("train", [])])
    hwa_val = np.array([m.get("HWA", np.nan) for m in ed.get("metrics", {}).get("val", [])])
    preds = np.array(ed.get("predictions", []))
    gts = np.array(ed.get("ground_truth", []))
    try:
        # Figure H: 2 rows x 2 cols: Loss, HWA, Scatter (GT vs Pred), and Class Distribution
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        # Loss
        axs[0,0].plot(epochs, loss_tr, label="Train Loss")
        axs[0,0].plot(epochs, loss_val, label="Validation Loss")
        axs[0,0].set_xlabel("Epoch")
        axs[0,0].set_ylabel("Loss")
        axs[0,0].set_title("No-Node-Features: Loss Curves")
        axs[0,0].legend()
        # HWA
        axs[0,1].plot(epochs, hwa_tr, label="Train HWA")
        axs[0,1].plot(epochs, hwa_val, label="Validation HWA")
        axs[0,1].set_xlabel("Epoch")
        axs[0,1].set_ylabel("HWA")
        axs[0,1].set_title("No-Node-Features: HWA Curves")
        axs[0,1].legend()
        # Scatter plot GT vs Pred with jitter
        if gts.size and preds.size:
            jitter = (np.random.rand(len(gts))-0.5)*0.2
            axs[1,0].scatter(gts + jitter, preds + jitter, alpha=0.6)
            max_lab = int(max(gts.max(), preds.max())) if gts.size and preds.size else 1
            axs[1,0].plot([0, max_lab], [0, max_lab], "k--", linewidth=1)
            axs[1,0].set_xlabel("Ground Truth")
            axs[1,0].set_ylabel("Prediction")
            axs[1,0].set_title("No-Node-Features: GT vs Pred Scatter")
        else:
            axs[1,0].text(0.5, 0.5, "No Data", ha="center", va="center")
        # Bar chart for class distribution
        if gts.size and preds.size:
            classes = np.arange(int(max(gts.max(), preds.max()))+1)
            gt_counts = np.bincount(gts, minlength=len(classes))
            pred_counts = np.bincount(preds, minlength=len(classes))
            width = 0.35
            x = np.arange(len(classes))
            axs[1,1].bar(x - width/2, gt_counts, width, label="Ground Truth")
            axs[1,1].bar(x + width/2, pred_counts, width, label="Predictions")
            axs[1,1].set_xlabel("Class")
            axs[1,1].set_ylabel("Count")
            axs[1,1].set_title("No-Node-Features: Class Distribution")
            axs[1,1].legend()
        else:
            axs[1,1].text(0.5, 0.5, "No Data", ha="center", va="center")
        save_fig(fig, "ablation_no_node_features.png")
    except Exception as e:
        print(f"Error in ablation (No-Node-Features) plot: {e}")

# F. Single-GAT-Layer (1-Hop Message Passing Only)
# npy: "experiment_results/experiment_06151e27405145618c565b94704b728d_proc_1520778/experiment_data.npy"
single_gat_path = "experiment_results/experiment_06151e27405145618c565b94704b728d_proc_1520778/experiment_data.npy"
single_gat_data = load_experiment_data(single_gat_path)
if single_gat_data:
    ed = single_gat_data.get("single_gat_layer", {}).get("SPR", {})
    epochs = np.array(ed.get("epochs", []))
    loss_tr = np.array(ed.get("losses", {}).get("train", []))
    loss_val = np.array(ed.get("losses", {}).get("val", []))
    train_hwa = np.array([m.get("HWA", np.nan) for m in ed.get("metrics", {}).get("train", [])])
    val_hwa = np.array([m.get("HWA", np.nan) for m in ed.get("metrics", {}).get("val", [])])
    # Final test metrics from the last epoch in validation metrics dict:
    final_metrics = ed.get("metrics", {}).get("val", [])[-1] if ed.get("metrics", {}).get("val", []) else {}
    try:
        # Figure I: 3 subplots: Loss curves, HWA curves, and Test Metrics Bar Chart (CWA, SWA, HWA)
        fig, axs = plt.subplots(1, 3, figsize=(18,5))
        axs[0].plot(epochs, loss_tr, label="Train Loss")
        axs[0].plot(epochs, loss_val, label="Val Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("1-Hop GAT: Loss Curves")
        axs[0].legend()
        axs[1].plot(epochs, train_hwa, label="Train HWA")
        axs[1].plot(epochs, val_hwa, label="Val HWA")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("HWA")
        axs[1].set_title("1-Hop GAT: HWA Curves")
        axs[1].legend()
        # Test bar chart: get CWA, SWA, HWA from final_metrics if available
        bars = ["CWA", "SWA", "HWA"]
        vals = [final_metrics.get("CWA", np.nan), final_metrics.get("SWA", np.nan), final_metrics.get("HWA", np.nan)]
        axs[2].bar(bars, vals, color=["tab:blue", "tab:orange", "tab:green"])
        axs[2].set_ylim(0, 1)
        axs[2].set_ylabel("Metric Value")
        axs[2].set_title("1-Hop GAT: Test Metrics")
        save_fig(fig, "ablation_single_gat_layer.png")
    except Exception as e:
        print(f"Error in ablation (Single-GAT-Layer) plot: {e}")

# G. Remove-Color-Features (No-Color-Embedding)
# npy: "experiment_results/experiment_870e273b393142b49f2e915d38666577_proc_1520780/experiment_data.npy"
no_color_path = "experiment_results/experiment_870e273b393142b49f2e915d38666577_proc_1520780/experiment_data.npy"
no_color_data = load_experiment_data(no_color_path)
if no_color_data:
    ed = no_color_data.get("NoColorEmbedding", {}).get("SPR", {})
    epochs = np.array(ed.get("epochs", []))
    loss_tr = np.array(ed.get("losses", {}).get("train", []))
    loss_val = np.array(ed.get("losses", {}).get("val", []))
    train_hwa = np.array([m.get("HWA", np.nan) for m in ed.get("metrics", {}).get("train", [])])
    val_hwa = np.array([m.get("HWA", np.nan) for m in ed.get("metrics", {}).get("val", [])])
    y_true = np.array(ed.get("ground_truth", []))
    y_pred = np.array(ed.get("predictions", []))
    try:
        # Figure J: 3 subplots: Loss curves, HWA curves, and Confusion Matrix (No-Color-Embedding)
        fig, axs = plt.subplots(1,3, figsize=(18,5))
        axs[0].plot(epochs, loss_tr, label="Train Loss")
        axs[0].plot(epochs, loss_val, label="Validation Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("No-Color-Embedding: Loss Curves")
        axs[0].legend()
        axs[1].plot(epochs, train_hwa, label="Train HWA")
        axs[1].plot(epochs, val_hwa, label="Validation HWA")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("HWA")
        axs[1].set_title("No-Color-Embedding: HWA Curves")
        axs[1].legend()
        if y_true.size and y_pred.size:
            classes = sorted(set(np.concatenate([y_true, y_pred])))
            n_cls = len(classes)
            cm = np.zeros((n_cls, n_cls), dtype=int)
            for t, p in zip(y_true, y_pred):
                # assume classes are integers and in sorted order
                idx_t = classes.index(t)
                idx_p = classes.index(p)
                cm[idx_t, idx_p] += 1
            im = axs[2].imshow(cm, cmap="Blues")
            axs[2].set_xlabel("Predicted")
            axs[2].set_ylabel("Ground Truth")
            axs[2].set_title("No-Color-Embedding: Confusion Matrix")
            fig.colorbar(im, ax=axs[2])
            for i in range(n_cls):
                for j in range(n_cls):
                    axs[2].text(j, i, cm[i, j], ha="center", va="center",
                                color="white" if cm[i, j]>cm.max()/2 else "black")
        else:
            axs[2].text(0.5, 0.5, "No Data", ha="center", va="center")
        save_fig(fig, "ablation_no_color_features.png")
    except Exception as e:
        print(f"Error in ablation (No-Color-Embedding) plot: {e}")

print("Final aggregation complete. All figures saved in 'figures/' directory.")
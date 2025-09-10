#!/usr/bin/env python3
"""
Aggregator script for final research paper figures.
This script aggregates experiment results from baseline, research and selected ablation studies.
All final figures are saved under the "figures/" directory.
Each plot is wrapped in a try‐except block so that if one fails the others are still produced.

Before running, make sure that the following .npy files exist at the full paths indicated:
  • Baseline: 
       "experiment_results/experiment_869bad450d6e4d8fb3f6149546f7258c_proc_1441297/experiment_data.npy"
  • Research:
       "experiment_results/experiment_8c3bdfc4c3064befa8eae50a48f257bd_proc_1445298/experiment_data.npy"
  • Ablation -- No-Homophily-Edges:
       "experiment_results/experiment_fd641770149e403ab958308bd2c06a11_proc_1449034/experiment_data.npy"
  • Ablation -- No-Sequential-Edges:
       "experiment_results/experiment_9d48e2ca1b0249de9471614b485b36f3_proc_1449035/experiment_data.npy"
  • Ablation -- Edge-Type-Collapsed Graph:
       "experiment_results/experiment_4f07a227d0ea43019a5e283703f49375_proc_1449036/experiment_data.npy"
  • Ablation -- Multi-Synthetic-Dataset Evaluation:
       "experiment_results/experiment_98de24e3a2e2452c88a9787be84cabd5_proc_1449035/experiment_data.npy"
  • Ablation -- Relation-Type Shuffling:
       "experiment_results/experiment_c88de413f592476187b4cfd15347fe73_proc_1449034/experiment_data.npy"
  • Ablation -- Combined-Token-Embedding:
       "experiment_results/experiment_f960f6f05b874fc888bf0932204136ba_proc_1449037/experiment_data.npy"

All plots use a larger-than-default font size for clarity.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Use a larger font size
plt.rcParams.update({"font.size": 14})

# Create final figures directory
os.makedirs("figures", exist_ok=True)

# Helper function to remove top/right spines
def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis='both', labelsize=12)

# -----------------------------------------------------------------------------
# 1. Baseline: Loss Curves Across Learning Rates (from Baseline summary)
baseline_path = "experiment_results/experiment_869bad450d6e4d8fb3f6149546f7258c_proc_1441297/experiment_data.npy"
try:
    data = np.load(baseline_path, allow_pickle=True).item()
    # Data is under key "learning_rate" -> "SPR"
    runs = data["learning_rate"]["SPR"]
    # sort learning rates by numeric value
    lrs = sorted(runs.keys(), key=lambda x: float(x.replace("e-","e-0")))
    # Epochs: assuming all runs have same number of epochs
    epochs = range(1, len(next(iter(runs.values()))["losses"]["train"]) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    for lr in lrs:
        run = runs[lr]
        # Plot training loss in left subplot
        axes[0].plot(epochs, run["losses"]["train"], label=f"LR={lr}")
        # Plot validation loss in right subplot
        axes[1].plot(epochs, run["losses"]["val"], label=f"LR={lr}")
    
    axes[0].set_title("Train Loss (Baseline)")
    axes[1].set_title("Validation Loss (Baseline)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        style_ax(ax)
        ax.legend()
    fig.suptitle("Baseline: SPR Loss Curves Across Learning Rates", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    outpath = os.path.join("figures", "SPR_loss_curves_baseline.png")
    plt.savefig(outpath)
    plt.close(fig)
    print(f"Saved Baseline loss curves to {outpath}")
except Exception as e:
    print(f"Error creating Baseline loss curves: {e}")

# -----------------------------------------------------------------------------
# 2. Baseline: Accuracy vs Complexity-Aware Accuracy
try:
    # Using same baseline file and data as above.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    for lr in lrs:
        run = runs[lr]
        # extract validation metrics lists from each epoch for "acc" and "caa"
        acc = [m["acc"] for m in run["metrics"]["val"]]
        caa = [m["caa"] for m in run["metrics"]["val"]]
        axes[0].plot(epochs, acc, label=f"LR={lr}")
        axes[1].plot(epochs, caa, label=f"LR={lr}")
    
    axes[0].set_title("Accuracy (Baseline)")
    axes[1].set_title("Complexity-Aware Accuracy (Baseline)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        style_ax(ax)
        ax.legend()
    fig.suptitle("Baseline: Accuracy vs. Complexity-Aware Accuracy", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    outpath = os.path.join("figures", "SPR_accuracy_complexity_baseline.png")
    plt.savefig(outpath)
    plt.close(fig)
    print(f"Saved Baseline accuracy curves to {outpath}")
except Exception as e:
    print(f"Error creating Baseline accuracy curves: {e}")

# -----------------------------------------------------------------------------
# 3. Baseline: Color vs Shape Weighted Accuracies
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    for lr in lrs:
        run = runs[lr]
        cwa = [m["cwa"] for m in run["metrics"]["val"]]
        swa = [m["swa"] for m in run["metrics"]["val"]]
        axes[0].plot(epochs, cwa, label=f"LR={lr}")
        axes[1].plot(epochs, swa, label=f"LR={lr}")
    axes[0].set_title("Color-Weighted Accuracy (Baseline)")
    axes[1].set_title("Shape-Weighted Accuracy (Baseline)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        style_ax(ax)
        ax.legend()
    fig.suptitle("Baseline: Color vs. Shape Weighted Accuracies", fontsize=16)
    fig.tight_layout(rect=[0,0,1,0.95])
    outpath = os.path.join("figures", "SPR_cwa_swa_baseline.png")
    plt.savefig(outpath)
    plt.close(fig)
    print(f"Saved Baseline weighted accuracy curves to {outpath}")
except Exception as e:
    print(f"Error creating Baseline weighted accuracy plot: {e}")

# -----------------------------------------------------------------------------
# 4. Research: Loss Curves (Train vs Val)
research_path = "experiment_results/experiment_8c3bdfc4c3064befa8eae50a48f257bd_proc_1445298/experiment_data.npy"
try:
    data_r = np.load(research_path, allow_pickle=True).item()
    run = data_r["SPR"]
    epochs_list = run["epochs"]
    
    fig, ax = plt.subplots(figsize=(8,6), dpi=300)
    ax.plot(epochs_list, run["losses"]["train"], label="Train Loss")
    ax.plot(epochs_list, run["losses"]["val"], label="Validation Loss", color="orange")
    ax.set_title("Research: Training vs. Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    style_ax(ax)
    ax.legend()
    fig.tight_layout()
    outpath = os.path.join("figures", "SPR_loss_curves_research.png")
    plt.savefig(outpath)
    plt.close(fig)
    print(f"Saved Research loss curves to {outpath}")
except Exception as e:
    print(f"Error creating Research loss curves: {e}")

# -----------------------------------------------------------------------------
# 5. Research: Weighted Accuracies (CWA, SWA, HPA)
try:
    fig, axes = plt.subplots(1, 2, figsize=(12,6), dpi=300)
    # Left: CWA and SWA in one subplot
    cwa = [m["CWA"] for m in run["metrics"]["val"]]
    swa = [m["SWA"] for m in run["metrics"]["val"]]
    axes[0].plot(epochs_list, cwa, label="CWA")
    axes[0].plot(epochs_list, swa, label="SWA")
    axes[0].set_title("CWA & SWA")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Score")
    style_ax(axes[0])
    axes[0].legend()
    # Right: HPA
    hpa = [m["HPA"] for m in run["metrics"]["val"]]
    axes[1].plot(epochs_list, hpa, marker="o", color="green", label="HPA")
    axes[1].set_title("Harmonic Poly Accuracy (HPA)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    style_ax(axes[1])
    axes[1].legend()
    fig.suptitle("Research: Validation Weighted Accuracies", fontsize=16)
    fig.tight_layout(rect=[0,0,1,0.92])
    outpath = os.path.join("figures", "SPR_weighted_accuracy_research.png")
    plt.savefig(outpath)
    plt.close(fig)
    print(f"Saved Research weighted accuracies to {outpath}")
except Exception as e:
    print(f"Error creating Research weighted accuracies plot: {e}")

# -----------------------------------------------------------------------------
# 6. Research: Test Label Distribution
try:
    fig, ax = plt.subplots(figsize=(8,6), dpi=300)
    preds = np.array(run["predictions"])
    gts = np.array(run["ground_truth"])
    labels = sorted(list(set(gts)))
    gt_counts = [int((gts == l).sum()) for l in labels]
    pred_counts = [int((preds == l).sum()) for l in labels]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, gt_counts, width, label="Ground Truth")
    ax.bar(x + width/2, pred_counts, width, label="Predictions")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Class Label")
    ax.set_ylabel("Count")
    ax.set_title("Research: Test Label Distribution")
    style_ax(ax)
    ax.legend()
    fig.tight_layout()
    outpath = os.path.join("figures", "SPR_test_label_distribution_research.png")
    plt.savefig(outpath)
    plt.close(fig)
    print(f"Saved Research test label distribution to {outpath}")
except Exception as e:
    print(f"Error creating Research test label distribution plot: {e}")

# -----------------------------------------------------------------------------
# 7. Ablation: No-Homophily-Edges (Sequential-only Graph)
no_homo_path = "experiment_results/experiment_fd641770149e403ab958308bd2c06a11_proc_1449034/experiment_data.npy"
try:
    data_nohomo = np.load(no_homo_path, allow_pickle=True).item()
    run_nohomo = data_nohomo["no_homophily_edges"]["SPR"]
    epochs_no = run_nohomo["epochs"]
    # Prepare figure with 2 subplots (vertical)
    fig, axes = plt.subplots(2, 1, figsize=(8,10), dpi=300)
    # Top: loss curve
    axes[0].plot(epochs_no, run_nohomo["losses"]["train"], label="Train Loss")
    axes[0].plot(epochs_no, run_nohomo["losses"]["val"], label="Validation Loss")
    axes[0].set_title("No-Homophily-Edges: Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    style_ax(axes[0])
    axes[0].legend()
    # Bottom: validation metrics (CWA, SWA, HPA)
    cwa_no = [m["CWA"] for m in run_nohomo["metrics"]["val"]]
    swa_no = [m["SWA"] for m in run_nohomo["metrics"]["val"]]
    hpa_no = [m["HPA"] for m in run_nohomo["metrics"]["val"]]
    axes[1].plot(epochs_no, cwa_no, label="CWA")
    axes[1].plot(epochs_no, swa_no, label="SWA")
    axes[1].plot(epochs_no, hpa_no, label="HPA")
    axes[1].set_title("No-Homophily-Edges: Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    style_ax(axes[1])
    axes[1].legend()
    fig.suptitle("Ablation: No-Homophily-Edges", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    outpath = os.path.join("figures", "SPR_no_homophily_edges.png")
    plt.savefig(outpath)
    plt.close(fig)
    print(f"Saved No-Homophily-Edges figure to {outpath}")
except Exception as e:
    print(f"Error creating No-Homophily-Edges plot: {e}")

# -----------------------------------------------------------------------------
# 8. Ablation: No-Sequential-Edges (Homophily-only Graph)
no_seq_path = "experiment_results/experiment_9d48e2ca1b0249de9471614b485b36f3_proc_1449035/experiment_data.npy"
try:
    data_noseq = np.load(no_seq_path, allow_pickle=True).item()
    run_noseq = data_noseq["no_seq_edges"]["SPR"]
    epochs_ns = run_noseq["epochs"]
    # Create one figure with 3 subplots in one row
    fig, axes = plt.subplots(1, 3, figsize=(18,5), dpi=300)
    # Subplot 1: Loss curves
    axes[0].plot(epochs_ns, run_noseq["losses"]["train"], label="Train Loss")
    axes[0].plot(epochs_ns, run_noseq["losses"]["val"], label="Val Loss")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    style_ax(axes[0])
    axes[0].legend()
    # Subplot 2: Metrics curves (CWA, SWA, HPA)
    cwa_ns = [m["CWA"] for m in run_noseq["metrics"]["val"]]
    swa_ns = [m["SWA"] for m in run_noseq["metrics"]["val"]]
    hpa_ns = [m["HPA"] for m in run_noseq["metrics"]["val"]]
    axes[1].plot(epochs_ns, cwa_ns, label="CWA")
    axes[1].plot(epochs_ns, swa_ns, label="SWA")
    axes[1].plot(epochs_ns, hpa_ns, label="HPA")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    style_ax(axes[1])
    axes[1].legend()
    # Subplot 3: Confusion Matrix (Test set)
    preds_ns = np.array(run_noseq["predictions"])
    gts_ns = np.array(run_noseq["ground_truth"])
    num_cls_ns = int(max(gts_ns.max(), preds_ns.max()) + 1) if len(gts_ns) > 0 else 2
    cm_ns = np.zeros((num_cls_ns, num_cls_ns), dtype=int)
    for t, p in zip(gts_ns, preds_ns):
        cm_ns[t, p] += 1
    im = axes[2].imshow(cm_ns, cmap="Blues")
    axes[2].set_title("Confusion Matrix")
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("True")
    style_ax(axes[2])
    for i in range(num_cls_ns):
        for j in range(num_cls_ns):
            axes[2].text(j, i, cm_ns[i,j], ha="center", va="center", color="black")
    fig.suptitle("Ablation: No-Sequential-Edges", fontsize=16)
    fig.tight_layout(rect=[0,0,1,0.95])
    outpath = os.path.join("figures", "SPR_no_seq_edges.png")
    plt.savefig(outpath)
    plt.close(fig)
    print(f"Saved No-Sequential-Edges figure to {outpath}")
except Exception as e:
    print(f"Error creating No-Sequential-Edges plot: {e}")

# -----------------------------------------------------------------------------
# 9. Ablation: Edge-Type-Collapsed Graph (Single-Relation RGCN)
etc_path = "experiment_results/experiment_4f07a227d0ea43019a5e283703f49375_proc_1449036/experiment_data.npy"
try:
    data_etc = np.load(etc_path, allow_pickle=True).item()
    run_etc = data_etc["edge_type_collapsed"]["SPR"]
    epochs_etc = run_etc.get("epochs", [])
    fig, axes = plt.subplots(1, 3, figsize=(18,5), dpi=300)
    # Subplot 1: Loss Curves
    axes[0].plot(epochs_etc, run_etc["losses"]["train"], label="Train Loss")
    axes[0].plot(epochs_etc, run_etc["losses"]["val"], label="Val Loss")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    style_ax(axes[0])
    axes[0].legend()
    # Subplot 2: Metrics curves
    cwa_etc = [m["CWA"] for m in run_etc["metrics"]["val"]]
    swa_etc = [m["SWA"] for m in run_etc["metrics"]["val"]]
    hpa_etc = [m["HPA"] for m in run_etc["metrics"]["val"]]
    axes[1].plot(epochs_etc, cwa_etc, label="CWA")
    axes[1].plot(epochs_etc, swa_etc, label="SWA")
    axes[1].plot(epochs_etc, hpa_etc, label="HPA")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    style_ax(axes[1])
    axes[1].legend()
    # Subplot 3: Confusion Matrix
    preds_etc = np.array(run_etc["predictions"])
    gts_etc = np.array(run_etc["ground_truth"])
    num_cls_etc = len(set(gts_etc) | set(preds_etc))
    cm_etc = np.zeros((num_cls_etc, num_cls_etc), dtype=int)
    for t, p in zip(gts_etc, preds_etc):
        cm_etc[t, p] += 1
    im = axes[2].imshow(cm_etc, cmap="Blues")
    axes[2].set_title("Confusion Matrix")
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("True")
    style_ax(axes[2])
    for i in range(num_cls_etc):
        for j in range(num_cls_etc):
            axes[2].text(j, i, cm_etc[i, j], ha="center", va="center", color="black")
    fig.suptitle("Ablation: Edge-Type-Collapsed Graph", fontsize=16)
    fig.tight_layout(rect=[0,0,1,0.93])
    outpath = os.path.join("figures", "SPR_edge_type_collapsed.png")
    plt.savefig(outpath)
    plt.close(fig)
    print(f"Saved Edge-Type-Collapsed Graph figure to {outpath}")
except Exception as e:
    print(f"Error creating Edge-Type-Collapsed Graph plot: {e}")

# -----------------------------------------------------------------------------
# 10. Ablation: Multi-Synthetic-Dataset Evaluation (Cross-Task HPA Heatmap)
multi_syn_path = "experiment_results/experiment_98de24e3a2e2452c88a9787be84cabd5_proc_1449035/experiment_data.npy"
try:
    data_multi = np.load(multi_syn_path, allow_pickle=True).item()
    # Use key "MultiSynthetic" containing synthetic dataset tasks (rules)
    ds = data_multi["MultiSynthetic"]
    rules = list(ds.keys())
    # Build cross-task HPA matrix: rows: train rule, cols: test rule
    hpa_mat = np.zeros((len(rules), len(rules)))
    for i, src in enumerate(rules):
        for j, tgt in enumerate(rules):
            hpa_mat[i, j] = ds[src]["cross_test"][tgt]["HPA"]
    fig, ax = plt.subplots(figsize=(6,5), dpi=300)
    im = ax.imshow(hpa_mat, vmin=0, vmax=1, cmap="viridis")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(rules)))
    ax.set_xticklabels(rules, rotation=45, ha="right")
    ax.set_yticks(range(len(rules)))
    ax.set_yticklabels(rules)
    ax.set_title("MultiSynthetic: Cross-Task HPA")
    style_ax(ax)
    # Annotate each cell
    for i in range(len(rules)):
        for j in range(len(rules)):
            txt_color = "w" if hpa_mat[i, j] < 0.5 else "black"
            ax.text(j, i, f"{hpa_mat[i,j]:.2f}", ha="center", va="center", color=txt_color, fontsize=12)
    fig.tight_layout()
    outpath = os.path.join("figures", "MultiSynthetic_cross_task_HPA_heatmap.png")
    plt.savefig(outpath)
    plt.close(fig)
    print(f"Saved MultiSynthetic cross-task HPA heatmap to {outpath}")
except Exception as e:
    print(f"Error creating MultiSynthetic cross-task HPA heatmap: {e}")

# -----------------------------------------------------------------------------
# 11. Ablation: Relation-Type Shuffling (Edge-Type-Shuffled Graph)
edge_shuffled_path = "experiment_results/experiment_c88de413f592476187b4cfd15347fe73_proc_1449034/experiment_data.npy"
try:
    data_shuffle = np.load(edge_shuffled_path, allow_pickle=True).item()
    run_shuffle = data_shuffle["edge_type_shuffled"]["SPR"]
    epochs_shuffle = np.array(run_shuffle["epochs"])
    # Create a figure with 2 subplots side-by-side: left for HPA curve, right for test accuracy bar.
    fig, axes = plt.subplots(1, 2, figsize=(12,5), dpi=300)
    # Left: HPA curve
    hpa_shuf = np.array([m["HPA"] for m in run_shuffle["metrics"]["val"]])
    axes[0].plot(epochs_shuffle, hpa_shuf, marker="o", color="purple", label="HPA")
    axes[0].set_title("HPA over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("HPA")
    style_ax(axes[0])
    axes[0].legend()
    # Right: Test Accuracy Bar
    preds_shuf = np.array(run_shuffle["predictions"])
    gts_shuf = np.array(run_shuffle["ground_truth"])
    test_acc_shuf = (preds_shuf == gts_shuf).mean() if preds_shuf.size > 0 else 0.0
    axes[1].bar(["Test Accuracy"], [test_acc_shuf], color="tab:blue")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Test Accuracy")
    style_ax(axes[1])
    fig.suptitle("Ablation: Relation-Type Shuffling", fontsize=16)
    fig.tight_layout(rect=[0,0,1,0.92])
    outpath = os.path.join("figures", "SPR_edge_type_shuffled.png")
    plt.savefig(outpath)
    plt.close(fig)
    print(f"Saved Relation-Type Shuffling figure to {outpath}")
except Exception as e:
    print(f"Error creating Relation-Type Shuffling plot: {e}")

# -----------------------------------------------------------------------------
# 12. Ablation: Combined-Token-Embedding (No Factorization)
combined_emb_path = "experiment_results/experiment_f960f6f05b874fc888bf0932204136ba_proc_1449037/experiment_data.npy"
try:
    data_comb = np.load(combined_emb_path, allow_pickle=True).item()
    # Assuming the run is stored under the top-level key for this ablation.
    # Use the first available key in the dictionary.
    model_key = next(iter(data_comb.keys()))
    run_comb = data_comb[model_key]["SPR"]
    epochs_comb = run_comb["epochs"]
    
    fig, axes = plt.subplots(1, 3, figsize=(18,5), dpi=300)
    # Subplot 1: Loss curves
    axes[0].plot(epochs_comb, run_comb["losses"]["train"], label="Train Loss")
    axes[0].plot(epochs_comb, run_comb["losses"]["val"], label="Validation Loss")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    style_ax(axes[0])
    axes[0].legend()
    # Subplot 2: Validation Metrics (CWA, SWA, HPA)
    cwa_comb = [d["CWA"] for d in run_comb["metrics"]["val"]]
    swa_comb = [d["SWA"] for d in run_comb["metrics"]["val"]]
    hpa_comb = [d["HPA"] for d in run_comb["metrics"]["val"]]
    axes[1].plot(epochs_comb, cwa_comb, label="CWA")
    axes[1].plot(epochs_comb, swa_comb, label="SWA")
    axes[1].plot(epochs_comb, hpa_comb, label="HPA")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    style_ax(axes[1])
    axes[1].legend()
    # Subplot 3: Confusion Matrix (Test)
    preds_comb = np.array(run_comb["predictions"])
    gts_comb = np.array(run_comb["ground_truth"])
    num_cls_comb = len(set(gts_comb) | set(preds_comb))
    cm_comb = np.zeros((num_cls_comb, num_cls_comb), dtype=int)
    for t, p in zip(gts_comb, preds_comb):
        cm_comb[t, p] += 1
    im = axes[2].imshow(cm_comb, cmap="Blues")
    axes[2].set_title("Confusion Matrix")
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("True")
    style_ax(axes[2])
    for i in range(num_cls_comb):
        for j in range(num_cls_comb):
            axes[2].text(j, i, cm_comb[i,j], ha="center", va="center", color="black")
    fig.suptitle("Ablation: Combined-Token-Embedding", fontsize=16)
    fig.tight_layout(rect=[0,0,1,0.93])
    outpath = os.path.join("figures", "SPR_combined_token_embedding.png")
    plt.savefig(outpath)
    plt.close(fig)
    print(f"Saved Combined-Token-Embedding figure to {outpath}")
except Exception as e:
    print(f"Error creating Combined-Token-Embedding plot: {e}")

print("All done. Final figures are saved in the 'figures/' directory.")
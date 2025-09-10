#!/usr/bin/env python3
"""
Final Aggregator Script for GNN_for_SPR Experiments
This script loads the final experiment .npy data and produces a comprehensive set
of final scientific plots saved in the "figures/" directory.
Each plot is wrapped in a try-except block so that failure in one does not block others.
The figures are designed for publication with increased font sizes and professional styling.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

plt.rcParams.update({
    'font.size': 16,
    'axes.spines.top': False,
    'axes.spines.right': False
})

os.makedirs("figures", exist_ok=True)

def load_data(file_path):
    try:
        data = np.load(file_path, allow_pickle=True).item()
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# -----------------------------------------------------------------------
# Define experiment file paths (full and exact as given)
baseline_file = "experiment_results/experiment_47573acc32864df7977c69d34068000b_proc_1476161/experiment_data.npy"
research_file = "experiment_results/experiment_c44ac6235757449b9376cf3b359e3cbd_proc_1480340/experiment_data.npy"
no_color_file = "experiment_results/experiment_f3f6cab0a2fa4c74b5daf65c62421cfe_proc_1483402/experiment_data.npy"
no_pos_file = "experiment_results/experiment_dbcce41a9918415eb742dc22c6965bed_proc_1480343/experiment_data.npy"
no_shape_file = "experiment_results/experiment_eef5b18e62a2460690ea8e27dbf1fabc_proc_1483405/experiment_data.npy"
collapsed_edge_file = "experiment_results/experiment_fe1a788c93194b63a8836d263df471ce_proc_1483403/experiment_data.npy"
uniform_file = "experiment_results/experiment_2956e885a67b4f9bbbfaedd997f17fe0_proc_1483405/experiment_data.npy"

# Load each experiment data dict
baseline_data = load_data(baseline_file)        # structure: {"epochs": {epoch_str: {losses:{train, val}, metrics:{val_cwa2: [...]}}} }
research_data = load_data(research_file)          # structure: {"SPR": {losses:{train, val}, metrics: {val:[{CWA, SWA, CompWA},...], test:{...}}}}
no_color_data = load_data(no_color_file)          # structure: { "SPR": {losses:{train, val}, metrics:{val:[...], test:{...}} } }
no_pos_data   = load_data(no_pos_file)            # structure: { "SPR": {losses:{train, val}, metrics:{val:[...], test:{...}}, ground_truth: [...], predictions: [...] } }
no_shape_data = load_data(no_shape_file)          # structure: { "SPR": {losses:{train, val}, metrics:{val:[...], test:{...}}, ground_truth: [...], predictions: [...] } }
collapsed_edge_data = load_data(collapsed_edge_file)  # structure: { "SPR": {losses:{train, val}, metrics:{val:[...], test:{...}} } }
uniform_data = load_data(uniform_file)            # structure: { "baseline": {"SPR": {...}}, "uniform_node_feature": {"SPR": {...}} }

# -----------------------------------------------------------------------
# Figure 1: Baseline Loss Curves for Different Epoch Budgets (2x2 subplots)
try:
    exp = baseline_data.get("epochs", {})
    epochs_keys = sorted(exp.keys(), key=lambda x: int(x))
    n_plots = len(epochs_keys)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    for i, key in enumerate(epochs_keys):
        res = exp[key]
        train_loss = res.get("losses", {}).get("train", [])
        val_loss = res.get("losses", {}).get("val", [])
        epochs_range = range(1, len(train_loss)+1)
        axes[i].plot(epochs_range, train_loss, label="Train Loss", marker="o")
        axes[i].plot(epochs_range, val_loss, label="Val Loss", marker="s")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Cross-Entropy Loss")
        axes[i].set_title(f"Loss Curves ({key} Epochs)")
        axes[i].legend()
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "Baseline_Loss_Curves.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print(f"Error creating Baseline Loss Curves: {e}")

# -----------------------------------------------------------------------
# Figure 2: Baseline Aggregated Validation CWA Curves
try:
    fig, ax = plt.subplots(figsize=(10, 8))
    for key in sorted(exp.keys(), key=lambda x: int(x)):
        res = exp[key]
        cwa = res.get("metrics", {}).get("val_cwa2", [])
        epochs_range = range(1, len(cwa) + 1)
        ax.plot(epochs_range, cwa, marker="o", label=f"{key} Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Complexity-Weighted Accuracy")
    ax.set_title("Baseline: Validation CWA Across Epoch Budgets")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "Baseline_Validation_CWA.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print(f"Error creating Baseline Aggregated CWA Curves: {e}")

# -----------------------------------------------------------------------
# Figure 3: Research - Training vs Validation Loss Curves
try:
    spr_exp = research_data.get("SPR", {})
    tr_loss = spr_exp.get("losses", {}).get("train", [])
    val_loss = spr_exp.get("losses", {}).get("val", [])
    epochs_range = range(1, len(tr_loss)+1)
    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(epochs_range, tr_loss, label="Train Loss", marker="o")
    ax.plot(epochs_range, val_loss, label="Val Loss", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Research: Training vs Validation Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "Research_Loss_Curves.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print(f"Error creating Research Loss Curves: {e}")

# -----------------------------------------------------------------------
# Figure 4: Research - Validation Weighted Accuracy Curves (CWA, SWA, CompWA)
try:
    val_metrics = spr_exp.get("metrics", {}).get("val", [])
    cwa = [m.get("CWA", 0) for m in val_metrics]
    swa = [m.get("SWA", 0) for m in val_metrics]
    comp = [m.get("CompWA", 0) for m in val_metrics]
    epochs_range = range(1, len(cwa)+1)
    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(epochs_range, cwa, label="CWA", marker="o")
    ax.plot(epochs_range, swa, label="SWA", marker="s")
    ax.plot(epochs_range, comp, label="CompWA", marker="^")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted Accuracy")
    ax.set_title("Research: Validation Weighted Accuracies")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "Research_Validation_Accuracies.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print(f"Error creating Research Validation Accuracy Curves: {e}")

# -----------------------------------------------------------------------
# Figure 5: Research - Test Weighted Accuracy Bar Chart
try:
    test_metrics = spr_exp.get("metrics", {}).get("test", {})
    labels = list(test_metrics.keys())
    values = [test_metrics[k] for k in labels]
    fig, ax = plt.subplots(figsize=(8,6))
    ax.bar(labels, values, color=["tab:blue", "tab:orange", "tab:green"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Weighted Accuracy")
    ax.set_title("Research: Final Test Weighted Accuracies")
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "Research_Test_Accuracies.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print(f"Error creating Research Test Accuracy Bar Chart: {e}")

# -----------------------------------------------------------------------
# Figure 6: Research - Overlaid (Zoomed) Loss Curves
try:
    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(epochs_range, tr_loss, linestyle="--", marker="o", label="Train Loss")
    ax.plot(epochs_range, val_loss, linestyle="-", marker="s", label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Research: Overlaid Loss Curves (Zoom View)")
    ax.legend()
    # Set y-limits based on losses (with a 10% margin)
    ymin = min(val_loss) * 0.9 if len(val_loss) > 0 else 0
    ymax = max(tr_loss) * 1.1 if len(tr_loss) > 0 else 1
    ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "Research_Loss_Overlay_Zoom.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print(f"Error creating Research Overlaid Loss Plot: {e}")

# -----------------------------------------------------------------------
# Figure 7: Ablation - No-Color-Edges (One row, three subplots)
try:
    spr_nc = no_color_data.get("SPR", {})
    # Loss curves
    nc_train = spr_nc.get("losses", {}).get("train", [])
    nc_val = spr_nc.get("losses", {}).get("val", [])
    ep_nc = range(1, len(nc_train)+1)
    # Validation metrics (assumed structure similar to others)
    val_nc = spr_nc.get("metrics", {}).get("val", [])
    nc_cwa = [d.get("CWA", 0) for d in val_nc]
    nc_swa = [d.get("SWA", 0) for d in val_nc]
    nc_comp = [d.get("CompWA", 0) for d in val_nc]
    # Test metrics
    test_nc = spr_nc.get("metrics", {}).get("test", {})
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    # Subplot 1: Loss curves
    axes[0].plot(ep_nc, nc_train, label="Train", marker="o")
    axes[0].plot(ep_nc, nc_val, label="Val", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("No-Color-Edges: Loss Curves")
    axes[0].legend()
    # Subplot 2: Validation Metrics Curve
    axes[1].plot(ep_nc, nc_cwa, label="CWA", marker="o")
    axes[1].plot(ep_nc, nc_swa, label="SWA", marker="s")
    axes[1].plot(ep_nc, nc_comp, label="CompWA", marker="^")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Weighted Accuracy")
    axes[1].set_title("No-Color-Edges: Validation Metrics")
    axes[1].legend()
    # Subplot 3: Test Metrics Bar Chart
    labels_nc = list(test_nc.keys())
    vals_nc = [test_nc[k] for k in labels_nc]
    axes[2].bar(labels_nc, vals_nc, color=["tab:blue", "tab:orange", "tab:green"])
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Weighted Accuracy")
    axes[2].set_title("No-Color-Edges: Test Metrics")
    for i, v in enumerate(vals_nc):
        axes[2].text(i, v + 0.02, f"{v:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "Ablation_NoColorEdges_Combined.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print(f"Error creating Ablation (No-Color-Edges) figure: {e}")

# -----------------------------------------------------------------------
# Figure 8: Ablation - No-Positional-Embedding (One row, three subplots)
try:
    spr_np = no_pos_data.get("SPR", {})
    np_train = np.array(spr_np.get("losses", {}).get("train", []))
    np_val = np.array(spr_np.get("losses", {}).get("val", []))
    ep_np = range(1, len(np_train)+1)
    val_np = spr_np.get("metrics", {}).get("val", [])
    np_cwa = [d.get("CWA", 0) for d in val_np]
    np_swa = [d.get("SWA", 0) for d in val_np]
    np_comp = [d.get("CompWA", 0) for d in val_np]
    test_np = spr_np.get("metrics", {}).get("test", {})
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    # Loss curves
    axes[0].plot(ep_np, np_train, label="Train", marker="o")
    axes[0].plot(ep_np, np_val, label="Val", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("No-Positional-Embedding: Loss Curves")
    axes[0].legend()
    # Validation metrics
    axes[1].plot(ep_np, np_cwa, label="CWA", marker="o")
    axes[1].plot(ep_np, np_swa, label="SWA", marker="s")
    axes[1].plot(ep_np, np_comp, label="CompWA", marker="^")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Weighted Accuracy")
    axes[1].set_title("No-Positional-Embedding: Validation Metrics")
    axes[1].legend()
    # Test metrics bar chart
    labels_np = list(test_np.keys())
    vals_np = [test_np[k] for k in labels_np]
    axes[2].bar(labels_np, vals_np, color=["tab:blue", "tab:orange", "tab:green"])
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Weighted Accuracy")
    axes[2].set_title("No-Positional-Embedding: Test Metrics")
    for i, v in enumerate(vals_np):
        axes[2].text(i, v + 0.02, f"{v:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "Ablation_NoPositionalEmbedding_Combined.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print(f"Error creating Ablation (No-Positional-Embedding) figure: {e}")

# -----------------------------------------------------------------------
# Figure 9: Ablation - No-Shape-Edges (2x2 subplots)
try:
    spr_ns = no_shape_data.get("SPR", {})
    ns_train = spr_ns.get("losses", {}).get("train", [])
    ns_val = spr_ns.get("losses", {}).get("val", [])
    ep_ns = range(1, len(ns_train)+1)
    val_ns = spr_ns.get("metrics", {}).get("val", [])
    ns_cwa = [d.get("CWA", 0) for d in val_ns]
    ns_swa = [d.get("SWA", 0) for d in val_ns]
    ns_comp = [d.get("CompWA", 0) for d in val_ns]
    test_ns = spr_ns.get("metrics", {}).get("test", {})
    gt = np.array(spr_ns.get("ground_truth", []))
    pred = np.array(spr_ns.get("predictions", []))
    # Build confusion matrix if ground truth exists
    if gt.size and pred.size:
        num_cls = int(max(np.max(gt), np.max(pred))) + 1
        cm = np.zeros((num_cls, num_cls), dtype=int)
        for t, p in zip(gt, pred):
            cm[int(t), int(p)] += 1
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    # Top-left: Loss curves
    axes[0,0].plot(ep_ns, ns_train, label="Train", marker="o")
    axes[0,0].plot(ep_ns, ns_val, label="Val", marker="s")
    axes[0,0].set_xlabel("Epoch")
    axes[0,0].set_ylabel("Loss")
    axes[0,0].set_title("No-Shape-Edges: Loss Curves")
    axes[0,0].legend()
    # Top-right: Validation Metrics
    axes[0,1].plot(ep_ns, ns_cwa, label="CWA", marker="o")
    axes[0,1].plot(ep_ns, ns_swa, label="SWA", marker="s")
    axes[0,1].plot(ep_ns, ns_comp, label="CompWA", marker="^")
    axes[0,1].set_xlabel("Epoch")
    axes[0,1].set_ylabel("Weighted Accuracy")
    axes[0,1].set_title("No-Shape-Edges: Validation Metrics")
    axes[0,1].legend()
    # Bottom-left: Test Metrics Bar Chart
    labels_ns = list(test_ns.keys())
    vals_ns = [test_ns[k] for k in labels_ns]
    axes[1,0].bar(labels_ns, vals_ns, color=["tab:blue", "tab:orange", "tab:green"])
    axes[1,0].set_ylim(0, 1)
    axes[1,0].set_ylabel("Weighted Accuracy")
    axes[1,0].set_title("No-Shape-Edges: Test Metrics")
    for i, v in enumerate(vals_ns):
        axes[1,0].text(i, v + 0.02, f"{v:.2f}", ha="center")
    # Bottom-right: Confusion Matrix
    im = axes[1,1].imshow(cm, cmap="Blues")
    axes[1,1].set_xlabel("Predicted")
    axes[1,1].set_ylabel("Ground Truth")
    axes[1,1].set_title("No-Shape-Edges: Confusion Matrix")
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        axes[1,1].text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im, ax=axes[1,1])
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "Ablation_NoShapeEdges_Combined.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print(f"Error creating Ablation (No-Shape-Edges) figure: {e}")

# -----------------------------------------------------------------------
# Figure 10: Ablation - Collapsed-Edge-Type (One row, three subplots)
try:
    spr_ce = collapsed_edge_data.get("SPR", {})
    ce_train = spr_ce.get("losses", {}).get("train", [])
    ce_val = spr_ce.get("losses", {}).get("val", [])
    ep_ce = range(1, len(ce_train)+1)
    val_ce = spr_ce.get("metrics", {}).get("val", [])
    ce_cwa = [d.get("CWA", 0) for d in val_ce]
    ce_swa = [d.get("SWA", 0) for d in val_ce]
    ce_comp = [d.get("CompWA", 0) for d in val_ce]
    test_ce = spr_ce.get("metrics", {}).get("test", {})
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    # Loss curves
    axes[0].plot(ep_ce, ce_train, label="Train", marker="o")
    axes[0].plot(ep_ce, ce_val, label="Val", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Collapsed-Edge-Type: Loss Curves")
    axes[0].legend()
    # Validation metrics
    axes[1].plot(ep_ce, ce_cwa, label="CWA", marker="o")
    axes[1].plot(ep_ce, ce_swa, label="SWA", marker="s")
    axes[1].plot(ep_ce, ce_comp, label="CompWA", marker="^")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Weighted Accuracy")
    axes[1].set_title("Collapsed-Edge-Type: Validation Metrics")
    axes[1].legend()
    # Test metrics bar chart
    labels_ce = list(test_ce.keys())
    vals_ce = [test_ce[k] for k in labels_ce]
    axes[2].bar(labels_ce, vals_ce, color=["steelblue", "salmon", "seagreen"])
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Weighted Accuracy")
    axes[2].set_title("Collapsed-Edge-Type: Test Metrics")
    for i, v in enumerate(vals_ce):
        axes[2].text(i, v + 0.02, f"{v:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "Ablation_CollapsedEdgeType_Combined.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print(f"Error creating Ablation (Collapsed-Edge-Type) figure: {e}")

# -----------------------------------------------------------------------
# Figure 11: Ablation - Uniform-Node-Feature Comparison
try:
    # uniform_data has two keys: "baseline" and "uniform_node_feature", each with "SPR"
    baseline_uniform = uniform_data.get("baseline", {}).get("SPR", {})
    uniform_nf = uniform_data.get("uniform_node_feature", {}).get("SPR", {})
    # For both, get loss curves and test metrics.
    # Baseline loss
    bl_train = baseline_uniform.get("losses", {}).get("train", [])
    bl_val = baseline_uniform.get("losses", {}).get("val", [])
    ep_bl = range(1, len(bl_train)+1)
    # Uniform loss
    un_train = uniform_nf.get("losses", {}).get("train", [])
    un_val = uniform_nf.get("losses", {}).get("val", [])
    ep_un = range(1, len(un_train)+1)
    # Test metrics
    test_bl = baseline_uniform.get("metrics", {}).get("test", {})
    test_un = uniform_nf.get("metrics", {}).get("test", {})
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    # Subplot 1: Baseline Loss Curve (Uniform Ablation baseline)
    axes[0].plot(ep_bl, bl_train, label="Train", marker="o")
    axes[0].plot(ep_bl, bl_val, label="Val", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Uniform Ablation: Baseline Loss")
    axes[0].legend()
    # Subplot 2: Uniform_Node_Feature Loss Curve
    axes[1].plot(ep_un, un_train, label="Train", marker="o")
    axes[1].plot(ep_un, un_val, label="Val", marker="s")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Uniform Ablation: Uniform_Node_Feature Loss")
    axes[1].legend()
    # Subplot 3: Test Metrics Comparison Bar Chart
    labels_un = ["CWA", "SWA", "CompWA"]
    width = 0.35
    x = np.arange(len(labels_un))
    bl_vals = [test_bl.get(m, 0) for m in labels_un]
    un_vals = [test_un.get(m, 0) for m in labels_un]
    axes[2].bar(x - width/2, bl_vals, width, label="Baseline")
    axes[2].bar(x + width/2, un_vals, width, label="Uniform_Node_Feature")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels_un)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Weighted Accuracy")
    axes[2].set_title("Uniform Ablation: Test Metrics")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "Ablation_UniformNodeFeature_Comparison.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print(f"Error creating Ablation (Uniform-Node-Feature) Comparison figure: {e}")

print("Final plots have been saved in the 'figures/' directory.")
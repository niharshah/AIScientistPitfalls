#!/usr/bin/env python3
"""
Final Aggregator Script for "Zero-Shot Synthetic PolyRule Reasoning with Neural Symbolic Integration"

This script aggregates and visualizes final experiment results from baseline, research, and various ablation studies.
Final published figures are saved in the "figures/" directory.
All data are loaded from existing .npy files using full file paths as specified in the experimental summaries.
Each final figure is wrapped in its own try/except block so that errors in one plot do not prevent the others.
All plots are made with improved font sizes, clear labels, legends, and professional styling.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Set a higher font size for clarity in the final paper.
plt.rcParams.update({'font.size': 14, 'axes.spines.top': False, 'axes.spines.right': False})

# Create final output directory for figures.
os.makedirs("figures", exist_ok=True)


def load_experiment_data(np_file_path):
    """Utility to load experiment data from a given .npy file."""
    try:
        data = np.load(np_file_path, allow_pickle=True).item()
        return data
    except Exception as e:
        print(f"Error loading {np_file_path}: {e}")
        return {}


# =============================================================================
# 1. Baseline/Research Aggregated Figure:
#    Combines accuracy curves, loss curves, and final validation accuracy by hidden-dim
#    from hidden_dim_tuning/SPR_BENCH experiments.
# =============================================================================
try:
    baseline_path = "experiment_results/experiment_2d2ef9d4cf08449ba8e6d7a8f5a36a88_proc_312327/experiment_data.npy"
    baseline_data = load_experiment_data(baseline_path)
    exp = baseline_data.get("hidden_dim_tuning", {}).get("SPR_BENCH", {})
    if not exp:
        raise ValueError("Baseline data for 'hidden_dim_tuning/SPR_BENCH' not found")
    
    # Get sorted hidden-dimensions (keys come as "hidden_64", etc.)
    hidden_keys = sorted(exp.keys(), key=lambda k: int(k.split("_")[-1]))
    # Assume all runs have the same number of epochs.
    sample_run = exp[hidden_keys[0]]
    epochs = list(range(1, len(sample_run["metrics"]["train_acc"]) + 1))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    fig.suptitle("Baseline/Research: SPR_BENCH Performance", fontsize=16)
    
    # Subplot 1: Accuracy Curves (Train & Validation)
    ax = axes[0]
    for key in hidden_keys:
        hd = int(key.split("_")[-1])
        tr_acc = exp[key]["metrics"]["train_acc"]
        val_acc = exp[key]["metrics"]["train_acc"]
        # Plot train accuracy
        ax.plot(epochs, exp[key]["metrics"]["train_acc"], marker="o", label=f"Train (hd {hd})")
        # Plot validation accuracy (if available)
        ax.plot(epochs, exp[key]["metrics"]["val_acc"], marker="x", linestyle="--", label=f"Val (hd {hd})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Curves")
    ax.legend()
    
    # Subplot 2: Loss Curves (Train & Validation)
    ax = axes[1]
    for key in hidden_keys:
        hd = int(key.split("_")[-1])
        ax.plot(epochs, exp[key]["losses"]["train"], marker="o", label=f"Train (hd {hd})")
        ax.plot(epochs, exp[key]["losses"]["val"], marker="x", linestyle="--", label=f"Val (hd {hd})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Loss Curves")
    ax.legend()
    
    # Subplot 3: Final Validation Accuracy (Bar chart)
    ax = axes[2]
    final_acc = [exp[key]["metrics"]["val_acc"][-1] for key in hidden_keys]
    hidden_dims = [int(key.split("_")[-1]) for key in hidden_keys]
    ax.bar([str(hd) for hd in hidden_dims], final_acc, color="skyblue")
    ax.set_xlabel("Hidden Dimension")
    ax.set_ylabel("Final Validation Accuracy")
    ax.set_title("Final Val Accuracy")
    
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = os.path.join("figures", "Baseline_Research_Aggregated.png")
    plt.savefig(out_path)
    print(f"Saved Baseline/Research aggregated figure to {out_path}")
    plt.close(fig)
except Exception as e:
    print(f"Error creating Baseline/Research aggregated figure: {e}")


# =============================================================================
# 2. Ablation: UniDirectional-GRU Aggregated Figure
#    Plots accuracy and loss curves from the unidirectional GRU ablation.
# =============================================================================
try:
    undir_path = "experiment_results/experiment_b79b0948b227485f946b2cfe64539468_proc_319695/experiment_data.npy"
    undir_data = load_experiment_data(undir_path)
    spr_runs = undir_data.get("UniDirectional_GRU", {}).get("SPR_BENCH", {})
    if not spr_runs:
        raise ValueError("UniDirectional_GRU/SPR_BENCH data not found")
    
    hidden_keys = sorted(spr_runs.keys(), key=lambda k: int(k.split("_")[-1]))
    sample_run = spr_runs[hidden_keys[0]]
    epochs = list(range(1, len(sample_run["metrics"]["train_acc"]) + 1))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    fig.suptitle("Ablation: UniDirectional-GRU on SPR_BENCH", fontsize=16)
    
    # Left: Accuracy curves (Train & Validation)
    ax = axes[0]
    for key in hidden_keys:
        hd = key.split("_")[-1]
        metrics = spr_runs[key]["metrics"]
        ax.plot(epochs, metrics["train_acc"], marker="o", linestyle="--", label=f"{key} Train")
        ax.plot(epochs, metrics["val_acc"], marker="x", label=f"{key} Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Curves")
    ax.legend()
    
    # Right: Loss curves (Train & Validation)
    ax = axes[1]
    for key in hidden_keys:
        losses = spr_runs[key]["losses"]
        ax.plot(epochs, losses["train"], marker="o", linestyle="--", label=f"{key} Train")
        ax.plot(epochs, losses["val"], marker="x", label=f"{key} Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves")
    ax.legend()
    
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = os.path.join("figures", "Ablation_UniDirectional_GRU.png")
    plt.savefig(out_path)
    print(f"Saved UniDirectional-GRU aggregated figure to {out_path}")
    plt.close(fig)
except Exception as e:
    print(f"Error creating UniDirectional-GRU aggregated figure: {e}")


# =============================================================================
# 3. Ablation: Frozen-Embedding (No Embedding Fine-Tuning) Figures
#    (a) Aggregated Accuracy and Loss Curves;
#    (b) ZSRTA Bar Chart.
# =============================================================================
try:
    frozen_path = "experiment_results/experiment_90e58e783df640db9bf95c97bd0f9aba_proc_319696/experiment_data.npy"
    frozen_data = load_experiment_data(frozen_path)
    runs = frozen_data.get("frozen_embedding_ablation", {}).get("SPR_BENCH", {})
    if not runs:
        raise ValueError("Frozen-Embedding Ablation data not found")
    
    run_names = sorted(runs.keys())
    sample_run = runs[run_names[0]]
    epochs = list(range(1, len(sample_run["metrics"]["train_acc"]) + 1))
    
    # (a) Accuracy and Loss curves aggregated in one figure with 2 rows.
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), dpi=300)
    fig.suptitle("Ablation: Frozen-Embedding (No Embedding Fine-Tuning)\nAccuracy & Loss Curves", fontsize=16)
    
    # Top: Accuracy curves for each run
    ax = axes[0]
    for run in run_names:
        metrics = runs[run]["metrics"]
        ax.plot(epochs, metrics["train_acc"], marker="o", linestyle="--", label=f"{run} Train")
        ax.plot(epochs, metrics["val_acc"], marker="x", label=f"{run} Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Curves")
    ax.legend()
    
    # Bottom: Loss curves for each run
    ax = axes[1]
    for run in run_names:
        losses = runs[run]["losses"]
        ax.plot(epochs, losses["train"], marker="o", linestyle="--", label=f"{run} Train")
        ax.plot(epochs, losses["val"], marker="x", label=f"{run} Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves")
    ax.legend()
    
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = os.path.join("figures", "Ablation_FrozenEmbedding_Curves.png")
    plt.savefig(out_path)
    print(f"Saved Frozen-Embedding accuracy/loss curves to {out_path}")
    plt.close(fig)
    
    # (b) ZSRTA Bar Chart for Frozen-Embedding Ablation
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    zsrta_vals = []
    for run in run_names:
        metric = runs[run]["metrics"].get("ZSRTA", [])
        val = metric[0] if metric else np.nan
        zsrta_vals.append(val)
    ax.bar(run_names, zsrta_vals, color="salmon")
    ax.set_xlabel("Run")
    ax.set_ylabel("ZSRTA")
    ax.set_title("Frozen-Embedding ZSRTA Comparison")
    ax.set_ylim(0, 1)
    for i, v in enumerate(zsrta_vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom")
    out_path = os.path.join("figures", "Ablation_FrozenEmbedding_ZSRTA.png")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved Frozen-Embedding ZSRTA chart to {out_path}")
    plt.close(fig)
except Exception as e:
    print(f"Error creating Frozen-Embedding Ablation figures: {e}")


# =============================================================================
# 4. Ablation: Multi-Synthetic-Dataset Training
#    Aggregated figure with (a) Training/Validation curves (SPR_BENCH only),
#    (b) Test Accuracy vs Hidden Dimension (for all datasets),
#    (c) SWA and CWA vs Hidden Dimension (for all datasets).
# =============================================================================
try:
    multi_path = "experiment_results/experiment_0e1b852adea34a3196b776afb126ee75_proc_319697/experiment_data.npy"
    multi_data = load_experiment_data(multi_path)
    multi = multi_data.get("multi_dataset_training", {})
    if not multi:
        raise ValueError("Multi-dataset training data not found")
    
    # (a) SPR_BENCH Training/Validation curves
    ds_sp = multi.get("SPR_BENCH", {})
    # Assume each hidden run has a "train_curve" with metrics
    hidden_list = sorted([int(k.split("_")[-1]) for k in ds_sp.keys()])
    
    # Create aggregated figure with 3 subplots.
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=300)
    fig.suptitle("Multi-Synthetic-Dataset Training Ablation", fontsize=16)
    
    # Subplot 1: SPR_BENCH Train & Val Accuracy Curves (from train_curve field)
    ax = axes[0]
    for hd in hidden_list:
        run = ds_sp.get(f"hidden_{hd}", {})
        if run:
            curve = run.get("train_curve", {}).get("metrics", {})
            tr = curve.get("train_acc", [])
            val = curve.get("val_acc", [])
            epochs_curve = list(range(1, len(tr) + 1))
            ax.plot(epochs_curve, tr, marker="o", linestyle="--", label=f"Train hd{hd}")
            ax.plot(epochs_curve, val, marker="x", label=f"Val hd{hd}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("SPR_BENCH Train/Val Curve")
    ax.legend()
    
    # Subplot 2: Test Accuracy vs Hidden Dimension across datasets
    ax = axes[1]
    for ds_name, ds_runs in multi.items():
        hds = sorted([int(k.split("_")[-1]) for k in ds_runs.keys()])
        test_accs = [ds_runs.get(f"hidden_{hd}", {}).get("metrics", {}).get("test_acc", np.nan) for hd in hds]
        ax.plot(hds, test_accs, marker="o", label=ds_name)
    ax.set_xlabel("Hidden Dimension")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Test Accuracy vs Hidden Dimension")
    ax.legend()
    
    # Subplot 3: SWA and CWA vs Hidden Dimension across datasets (combined)
    ax = axes[2]
    for ds_name, ds_runs in multi.items():
        hds = sorted([int(k.split("_")[-1]) for k in ds_runs.keys()])
        swa_vals = [ds_runs.get(f"hidden_{hd}", {}).get("metrics", {}).get("SWA", np.nan) for hd in hds]
        cwa_vals = [ds_runs.get(f"hidden_{hd}", {}).get("metrics", {}).get("CWA", np.nan) for hd in hds]
        ax.plot(hds, swa_vals, marker="o", linestyle="--", label=f"{ds_name} SWA")
        ax.plot(hds, cwa_vals, marker="x", linestyle=":", label=f"{ds_name} CWA")
    ax.set_xlabel("Hidden Dimension")
    ax.set_ylabel("Accuracy Metric")
    ax.set_title("SWA/CWA vs Hidden Dimension")
    ax.legend()
    
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = os.path.join("figures", "Ablation_MultiSynthetic_Dataset.png")
    plt.savefig(out_path)
    print(f"Saved Multi-Synthetic-Dataset Training aggregated figure to {out_path}")
    plt.close(fig)
except Exception as e:
    print(f"Error creating Multi-Synthetic-Dataset Training figure: {e}")


# =============================================================================
# 5. Ablation: No-Length-Masking Figure
#    Aggregates (a) Accuracy curves, (b) Loss curves, and (c) ZSRTA bar chart.
# =============================================================================
try:
    nomask_path = "experiment_results/experiment_7bd39f3e106141618f9c65ae14e1a84b_proc_319694/experiment_data.npy"
    nomask_data = load_experiment_data(nomask_path)
    runs = nomask_data.get("no_length_masking", {}).get("SPR_BENCH", {})
    if not runs:
        raise ValueError("No-Length-Masking data not found")
    
    hidden_keys = sorted(runs.keys(), key=lambda k: int(k.split("_")[-1]))
    # Each run's metrics: list arrays for train_acc, val_acc and losses.
    # Prepare epochs for each run (assuming same length)
    sample_run = runs[hidden_keys[0]]
    epochs = list(range(1, len(sample_run["metrics"]["train_acc"]) + 1))
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), dpi=300)
    fig.suptitle("Ablation: No-Length-Masking on SPR_BENCH", fontsize=16)
    
    # Subplot 1: Training Accuracies (left: train curves; overlay validation curves in same panel)
    ax = axes[0]
    for key in hidden_keys:
        hd = key.split("_")[-1]
        metrics = runs[key]["metrics"]
        ax.plot(epochs, metrics["train_acc"], marker="o", linestyle="--", label=f"{key} Train")
        ax.plot(epochs, metrics["val_acc"], marker="x", label=f"{key} Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Curves")
    ax.legend()
    
    # Subplot 2: Loss Curves
    ax = axes[1]
    for key in hidden_keys:
        losses = runs[key]["losses"]
        ax.plot(epochs, losses["train"], marker="o", linestyle="--", label=f"{key} Train")
        ax.plot(epochs, losses["val"], marker="x", label=f"{key} Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves")
    ax.legend()
    
    # Subplot 3: ZSRTA Bar Chart
    ax = axes[2]
    zsrta_vals = []
    for key in hidden_keys:
        metric = runs[key]["metrics"].get("ZSRTA", [])
        val = metric[0] if metric else np.nan
        zsrta_vals.append(val)
    ax.bar([key.split("_")[-1] for key in hidden_keys], zsrta_vals, color="skyblue")
    ax.set_xlabel("Hidden Dimension")
    ax.set_ylabel("ZSRTA")
    ax.set_title("Zero-Shot Rule Transfer Accuracy")
    ax.set_ylim(0, 1)
    for i, v in enumerate(zsrta_vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=12)
    
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = os.path.join("figures", "Ablation_NoLengthMasking.png")
    plt.savefig(out_path)
    print(f"Saved No-Length-Masking figure to {out_path}")
    plt.close(fig)
except Exception as e:
    print(f"Error creating No-Length-Masking figure: {e}")


print("Final aggregation complete. All figures are saved in the 'figures/' directory.")
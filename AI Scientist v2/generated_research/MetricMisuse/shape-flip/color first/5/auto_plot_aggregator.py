#!/usr/bin/env python3
"""
Final Aggregated Plot Script for GNN for SPR Research Paper
This script loads experiment results from several .npy files (baseline, research, and ablation)
and creates publication‐ready figures. Each figure is saved to the "figures/" directory.
Every plot is wrapped in its own try‐except block such that a failure in one does not stop the rest.
All figures use an increased font size for readability.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Increase font size for publication-quality figures.
plt.rcParams.update({'font.size': 14, 'axes.spines.top': False, 'axes.spines.right': False})

# Create figures directory
os.makedirs("figures", exist_ok=True)

#######################################
#  Baseline Experiments (3 figures)  #
#######################################

# Baseline file path (exact as in summary)
BASELINE_FILE = "experiment_results/experiment_adbff8ff9dae4746884a8fc62cf92e20_proc_1490514/experiment_data.npy"

# Plot 1: Baseline Loss Curves
try:
    data = np.load(BASELINE_FILE, allow_pickle=True).item()
    # Expected structure: data["num_epochs"]["SPR"] is a dict with keys like "epochs_5", "epochs_20", etc.
    runs = data.get("num_epochs", {}).get("SPR", {})
    plt.figure(figsize=(8,6))
    for k, hist in runs.items():
        epochs = hist.get("epochs", [])
        # Plot training losses (solid) and validation losses (dashed)
        plt.plot(epochs, hist.get("losses", {}).get("train", []), label=f"Train {k.split('_')[-1]}")
        plt.plot(epochs, hist.get("losses", {}).get("val", []), "--", label=f"Val {k.split('_')[-1]}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Baseline: Training vs Validation Loss (SPR)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Baseline_Loss_Curves.png"), dpi=300)
    plt.close()
except Exception as e:
    print("Error in Baseline Loss Curves:", e)
    plt.close()

# Plot 2: Baseline Accuracy Curves (Complexity-Weighted Accuracy)
try:
    plt.figure(figsize=(8,6))
    for k, hist in runs.items():
        epochs = hist.get("epochs", [])
        plt.plot(epochs, hist.get("metrics", {}).get("train", []), label=f"Train {k.split('_')[-1]}")
        plt.plot(epochs, hist.get("metrics", {}).get("val", []), "--", label=f"Val {k.split('_')[-1]}")
    plt.xlabel("Epoch")
    plt.ylabel("Complexity-Weighted Accuracy")
    plt.title("Baseline: Training vs Validation Accuracy (CpxWA)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Baseline_Accuracy_Curves.png"), dpi=300)
    plt.close()
except Exception as e:
    print("Error in Baseline Accuracy Curves:", e)
    plt.close()

# Plot 3: Baseline Test Performance Bar Chart
try:
    budgets = sorted(runs.keys(), key=lambda k: int(k.split('_')[-1]))
    xs = [int(k.split('_')[-1]) for k in budgets]
    test_scores = [runs[k].get("test_CpxWA", 0) for k in budgets]
    plt.figure(figsize=(8,6))
    bars = plt.bar(xs, test_scores, color="skyblue")
    # Annotate with test loss and training time if available.
    for bar, k in zip(bars, budgets):
        tloss = runs[k].get("test_loss", 0)
        ttime = runs[k].get("train_time_s", 0)
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"loss={tloss:.2f}\ntime={ttime:.0f}s",
                 ha="center", va="bottom", fontsize=10)
    plt.xlabel("Epoch Budget")
    plt.ylabel("Test Complexity-Weighted Accuracy")
    plt.title("Baseline: Test Performance vs Epoch Budget")
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Baseline_Test_Performance.png"), dpi=300)
    plt.close()
except Exception as e:
    print("Error in Baseline Test Performance Plot:", e)
    plt.close()

#################################
#  Research Experiments (3 figs)  #
#################################

# Research file path (exact as in summary)
RESEARCH_FILE = "experiment_results/experiment_b17edab791f14c288925c03b997cd4d7_proc_1494370/experiment_data.npy"

# Plot 4: Research SPR_RGCN Loss Curve
try:
    data = np.load(RESEARCH_FILE, allow_pickle=True).item()
    run = data.get("SPR_RGCN", {})
    plt.figure(figsize=(8,6))
    epochs = run.get("epochs", [])
    plt.plot(epochs, run.get("losses", {}).get("train", []), label="Train")
    plt.plot(epochs, run.get("losses", {}).get("val", []), "--", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_RGCN: Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "SPR_RGCN_Loss_Curve.png"), dpi=300)
    plt.close()
except Exception as e:
    print("Error in SPR_RGCN Loss Curve:", e)
    plt.close()

# Plot 5: Research SPR_RGCN Multi-Metric Curves (CWA, SWA, CmpWA) in subplots
try:
    metrics = ["CWA", "SWA", "CmpWA"]
    epochs = run.get("epochs", [])
    fig, axes = plt.subplots(1, 3, figsize=(18,5))
    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.plot(epochs, run.get("metrics", {}).get("train", {}).get(metric, []), label="Train")
        ax.plot(epochs, run.get("metrics", {}).get("val", {}).get(metric, []), "--", label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_title(f"SPR_RGCN: {metric} Curve")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "SPR_RGCN_MultiMetric_Curves.png"), dpi=300)
    plt.close()
except Exception as e:
    print("Error in SPR_RGCN Multi-Metric Curves:", e)
    plt.close()

# Plot 6: Research SPR_RGCN Test Metrics Summary Bar Chart
try:
    test_metrics = run.get("test_metrics", {})
    names = ["Loss", "CWA", "SWA", "CmpWA"]
    values = [test_metrics.get(k.lower(), 0) if k.lower() in test_metrics else test_metrics.get(k, 0) for k in names]
    plt.figure(figsize=(8,6))
    bars = plt.bar(names, values, color="skyblue")
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width()/2, b.get_height(), f"{v:.3f}",
                 ha="center", va="bottom")
    plt.title("SPR_RGCN: Test Set Performance")
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "SPR_RGCN_Test_Summary.png"), dpi=300)
    plt.close()
except Exception as e:
    print("Error in SPR_RGCN Test Summary Plot:", e)
    plt.close()

####################################
#  Ablation Experiments (6 figs)   #
####################################

# For ablation experiments, we assume each experiment_data file has a hierarchical dict:
# { model_key: { dataset_key: data_dict } }
# We will take the first (or only) key in each file for plotting.

# (7) Single-Relation Graph (No Relation Types)
SINGLE_RELATION_FILE = "experiment_results/experiment_3e6d9b87af2e4c8db558ab9323163637_proc_1497854/experiment_data.npy"
try:
    data = np.load(SINGLE_RELATION_FILE, allow_pickle=True).item()
    # take first model and dataset key
    model_key = next(iter(data))
    dataset_key = next(iter(data[model_key]))
    d = data[model_key][dataset_key]
    epochs = range(1, len(d.get("losses", {}).get("train", [])) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18,5))
    # Subplot A: Loss Curve
    axes[0].plot(epochs, d.get("losses", {}).get("train", []), label="Train")
    axes[0].plot(epochs, d.get("losses", {}).get("val", []), label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{dataset_key} | Single-Relation: Loss")
    axes[0].legend()
    # Subplot B: CWA Curve
    if "metrics" in d and "train" in d["metrics"] and "CWA" in d["metrics"]["train"]:
        axes[1].plot(epochs, d["metrics"]["train"]["CWA"], label="Train")
        axes[1].plot(epochs, d["metrics"]["val"]["CWA"], label="Validation")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Color-Weighted Accuracy")
        axes[1].set_title(f"{dataset_key} | Single-Relation: CWA")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "CWA data not found", ha="center")
    # Subplot C: Test Metrics Bar Chart
    test_met = d.get("test_metrics", {})
    names = ["CWA", "SWA", "CmpWA"]
    values = [test_met.get(k, 0) for k in names]
    axes[2].bar(names, values, color=["tab:blue", "tab:orange", "tab:green"])
    axes[2].set_ylim(0, 1)
    axes[2].set_title(f"{dataset_key} | Single-Relation: Test Metrics")
    for i, v in enumerate(values):
        axes[2].text(i, v+0.02, f"{v:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "SingleRelation_Graph_Final.png"), dpi=300)
    plt.close()
except Exception as e:
    print("Error in Single-Relation Graph Plot:", e)
    plt.close()

# (8) No-Sequential-Edge Graphs
NO_SEQ_FILE = "experiment_results/experiment_874f6ec47c324ba2b684b2ccf2de3978_proc_1497855/experiment_data.npy"
try:
    data = np.load(NO_SEQ_FILE, allow_pickle=True).item()
    model_key = next(iter(data))
    dataset_key = next(iter(data[model_key]))
    d = data[model_key][dataset_key]
    epochs = range(1, len(d.get("losses", {}).get("train", [])) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18,5))
    # Left: Loss curves
    axes[0].plot(epochs, d.get("losses", {}).get("train", []), label="Train")
    axes[0].plot(epochs, d.get("losses", {}).get("val", []), label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{dataset_key} | No-Sequential-Edge: Loss")
    axes[0].legend()
    # Middle: CWA curve
    if "metrics" in d and "train" in d["metrics"] and "CWA" in d["metrics"]["train"]:
        axes[1].plot(epochs, d["metrics"]["train"]["CWA"], label="Train")
        axes[1].plot(epochs, d["metrics"]["val"]["CWA"], label="Validation")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Color-Weighted Accuracy")
        axes[1].set_title(f"{dataset_key} | No-Sequential-Edge: CWA")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "CWA not found", ha="center")
    # Right: SWA curve
    if "metrics" in d and "train" in d["metrics"] and "SWA" in d["metrics"]["train"]:
        axes[2].plot(epochs, d["metrics"]["train"]["SWA"], label="Train")
        axes[2].plot(epochs, d["metrics"]["val"]["SWA"], label="Validation")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Shape-Weighted Accuracy")
        axes[2].set_title(f"{dataset_key} | No-Sequential-Edge: SWA")
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, "SWA data not found", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "NoSequentialEdge_Graph_Final.png"), dpi=300)
    plt.close()
except Exception as e:
    print("Error in No-Sequential-Edge Graph Plot:", e)
    plt.close()

# (9) No-Color-Edge Graphs: Confusion Matrix Plot
NO_COLOR_FILE = "experiment_results/experiment_ea82a1e8919b4a78945a6d4121683b36_proc_1497856/experiment_data.npy"
try:
    data = np.load(NO_COLOR_FILE, allow_pickle=True).item()
    # Use helper function from provided plot code: get nested dict safely.
    def get_nested(d, *keys):
        for k in keys:
            d = d.get(k, {})
        return d
    ds_name = "no_color_edge"
    model_name = "SPR_RGCN"
    exp_data = get_nested(data, ds_name, model_name, default={})
    if exp_data:
        epochs = exp_data.get("epochs", [])
        # Use predictions and ground truth for confusion matrix
        y_true = np.array(exp_data.get("ground_truth", []))
        y_pred = np.array(exp_data.get("predictions", []))
        if y_true.size and y_pred.size:
            num_cls = int(max(y_true.max(), y_pred.max())) + 1
            cm = np.zeros((num_cls, num_cls), dtype=int)
            for t, p in zip(y_true, y_pred):
                cm[t, p] += 1
            plt.figure(figsize=(6,5))
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{ds_name}: {model_name} Confusion Matrix")
            for i in range(num_cls):
                for j in range(num_cls):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i,j]>cm.max()/2 else "black", fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join("figures", "NoColorEdge_Confusion_Matrix.png"), dpi=300)
            plt.close()
        else:
            print("No prediction/ground truth data in No-Color-Edge experiment.")
    else:
        print("No data found for No-Color-Edge experiment.")
except Exception as e:
    print("Error in No-Color-Edge Graph Confusion Matrix Plot:", e)
    plt.close()

# (10) Shallow-GNN (1-hop) Ablation: 4 subplots aggregated
SHALLOW_FILE = "experiment_results/experiment_6c25bd281098419b922c16cf95221e6e_proc_1497855/experiment_data.npy"
try:
    data = np.load(SHALLOW_FILE, allow_pickle=True).item()
    model_key = next(iter(data))
    dataset_key = next(iter(data[model_key]))
    d = data[model_key][dataset_key]
    epochs = range(1, len(d.get("losses", {}).get("train", [])) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14,10))
    # Loss curve
    axes[0,0].plot(epochs, d.get("losses", {}).get("train", []), label="Train")
    axes[0,0].plot(epochs, d.get("losses", {}).get("val", []), label="Validation")
    axes[0,0].set_xlabel("Epoch")
    axes[0,0].set_ylabel("Loss")
    axes[0,0].set_title(f"{dataset_key}: Shallow-GNN Loss")
    axes[0,0].legend()
    # CmpWA curve
    if "metrics" in d and "train" in d["metrics"] and "CmpWA" in d["metrics"]["train"]:
        axes[0,1].plot(epochs, d["metrics"]["train"]["CmpWA"], label="Train")
        axes[0,1].plot(epochs, d["metrics"]["val"]["CmpWA"], label="Validation")
        axes[0,1].set_xlabel("Epoch")
        axes[0,1].set_ylabel("Cmp-Weighted Accuracy")
        axes[0,1].set_title(f"{dataset_key}: Shallow-GNN CmpWA")
        axes[0,1].legend()
    else:
        axes[0,1].text(0.5,0.5,"CmpWA data missing", ha="center")
    # CWA curve
    if "metrics" in d and "train" in d["metrics"] and "CWA" in d["metrics"]["train"]:
        axes[1,0].plot(epochs, d["metrics"]["train"]["CWA"], label="Train")
        axes[1,0].plot(epochs, d["metrics"]["val"]["CWA"], label="Validation")
        axes[1,0].set_xlabel("Epoch")
        axes[1,0].set_ylabel("Color-Weighted Accuracy")
        axes[1,0].set_title(f"{dataset_key}: Shallow-GNN CWA")
        axes[1,0].legend()
    else:
        axes[1,0].text(0.5,0.5,"CWA data missing", ha="center")
    # SWA curve
    if "metrics" in d and "train" in d["metrics"] and "SWA" in d["metrics"]["train"]:
        axes[1,1].plot(epochs, d["metrics"]["train"]["SWA"], label="Train")
        axes[1,1].plot(epochs, d["metrics"]["val"]["SWA"], label="Validation")
        axes[1,1].set_xlabel("Epoch")
        axes[1,1].set_ylabel("Shape-Weighted Accuracy")
        axes[1,1].set_title(f"{dataset_key}: Shallow-GNN SWA")
        axes[1,1].legend()
    else:
        axes[1,1].text(0.5,0.5,"SWA data missing", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "ShallowGNN_Ablation.png"), dpi=300)
    plt.close()
except Exception as e:
    print("Error in Shallow-GNN Ablation Plot:", e)
    plt.close()

# (11) One-Hot Node Features (Frozen Vocabulary Encoding): 4 subplots aggregated
ONEHOT_FILE = "experiment_results/experiment_7fdd5ed9a76248eb935ae841065e0340_proc_1497856/experiment_data.npy"
try:
    data = np.load(ONEHOT_FILE, allow_pickle=True).item()
    model_key = next(iter(data))
    dataset_key = next(iter(data[model_key]))
    d = data[model_key][dataset_key]
    epochs = range(1, len(d.get("losses", {}).get("train", [])) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14,10))
    # Loss curve
    axes[0,0].plot(epochs, d.get("losses", {}).get("train", []), label="Train")
    axes[0,0].plot(epochs, d.get("losses", {}).get("val", []), label="Validation")
    axes[0,0].set_xlabel("Epoch")
    axes[0,0].set_ylabel("Loss")
    axes[0,0].set_title(f"{dataset_key}: One-Hot Loss")
    axes[0,0].legend()
    # CmpWA curve
    if "metrics" in d and "train" in d["metrics"] and "CmpWA" in d["metrics"]["train"]:
        axes[0,1].plot(epochs, d["metrics"]["train"]["CmpWA"], label="Train")
        axes[0,1].plot(epochs, d["metrics"]["val"]["CmpWA"], label="Validation")
        axes[0,1].set_xlabel("Epoch")
        axes[0,1].set_ylabel("Cmp-Weighted Accuracy")
        axes[0,1].set_title(f"{dataset_key}: One-Hot CmpWA")
        axes[0,1].legend()
    else:
        axes[0,1].text(0.5,0.5,"CmpWA missing", ha="center")
    # CWA curve
    if "metrics" in d and "train" in d["metrics"] and "CWA" in d["metrics"]["train"]:
        axes[1,0].plot(epochs, d["metrics"]["train"]["CWA"], label="Train")
        axes[1,0].plot(epochs, d["metrics"]["val"]["CWA"], label="Validation")
        axes[1,0].set_xlabel("Epoch")
        axes[1,0].set_ylabel("Color-Weighted Accuracy")
        axes[1,0].set_title(f"{dataset_key}: One-Hot CWA")
        axes[1,0].legend()
    else:
        axes[1,0].text(0.5,0.5,"CWA missing", ha="center")
    # SWA curve
    if "metrics" in d and "train" in d["metrics"] and "SWA" in d["metrics"]["train"]:
        axes[1,1].plot(epochs, d["metrics"]["train"]["SWA"], label="Train")
        axes[1,1].plot(epochs, d["metrics"]["val"]["SWA"], label="Validation")
        axes[1,1].set_xlabel("Epoch")
        axes[1,1].set_ylabel("Shape-Weighted Accuracy")
        axes[1,1].set_title(f"{dataset_key}: One-Hot SWA")
        axes[1,1].legend()
    else:
        axes[1,1].text(0.5,0.5,"SWA missing", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "OneHot_NodeFeatures.png"), dpi=300)
    plt.close()
except Exception as e:
    print("Error in One-Hot Node Features Plot:", e)
    plt.close()

# (12) Multi-Synthetic Dataset Generalization Ablation: aggregated 3 subplots
MULTISYN_FILE = "experiment_results/experiment_f77b4bc2f1fa400b933187f41812e33c_proc_1497857/experiment_data.npy"
try:
    data = np.load(MULTISYN_FILE, allow_pickle=True).item()
    # The key "MultiSyntheticGeneralization" holds experiments with multiple runs.
    exp_dict = data.get("MultiSyntheticGeneralization", {})
    exp_names = list(exp_dict.keys())
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    # Subplot 1: Loss Curves from each experiment
    fig, axes = plt.subplots(1, 3, figsize=(18,5))
    ax0 = axes[0]
    for idx, name in enumerate(exp_names):
        run_data = exp_dict[name]
        train_loss = run_data.get("losses", {}).get("train", [])
        val_loss = run_data.get("losses", {}).get("val", [])
        epochs_arr = np.arange(1, len(train_loss)+1)
        ax0.plot(epochs_arr, train_loss, color=colors[idx % len(colors)], 
                 label=f"{name}-train", linewidth=1.5)
        ax0.plot(epochs_arr, val_loss, color=colors[idx % len(colors)], 
                 label=f"{name}-val", linestyle="--", linewidth=1.5)
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Loss")
    ax0.set_title("MultiSyntheticGeneralization: Loss Curves")
    ax0.legend(fontsize="small")
    # Subplot 2: CmpWA Curves from each experiment
    ax1 = axes[1]
    for idx, name in enumerate(exp_names):
        run_data = exp_dict[name]
        train_cmp = run_data.get("CmpWA_train", [])
        val_cmp = run_data.get("CmpWA_val", [])
        epochs_arr = np.arange(1, len(train_cmp)+1)
        ax1.plot(epochs_arr, train_cmp, color=colors[idx % len(colors)], 
                 label=f"{name}-train", linewidth=1.5)
        ax1.plot(epochs_arr, val_cmp, color=colors[idx % len(colors)], 
                 label=f"{name}-val", linestyle="--", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Composite Weighted Accuracy")
    ax1.set_title("MultiSyntheticGeneralization: CmpWA Curves")
    ax1.legend(fontsize="small")
    # Subplot 3: Test Metrics Bar Chart (for metrics CWA, SWA, CmpWA)
    ax2 = axes[2]
    metrics_list = ["CWA", "SWA", "CmpWA"]
    # For each run, take its test_metrics and plot grouped bars.
    # We'll assume all experiments have a "test_metrics" key.
    n_exps = len(exp_names)
    x = np.arange(len(metrics_list))
    width = 0.8 / n_exps
    for idx, name in enumerate(exp_names):
        run_data = exp_dict[name]
        test_met = run_data.get("test_metrics", {})
        vals = [test_met.get(m, 0) for m in metrics_list]
        ax2.bar(x + idx * width, vals, width=width, color=colors[idx % len(colors)], label=name)
    ax2.set_xticks(x + width*(n_exps-1)/2)
    ax2.set_xticklabels(metrics_list)
    ax2.set_ylabel("Score")
    ax2.set_ylim(0,1.05)
    ax2.set_title("MultiSyntheticGeneralization: Test Metrics")
    ax2.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "MultiSyntheticGeneralization.png"), dpi=300)
    plt.close()
except Exception as e:
    print("Error in Multi-Synthetic Dataset Generalization Plot:", e)
    plt.close()

print("All plots generated and saved in the 'figures/' directory.")
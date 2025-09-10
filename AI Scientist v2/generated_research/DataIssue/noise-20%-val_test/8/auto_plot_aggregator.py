#!/usr/bin/env python3
"""
Final Aggregated Plotter for Interpretable Neural Rule Learning
This script loads experiment result data (.npy files) from several experiments 
(baseline, multi‐synthetic generalization, frequency vs presence ablation, 
length‐normalized ablation, tree depth sensitivity, positional-information ablation, 
character‐vocabulary reduction, training-data size ablation)
and produces a series of final, publication–quality figures in the "figures" directory.
Each figure is wrapped in a try–except block such that a failure for one plot does not
affect others.
    
Author: AI Researcher
Date: October 2023
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Increase font size for publication quality
plt.rcParams.update({'font.size': 14})

# Create output figures directory
os.makedirs("figures", exist_ok=True)

def style_axis(ax):
    """Remove top and right spines for a publication–quality look."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def load_experiment_data(filepath):
    """Load a .npy file given an exact filepath."""
    try:
        data = np.load(filepath, allow_pickle=True).item()
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

# Load each experiment's data using the exact file paths provided.
exp_files = {
    "baseline": "experiment_results/experiment_5cb774c39d44465284cc15e85dd077c6_proc_3198565/experiment_data.npy",
    "multi_synth": "experiment_results/experiment_033df0ac42ca4a99b65cb4cbb4d047ba_proc_3214201/experiment_data.npy",
    "frequency_presence": "experiment_results/experiment_34759030404041638fdf7f68b91c38df_proc_3214202/experiment_data.npy",
    "length_normalized": "experiment_results/experiment_7f5ef05924d149cd84df086d6859f442_proc_3214203/experiment_data.npy",
    "tree_depth": "experiment_results/experiment_748aa1fd8c5b480cb6d5a95b76b86877_proc_3214204/experiment_data.npy",
    "positional": "experiment_results/experiment_39b46123b0ec43269646b7ad882bf2c3_proc_3214201/experiment_data.npy",
    "vocab_reduction": "experiment_results/experiment_2d0c21a5126b441e947603a3b2ab130e_proc_3214202/experiment_data.npy",
    "training_size": "experiment_results/experiment_009e70fb8b844f24ad34f165e13da248_proc_3214203/experiment_data.npy",
}

experiments = {}
for key, fpath in exp_files.items():
    experiments[key] = load_experiment_data(fpath)

#######################################################################
# Figure 1: Baseline Experiment - Validation Loss & Accuracy Curves (Aggregated)
#######################################################################
try:
    # Baseline experiment_data is a dict with keys corresponding to datasets.
    baseline_data = experiments["baseline"]
    # For each dataset in baseline_data, if available, plot val loss and accuracy curves.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for dname, ddict in baseline_data.items():
        losses = ddict.get("losses", {})
        metrics = ddict.get("metrics", {})
        val_loss = losses.get("val", [])
        val_acc = metrics.get("val", [])
        epochs = np.arange(1, len(val_loss) + 1) if val_loss else []
        if epochs.size > 0:
            axes[0].plot(epochs, val_loss, marker="o", label=f"{dname}")
        if val_acc:
            axes[1].plot(np.arange(1, len(val_acc) + 1), val_acc, marker="s", label=f"{dname}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation Loss")
    axes[0].set_title("Baseline Validation Loss Curve")
    style_axis(axes[0])
    axes[0].legend(frameon=False)
    
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Accuracy")
    axes[1].set_title("Baseline Validation Accuracy Curve")
    style_axis(axes[1])
    axes[1].legend(frameon=False)
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "baseline_val_loss_accuracy.png"), dpi=300)
    plt.close()
except Exception as e:
    print(f"Error in Figure 1 (Baseline curves): {e}")
    plt.close()

#######################################################################
# Figure 2: Baseline Experiment - Confusion Matrix
#######################################################################
try:
    # For baseline, pick a dataset (first key) and get predicted and ground-truth.
    for dname, ddict in baseline_data.items():
        y_pred = np.array(ddict.get("predictions", []))
        y_true = np.array(ddict.get("ground_truth", []))
        if y_true.size and y_pred.size:
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(5, 5))
            im = ax.imshow(cm, cmap="Blues")
            style_axis(ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"Baseline Confusion Matrix: {dname}")
            # Annotate cells
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, str(v), ha="center", va="center", color="black")
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(os.path.join("figures", f"baseline_confusion_matrix_{dname}.png"), dpi=300)
            plt.close()
        break  # only plot one dataset's confusion matrix
except Exception as e:
    print(f"Error in Figure 2 (Baseline Confusion Matrix): {e}")
    plt.close()

#######################################################################
# Figure 3: Multi-Synthetic Dataset Generalization (Ablation) - Combined 2x2 Subplots
#######################################################################
try:
    # Load multi-synthetic data. Key in the npy file should be "multi_synth_generalization"
    multi_data = experiments["multi_synth"].get("Multi-Synthetic Dataset Generalization Test", {})
    # In the provided code, after loading: 
    # data_dict = experiment_data[exp_key]
    data_dict = multi_data  # assuming multi_data is already the inner dictionary.
    dnames = list(data_dict.keys())
    train_acc = [data_dict[d]["metrics"]["train"][0] for d in dnames]
    val_acc = [data_dict[d]["metrics"]["val"][0] for d in dnames]
    test_acc = [data_dict[d]["metrics"]["test"][0] for d in dnames]
    val_loss = [data_dict[d]["losses"]["val"][0] for d in dnames]
    complexity = [data_dict[d]["rule_complexity"] for d in dnames]
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    x = np.arange(len(dnames))
    width = 0.25
    # Subplot (0,0): Grouped Bar Chart: Accuracy per Split
    axs[0,0].bar(x - width, train_acc, width, label="Train")
    axs[0,0].bar(x, val_acc, width, label="Validation")
    axs[0,0].bar(x + width, test_acc, width, label="Test")
    axs[0,0].set_xticks(x)
    axs[0,0].set_xticklabels(dnames, rotation=45, ha="right")
    axs[0,0].set_ylabel("Accuracy")
    axs[0,0].set_title("Accuracy by Split per Dataset")
    style_axis(axs[0,0])
    axs[0,0].legend(frameon=False)
    
    # Subplot (0,1): Validation Loss Bar Chart
    axs[0,1].bar(dnames, val_loss, color="orange")
    axs[0,1].set_ylabel("Log Loss")
    axs[0,1].set_title("Validation Loss per Dataset")
    style_axis(axs[0,1])
    
    # Subplot (1,0): Rule Complexity Bar Chart
    axs[1,0].bar(dnames, complexity, color="green")
    axs[1,0].set_ylabel("Number of Tree Nodes")
    axs[1,0].set_title("Extracted Rule Complexity")
    style_axis(axs[1,0])
    
    # Subplot (1,1): Scatter Plot: Complexity vs Test Accuracy
    for d, c, a in zip(dnames, complexity, test_acc):
        axs[1,1].scatter(c, a, label=d)
        axs[1,1].text(c + 0.5, a, d)
    axs[1,1].set_xlabel("Rule Complexity (nodes)")
    axs[1,1].set_ylabel("Test Accuracy")
    axs[1,1].set_title("Complexity vs Test Accuracy")
    axs[1,1].set_ylim(0, 1.05)
    style_axis(axs[1,1])
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "multi_synth_generalization.png"), dpi=300)
    plt.close()
except Exception as e:
    print(f"Error in Figure 3 (Multi-Synthetic Generalization): {e}")
    plt.close()

#######################################################################
# Figure 4: Frequency vs Presence Feature Ablation - Combined 1x3 Subplots
#######################################################################
try:
    fp_data = experiments["frequency_presence"]
    # List all experimental variants from frequency_presence npy file.
    variants = list(fp_data.keys())
    dataset = "SPR_BENCH"  # as per summary
    val_acc = []
    test_acc = []
    val_loss = []
    depths = []
    rule_lens = []
    for v in variants:
        try:
            info = fp_data[v][dataset]
            m = info.get("metrics", {})
            l = info.get("losses", {})
            val_acc.append(m.get("val", [np.nan])[0])
            test_acc.append(m.get("test", [np.nan])[0])
            val_loss.append(l.get("val", [np.nan])[0])
            depths.append(m.get("rule_depth", [np.nan])[0])
            rule_lens.append(m.get("avg_rule_len", [np.nan])[0])
        except Exception:
            val_acc.append(np.nan)
            test_acc.append(np.nan)
            val_loss.append(np.nan)
            depths.append(np.nan)
            rule_lens.append(np.nan)
    x = np.arange(len(variants))
    width = 0.35
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    # Accuracy comparison
    axs[0].bar(x - width/2, val_acc, width, label="Validation")
    axs[0].bar(x + width/2, test_acc, width, label="Test")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(variants, rotation=45, ha="right")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_title(f"{dataset}: Validation vs Test Accuracy")
    style_axis(axs[0])
    axs[0].legend(frameon=False)
    
    # Validation Loss Comparison
    axs[1].bar(variants, val_loss, color="orange")
    axs[1].set_ylabel("Log Loss")
    axs[1].set_title(f"{dataset}: Validation Loss")
    style_axis(axs[1])
    
    # Complexity Metrics: dual axis plot
    ax1 = axs[2]
    ax2 = ax1.twinx()
    ax1.bar(x - width/2, depths, width, label="Tree Depth", color="green")
    ax2.bar(x + width/2, rule_lens, width, label="Avg Rule Length", color="purple")
    ax1.set_xticks(x)
    ax1.set_xticklabels(variants, rotation=45, ha="right")
    ax1.set_ylabel("Tree Depth")
    ax2.set_ylabel("Avg Rule Length")
    axs[2].set_title(f"{dataset}: Model Complexity Metrics")
    style_axis(ax1)
    style_axis(ax2)
    fig.legend(loc="upper right", bbox_to_anchor=(0.95, 0.95))
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "frequency_vs_presence_ablation.png"), dpi=300)
    plt.close()
except Exception as e:
    print(f"Error in Figure 4 (Frequency vs Presence Ablation): {e}")
    plt.close()

#######################################################################
# Figure 5: Length-Normalized Feature Ablation - Combined 2x2 Subplots
#######################################################################
try:
    ln_data = experiments["length_normalized"]
    # Expected structure: ln_data has keys "baseline" and "length_normalized" under dataset "SPR_BENCH"
    dataset = "SPR_BENCH"
    runs = ["baseline", "length_normalized"]
    val_losses = []
    test_accs = []
    y_preds = {}
    y_trues = {}
    for r in runs:
        run_data = ln_data[r][dataset]
        val_losses.append(run_data["losses"]["val"][0])
        test_accs.append(run_data["metrics"]["test"][0])
        y_preds[r] = np.array(run_data.get("predictions", []))
        y_trues[r] = np.array(run_data.get("ground_truth", []))
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    # Top-left: Bar chart of Validation Loss
    axs[0,0].bar(runs, val_losses, color=["steelblue", "orange"])
    axs[0,0].set_ylabel("Validation Loss")
    axs[0,0].set_title("Validation Loss Comparison")
    style_axis(axs[0,0])
    
    # Top-right: Bar chart of Test Accuracy
    axs[0,1].bar(runs, test_accs, color=["seagreen", "salmon"])
    axs[0,1].set_ylabel("Test Accuracy")
    axs[0,1].set_ylim(0,1)
    axs[0,1].set_title("Test Accuracy Comparison")
    style_axis(axs[0,1])
    
    # Bottom-left: Confusion Matrix for Baseline run
    cm_baseline = confusion_matrix(y_trues["baseline"], y_preds["baseline"])
    im0 = axs[1,0].imshow(cm_baseline, cmap="Blues")
    axs[1,0].set_title("Confusion Matrix - Baseline")
    axs[1,0].set_xlabel("Predicted")
    axs[1,0].set_ylabel("True")
    for (i, j), v in np.ndenumerate(cm_baseline):
        axs[1,0].text(j, i, str(v), ha="center", va="center", color="black")
    style_axis(axs[1,0])
    plt.colorbar(im0, ax=axs[1,0])
    
    # Bottom-right: Confusion Matrix for Length Normalized run
    cm_ln = confusion_matrix(y_trues["length_normalized"], y_preds["length_normalized"])
    im1 = axs[1,1].imshow(cm_ln, cmap="Blues")
    axs[1,1].set_title("Confusion Matrix - Length Normalized")
    axs[1,1].set_xlabel("Predicted")
    axs[1,1].set_ylabel("True")
    for (i, j), v in np.ndenumerate(cm_ln):
        axs[1,1].text(j, i, str(v), ha="center", va="center", color="black")
    style_axis(axs[1,1])
    plt.colorbar(im1, ax=axs[1,1])
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "length_normalized_ablation.png"), dpi=300)
    plt.close()
except Exception as e:
    print(f"Error in Figure 5 (Length-Normalized Ablation): {e}")
    plt.close()

#######################################################################
# Figure 6: Tree Depth Sensitivity Ablation - Combined 1x3 Subplots
#######################################################################
try:
    td_data_full = experiments["tree_depth"]
    # In this experiment, data is stored under key "tree_depth_sensitivity"
    td_root = td_data_full.get("tree_depth_sensitivity", {})
    if td_root:
        # Use the first (and only) dataset key
        dataset_name = list(td_root.keys())[0]
        td_data = td_root[dataset_name]
        depth_labels = [str(d) for d in td_data["depths"]]
        depth_xticks = [("∞" if d=="None" else d) for d in depth_labels]
        
        fig, axs = plt.subplots(1, 3, figsize=(18,5))
        x = np.arange(len(depth_labels))
        # Accuracy vs Depth (train/val/test)
        axs[0].plot(x, td_data["metrics"]["train"], "o-", label="Train")
        axs[0].plot(x, td_data["metrics"]["val"], "s-", label="Validation")
        axs[0].plot(x, td_data["metrics"]["test"], "^-", label="Test")
        axs[0].set_xticks(x)
        axs[0].set_xticklabels(depth_xticks)
        axs[0].set_xlabel("Tree Depth")
        axs[0].set_ylabel("Accuracy")
        axs[0].set_title(f"Accuracy vs Depth ({dataset_name})")
        style_axis(axs[0])
        axs[0].legend(frameon=False)
        
        # Loss vs Depth (train and validation)
        axs[1].plot(x, td_data["losses"]["train"], "o-", label="Train")
        axs[1].plot(x, td_data["losses"]["val"], "s-", label="Validation")
        axs[1].set_xticks(x)
        axs[1].set_xticklabels(depth_xticks)
        axs[1].set_xlabel("Tree Depth")
        axs[1].set_ylabel("Log Loss")
        axs[1].set_title(f"Log Loss vs Depth ({dataset_name})")
        style_axis(axs[1])
        axs[1].legend(frameon=False)
        
        # Rule Count vs Depth
        axs[2].bar(x, td_data["rule_counts"])
        axs[2].set_xticks(x)
        axs[2].set_xticklabels(depth_xticks)
        axs[2].set_xlabel("Tree Depth")
        axs[2].set_ylabel("Number of Extracted Rules")
        axs[2].set_title(f"Rule Count vs Depth ({dataset_name})")
        style_axis(axs[2])
        
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "tree_depth_sensitivity.png"), dpi=300)
        plt.close()
except Exception as e:
    print(f"Error in Figure 6 (Tree Depth Sensitivity): {e}")
    plt.close()

#######################################################################
# Figure 7: Positional-Information Feature Ablation - Combined 1x3 Subplots
#######################################################################
try:
    pos_data = experiments["positional"]
    dataset = "SPR_BENCH"
    models = list(pos_data.keys())
    val_accs = []
    val_losses = []
    # For label distribution, take predictions from the last model.
    for m in models:
        try:
            val_accs.append(pos_data[m][dataset]["metrics"]["val"][0])
            val_losses.append(pos_data[m][dataset]["losses"]["val"][0])
        except Exception:
            val_accs.append(np.nan)
            val_losses.append(np.nan)
    # Assume last model for distribution.
    last_model = models[-1]
    gt = np.array(pos_data[last_model][dataset]["ground_truth"])
    pred = np.array(pos_data[last_model][dataset]["predictions"])
    gt_cnt = np.bincount(gt, minlength=2)
    pr_cnt = np.bincount(pred, minlength=2)
    
    fig, axs = plt.subplots(1, 3, figsize=(18,5))
    axs[0].bar(models, val_accs, color="skyblue")
    axs[0].set_ylabel("Validation Accuracy")
    axs[0].set_title(f"{dataset} Validation Accuracy per Model")
    style_axis(axs[0])
    
    axs[1].bar(models, val_losses, color="salmon")
    axs[1].set_ylabel("Validation Log Loss")
    axs[1].set_title(f"{dataset} Validation Loss per Model")
    style_axis(axs[1])
    
    width = 0.35
    x_labels = np.arange(2)
    axs[2].bar(x_labels - width/2, gt_cnt, width, label="Ground Truth")
    axs[2].bar(x_labels + width/2, pr_cnt, width, label="Predictions")
    axs[2].set_xticks(x_labels)
    axs[2].set_xticklabels(["Label 0", "Label 1"])
    axs[2].set_ylabel("Count")
    axs[2].set_title(f"Label Distribution ({dataset})")
    style_axis(axs[2])
    axs[2].legend(frameon=False)
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "positional_information_ablation.png"), dpi=300)
    plt.close()
except Exception as e:
    print(f"Error in Figure 7 (Positional-Information Ablation): {e}")
    plt.close()

#######################################################################
# Figure 8: Character-Vocabulary Reduction Ablation
# (A) Grouped Bar Accuracy Comparison and (B) Confusion Matrices for up to 3 experiments
#######################################################################
try:
    vocab_data = experiments["vocab_reduction"]
    exp_names = list(vocab_data.keys())
    train_accs = []
    val_accs = []
    test_accs = []
    # Helper function to fetch first dataset key for each experiment entry
    def first_ds_key(exp_dict):
        return next(iter(exp_dict.keys()))
    for exp in exp_names:
        ds_key = first_ds_key(vocab_data[exp])
        res = vocab_data[exp][ds_key]
        train_accs.append(res["metrics"]["train"][0])
        val_accs.append(res["metrics"]["val"][0])
        test_accs.append(res["metrics"]["test"][0])
    x = np.arange(len(exp_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(x - width, train_accs, width, label="Train")
    ax.bar(x, val_accs, width, label="Validation")
    ax.bar(x + width, test_accs, width, label="Test")
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{first_ds_key(vocab_data[exp_names[0]])}: Accuracy Comparison")
    style_axis(ax)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "vocab_reduction_accuracy_comparison.png"), dpi=300)
    plt.close()
    
    # Confusion Matrices for up to 3 experiments
    max_conf = 3
    for idx, exp in enumerate(exp_names[:max_conf]):
        try:
            ds_key = first_ds_key(vocab_data[exp])
            res = vocab_data[exp][ds_key]
            y_true = np.array(res["ground_truth"])
            y_pred = np.array(res["predictions"])
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(5,5))
            im = ax.imshow(cm, cmap="Blues")
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, str(v), ha="center", va="center", color="black")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"{ds_key} Confusion Matrix\nExperiment: {exp}")
            style_axis(ax)
            plt.colorbar(im, ax=ax, fraction=0.046)
            plt.tight_layout()
            fname = os.path.join("figures", f"vocab_reduction_confusion_matrix_{exp}.png")
            plt.savefig(fname, dpi=300)
            plt.close()
        except Exception as ce:
            print(f"Error creating confusion matrix for {exp}: {ce}")
            plt.close()
except Exception as e:
    print(f"Error in Figure 8 (Vocabulary Reduction Ablation): {e}")
    plt.close()

#######################################################################
# Figure 9: Training-Data Size Ablation - Combined 1x3 Subplots
#######################################################################
try:
    ts_data = experiments["training_size"]["training_data_size_ablation"]["SPR_BENCH"]
    fractions = np.array(ts_data["fractions"])
    val_acc = np.array(ts_data["metrics"]["val_accuracy"])
    test_acc = np.array(ts_data["metrics"]["test_accuracy"])
    val_loss = np.array(ts_data["losses"]["val_logloss"])
    y_true = np.array(ts_data["ground_truth"])
    # Predictions stored in a dict keyed by fraction (as string keys maybe)
    preds_dict = {float(k): np.array(v) for k, v in ts_data["predictions"].items()}
    
    fig, axs = plt.subplots(1, 3, figsize=(18,5))
    # Subplot 1: Accuracy curves vs Training Fraction
    axs[0].plot(fractions, val_acc, "o-", label="Validation Accuracy")
    axs[0].plot(fractions, test_acc, "s-", label="Test Accuracy")
    axs[0].set_xlabel("Training Fraction")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_title("Accuracy vs Training Data Size")
    style_axis(axs[0])
    axs[0].legend(frameon=False)
    
    # Subplot 2: Log-Loss curve vs Training Fraction
    axs[1].plot(fractions, val_loss, "d-", color="purple")
    axs[1].set_xlabel("Training Fraction")
    axs[1].set_ylabel("Validation Log-Loss")
    axs[1].set_title("Validation Log-Loss vs Training Data Size")
    style_axis(axs[1])
    
    # Subplot 3: Confusion Matrix for 100% Data (fraction == 1.0)
    if 1.0 in preds_dict:
        y_pred = preds_dict[1.0]
        labels = np.unique(np.concatenate([y_true, y_pred]))
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        im = axs[2].imshow(cm, cmap="Blues")
        axs[2].set_xlabel("Predicted")
        axs[2].set_ylabel("True")
        axs[2].set_title("Confusion Matrix (100% Training Data)")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axs[2].text(j, i, str(cm[i,j]), ha="center", va="center", color="black")
        plt.colorbar(im, ax=axs[2])
        style_axis(axs[2])
    else:
        axs[2].text(0.5, 0.5, "No predictions for fraction=1.0", ha="center", va="center")
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "training_data_size_ablation.png"), dpi=300)
    plt.close()
except Exception as e:
    print(f"Error in Figure 9 (Training-Data Size Ablation): {e}")
    plt.close()

print("All figures generated and saved in the 'figures' directory.")
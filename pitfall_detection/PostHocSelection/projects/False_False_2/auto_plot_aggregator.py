#!/usr/bin/env python3
"""
Aggregator Script for Final Figures
This script loads experimental .npy results from several experiments and produces aggregated,
publication‐quality plots in the "figures/" folder. Each plot is created inside a try‐except
block to ensure one failure does not halt the entire script.

Experiment files (use exact paths):
  1) Baseline/Research:
     "experiment_results/experiment_1f0e024b9fe4461aa7e60322cf845911_proc_329318/experiment_data.npy"
     
  2) Remove-Color-Features:
     "experiment_results/experiment_f5e91367cd0e45eb9eef43272bd6fe87_proc_335107/experiment_data.npy"
     
  3) No-Hidden-Layer (Linear-Only Model):
     "experiment_results/experiment_34ed4827a6c54b7087493bf28648acd0_proc_335108/experiment_data.npy"
     
  4) Binary-Feature Representation (Remove Token Counts):
     "experiment_results/experiment_d179aed5c71440009b37b91dc06d29c7_proc_335109/experiment_data.npy"
     
  5) Length-Invariant Feature Normalization:
     "experiment_results/experiment_b9e1a2d7170d4816a31871fbe071b6a0_proc_335107/experiment_data.npy"
     
  6) Joint-Token-Only Representation:
     "experiment_results/experiment_149823280dbb4b41a95b87fbf8c94302_proc_335109/experiment_data.npy"
     
  7) No-Early-Stopping (Fixed-Epoch Training):
     "experiment_results/experiment_a6c47dccb8884e6db85f19b594fb2d42_proc_335110/experiment_data.npy"

All final figures are saved to "figures/".
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Increase font sizes for publication quality
plt.rcParams.update({"font.size": 14, "figure.dpi": 300})
# Remove top and right spines for all plots (we use a helper below)
def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Ensure figures directory exists
os.makedirs("figures", exist_ok=True)

#############################
# 1) Baseline/Research Plot #
#############################
try:
    baseline_path = "experiment_results/experiment_1f0e024b9fe4461aa7e60322cf845911_proc_329318/experiment_data.npy"
    baseline_data = np.load(baseline_path, allow_pickle=True).item()
    # The baseline data is stored under "epochs_tuning". We assume one key, e.g. "spr_bench"
    bt_dict = baseline_data.get("epochs_tuning", {})
    if not bt_dict:
        raise ValueError("No 'epochs_tuning' key found.")
    # For simplicity, combine the plots for the first key (e.g. "spr_bench")
    key = list(bt_dict.keys())[0]
    log = bt_dict[key]
    epochs = log.get("epochs", [])
    train_loss = log.get("losses", {}).get("train", [])
    dev_loss   = log.get("losses", {}).get("dev", [])
    train_pha  = log.get("metrics", {}).get("train_PHA", [])
    dev_pha    = log.get("metrics", {}).get("dev_PHA", [])
    test_metrics = log.get("test_metrics", {})
    ground_truth = np.asarray(log.get("ground_truth", []))
    predictions  = np.asarray(log.get("predictions", []))

    # Figure: Loss and PHA curves side-by-side
    fig, axs = plt.subplots(1, 2, figsize=(12,5))
    axs[0].plot(epochs, train_loss, label="Train")
    axs[0].plot(epochs, dev_loss, label="Validation")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title(f"{key} Loss Curve")
    axs[0].legend()
    style_axes(axs[0])
    
    axs[1].plot(epochs, train_pha, label="Train PHA")
    axs[1].plot(epochs, dev_pha, label="Validation PHA")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("PHA")
    axs[1].set_title(f"{key} PHA Curve")
    axs[1].legend()
    style_axes(axs[1])
    
    plt.tight_layout()
    save_path = os.path.join("figures", "baseline_loss_and_PHA_curves.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Baseline Loss and PHA curves to {save_path}")
except Exception as e:
    print("Error in Baseline/Research plots:", str(e))
    plt.close()

# Figure: Test Metrics Bar Chart and Confusion Matrix (side-by-side)
try:
    fig, axs = plt.subplots(1, 2, figsize=(12,5))
    # Bar chart for test metrics (SWA, CWA, PHA)
    if test_metrics:
        metric_names = list(test_metrics.keys())
        metric_vals = [test_metrics[k] for k in metric_names]
        axs[0].bar(metric_names, metric_vals, color=["tab:blue", "tab:orange", "tab:green"])
        axs[0].set_ylim(0,1)
        axs[0].set_ylabel("Score")
        axs[0].set_title(f"{key} Test Metrics")
        for i, v in enumerate(metric_vals):
            axs[0].text(i, v+0.02, f"{v:.2f}", ha="center")
    else:
        axs[0].text(0.5, 0.5, "No test metrics", ha="center")

    style_axes(axs[0])
    # Confusion Matrix (if ground_truth and predictions available)
    if ground_truth.size and predictions.size:
        n_cls = int(max(ground_truth.max(), predictions.max()))+1
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(ground_truth, predictions):
            cm[t, p] += 1
        im = axs[1].imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
        axs[1].set_xlabel("Predicted")
        axs[1].set_ylabel("True")
        axs[1].set_title(f"{key} Confusion Matrix")
        # Annotate
        for i in range(n_cls):
            for j in range(n_cls):
                color = "white" if cm[i,j] > cm.max()/2 else "black"
                axs[1].text(j, i, str(cm[i,j]), ha="center", va="center", color=color)
    else:
        axs[1].text(0.5, 0.5, "No confusion matrix data", ha="center")
    style_axes(axs[1])
    
    plt.tight_layout()
    save_path = os.path.join("figures", "baseline_test_metrics_and_confusion.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Baseline test metric and confusion matrix to {save_path}")
except Exception as e:
    print("Error in Baseline test/CM plots:", str(e))
    plt.close()

##################################################
# 2) Remove-Color-Features (Ablation Study) Plot  #
##################################################
try:
    rcf_path = "experiment_results/experiment_f5e91367cd0e45eb9eef43272bd6fe87_proc_335107/experiment_data.npy"
    data_rcf = np.load(rcf_path, allow_pickle=True).item()
    # Data stored under key "remove_color_features" -> "spr_bench"
    log = data_rcf.get("remove_color_features", {}).get("spr_bench", {})
    epochs   = log.get("epochs", [])
    train_loss = log.get("losses", {}).get("train", [])
    dev_loss   = log.get("losses", {}).get("dev", [])
    train_pha  = log.get("metrics", {}).get("train_PHA", [])
    dev_pha    = log.get("metrics", {}).get("dev_PHA", [])
    test_metrics = log.get("test_metrics", {})
    
    fig, axs = plt.subplots(1, 3, figsize=(18,5))
    # Loss Curve
    axs[0].plot(epochs, train_loss, label="Train")
    axs[0].plot(epochs, dev_loss, label="Validation")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Remove-Color Features: Loss Curve")
    axs[0].legend()
    style_axes(axs[0])
    
    # PHA Curve
    axs[1].plot(epochs, train_pha, label="Train PHA")
    axs[1].plot(epochs, dev_pha, label="Validation PHA")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("PHA")
    axs[1].set_title("Remove-Color Features: PHA Curve")
    axs[1].legend()
    style_axes(axs[1])
    
    # Test Metrics Bar Chart
    if test_metrics:
        names = list(test_metrics.keys())
        vals  = [test_metrics[n] for n in names]
        axs[2].bar(names, vals, color=["steelblue", "orange", "green"])
        axs[2].set_ylim(0,1)
        axs[2].set_ylabel("Score")
        axs[2].set_title("Remove-Color Features: Test Metrics")
        for i, v in enumerate(vals):
            axs[2].text(i, v+0.02, f"{v:.2f}", ha="center")
    else:
        axs[2].text(0.5, 0.5, "No test metrics", ha="center")
    style_axes(axs[2])
    
    plt.tight_layout()
    save_path = os.path.join("figures", "remove_color_features_combined.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Remove-Color-Features plots to {save_path}")
except Exception as e:
    print("Error in Remove-Color-Features plots:", str(e))
    plt.close()

###########################################################
# 3) No-Hidden-Layer (Linear-Only Model) Ablation Plots   #
###########################################################
try:
    nhl_path = "experiment_results/experiment_34ed4827a6c54b7087493bf28648acd0_proc_335108/experiment_data.npy"
    data_nhl = np.load(nhl_path, allow_pickle=True).item()
    # Data stored under key "ablation_no_hidden_layer" -> "spr_bench"
    log = data_nhl.get("ablation_no_hidden_layer", {}).get("spr_bench", {})
    epochs = log.get("epochs", [])
    train_loss = log.get("losses", {}).get("train", [])
    dev_loss   = log.get("losses", {}).get("dev", [])
    train_pha  = log.get("metrics", {}).get("train_PHA", [])
    dev_pha    = log.get("metrics", {}).get("dev_PHA", [])
    test_metrics = log.get("test_metrics", {})
    ground_truth = np.asarray(log.get("ground_truth", []))
    predictions  = np.asarray(log.get("predictions", []))
    
    fig, axs = plt.subplots(2, 2, figsize=(14,10))
    # Loss Curve
    axs[0,0].plot(epochs, train_loss, label="Train")
    axs[0,0].plot(epochs, dev_loss, label="Validation")
    axs[0,0].set_xlabel("Epoch")
    axs[0,0].set_ylabel("Loss")
    axs[0,0].set_title("No-Hidden-Layer: Loss Curve")
    axs[0,0].legend()
    style_axes(axs[0,0])
    
    # PHA Curve
    axs[0,1].plot(epochs, train_pha, label="Train PHA")
    axs[0,1].plot(epochs, dev_pha, label="Validation PHA")
    axs[0,1].set_xlabel("Epoch")
    axs[0,1].set_ylabel("PHA")
    axs[0,1].set_title("No-Hidden-Layer: PHA Curve")
    axs[0,1].legend()
    style_axes(axs[0,1])
    
    # Test Metrics Bar Chart
    if test_metrics:
        names = list(test_metrics.keys())
        vals  = [test_metrics[n] for n in names]
        axs[1,0].bar(names, vals, color=["tab:blue", "tab:orange", "tab:green"])
        axs[1,0].set_ylim(0,1)
        axs[1,0].set_ylabel("Score")
        axs[1,0].set_title("No-Hidden-Layer: Test Metrics")
        for i, v in enumerate(vals):
            axs[1,0].text(i, v+0.02, f"{v:.2f}", ha="center")
    else:
        axs[1,0].text(0.5, 0.5, "No test metrics", ha="center")
    style_axes(axs[1,0])
    
    # Confusion Matrix
    if ground_truth.size and predictions.size:
        n_cls = int(max(ground_truth.max(), predictions.max())) + 1
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(ground_truth, predictions):
            cm[t, p] += 1
        im = axs[1,1].imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=axs[1,1], fraction=0.046, pad=0.04)
        axs[1,1].set_xlabel("Predicted")
        axs[1,1].set_ylabel("True")
        axs[1,1].set_title("No-Hidden-Layer: Confusion Matrix")
        for i in range(n_cls):
            for j in range(n_cls):
                color = "white" if cm[i,j]> cm.max()/2 else "black"
                axs[1,1].text(j, i, str(cm[i,j]), ha="center", va="center", color=color)
    else:
        axs[1,1].text(0.5, 0.5, "No confusion matrix data", ha="center")
    style_axes(axs[1,1])
    
    plt.tight_layout()
    save_path = os.path.join("figures", "no_hidden_layer_combined.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved No-Hidden-Layer plots to {save_path}")
except Exception as e:
    print("Error in No-Hidden-Layer plots:", str(e))
    plt.close()

###########################################################
# 4) Binary-Feature Representation (Remove Token Counts)  #
###########################################################
try:
    binary_path = "experiment_results/experiment_d179aed5c71440009b37b91dc06d29c7_proc_335109/experiment_data.npy"
    data_binary = np.load(binary_path, allow_pickle=True).item()
    # Data stored under key "binary_no_counts" -> "spr_bench"
    log = data_binary.get("binary_no_counts", {}).get("spr_bench", {})
    epochs   = log.get("epochs", [])
    train_ls = log.get("losses", {}).get("train", [])
    dev_ls   = log.get("losses", {}).get("dev", [])
    train_pha = log.get("metrics", {}).get("train_PHA", [])
    dev_pha   = log.get("metrics", {}).get("dev_PHA", [])
    test_m = log.get("test_metrics", {})
    
    fig, axs = plt.subplots(1, 3, figsize=(18,5))
    # Loss Curve
    axs[0].plot(epochs, train_ls, label="Train")
    axs[0].plot(epochs, dev_ls, label="Validation")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Binary Features: Loss Curve")
    axs[0].legend()
    style_axes(axs[0])
    
    # PHA Curve
    axs[1].plot(epochs, train_pha, label="Train PHA")
    axs[1].plot(epochs, dev_pha, label="Validation PHA")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("PHA")
    axs[1].set_title("Binary Features: PHA Curve")
    axs[1].legend()
    style_axes(axs[1])
    
    # Test Metrics Bar Chart
    if test_m:
        names = list(test_m.keys())
        vals = [test_m[n] for n in names]
        axs[2].bar(names, vals, color=["tab:blue", "tab:orange", "tab:green"])
        axs[2].set_ylim(0,1)
        axs[2].set_ylabel("Score")
        axs[2].set_title("Binary Features: Test Metrics")
        for i, v in enumerate(vals):
            axs[2].text(i, v+0.02, f"{v:.2f}", ha="center")
    else:
        axs[2].text(0.5, 0.5, "No test metrics", ha="center")
    style_axes(axs[2])
    
    plt.tight_layout()
    save_path = os.path.join("figures", "binary_feature_representation.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Binary-Feature Representation plots to {save_path}")
except Exception as e:
    print("Error in Binary-Feature Representation plots:", str(e))
    plt.close()

##################################################################
# 5) Length-Invariant Feature Normalization Ablation Plots      #
##################################################################
try:
    lin_path = "experiment_results/experiment_b9e1a2d7170d4816a31871fbe071b6a0_proc_335107/experiment_data.npy"
    data_lin = np.load(lin_path, allow_pickle=True).item()
    # In this experiment, data is iterated over; assume key "length_inv_norm" exists with "spr_bench" inside.
    log = data_lin.get("length_inv_norm", {}).get("spr_bench", {})
    epochs   = log.get("epochs", [])
    train_loss = log.get("losses", {}).get("train", [])
    dev_loss   = log.get("losses", {}).get("dev", [])
    train_pha  = log.get("metrics", {}).get("train_PHA", [])
    dev_pha    = log.get("metrics", {}).get("dev_PHA", [])
    test_metrics = log.get("test_metrics", {})
    ground_truth = np.asarray(log.get("ground_truth", []))
    predictions  = np.asarray(log.get("predictions", []))
    
    fig, axs = plt.subplots(2, 2, figsize=(14,10))
    # Loss Curve
    axs[0,0].plot(epochs, train_loss, label="Train")
    axs[0,0].plot(epochs, dev_loss, label="Validation")
    axs[0,0].set_xlabel("Epoch")
    axs[0,0].set_ylabel("Loss")
    axs[0,0].set_title("Length-Invariant Norm: Loss Curve")
    axs[0,0].legend()
    style_axes(axs[0,0])
    
    # PHA Curve
    axs[0,1].plot(epochs, train_pha, label="Train PHA")
    axs[0,1].plot(epochs, dev_pha, label="Validation PHA")
    axs[0,1].set_xlabel("Epoch")
    axs[0,1].set_ylabel("PHA")
    axs[0,1].set_title("Length-Invariant Norm: PHA Curve")
    axs[0,1].legend()
    style_axes(axs[0,1])
    
    # Confusion Matrix
    if ground_truth.size and predictions.size:
        n_cls = int(max(ground_truth.max(), predictions.max())) + 1
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(ground_truth, predictions):
            cm[t, p] += 1
        im = axs[1,0].imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=axs[1,0], fraction=0.046, pad=0.04)
        axs[1,0].set_xlabel("Predicted")
        axs[1,0].set_ylabel("True")
        axs[1,0].set_title("Length-Invariant Norm: Confusion Matrix")
        for i in range(n_cls):
            for j in range(n_cls):
                color = "white" if cm[i,j] > cm.max()/2 else "black"
                axs[1,0].text(j, i, str(cm[i,j]), ha="center", va="center", color=color)
    else:
        axs[1,0].text(0.5, 0.5, "No confusion data", ha="center")
    style_axes(axs[1,0])
    
    # Test Metrics Bar Chart
    if test_metrics:
        names = list(test_metrics.keys())
        vals = [test_metrics[n] for n in names]
        axs[1,1].bar(names, vals, color=["green", "orange", "red"])
        axs[1,1].set_ylim(0,1)
        axs[1,1].set_ylabel("Score")
        axs[1,1].set_title("Length-Invariant Norm: Test Metrics")
        for i, v in enumerate(vals):
            axs[1,1].text(i, v+0.02, f"{v:.2f}", ha="center")
    else:
        axs[1,1].text(0.5, 0.5, "No test metrics", ha="center")
    style_axes(axs[1,1])
    
    plt.tight_layout()
    save_path = os.path.join("figures", "length_invariant_norm_combined.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Length-Invariant Feature Normalization plots to {save_path}")
except Exception as e:
    print("Error in Length-Invariant Feature Normalization plots:", str(e))
    plt.close()

##################################################################
# 6) Joint-Token-Only Representation Ablation Plots              #
##################################################################
try:
    jt_path = "experiment_results/experiment_149823280dbb4b41a95b87fbf8c94302_proc_335109/experiment_data.npy"
    data_jt = np.load(jt_path, allow_pickle=True).item()
    # Data stored under key "joint_token_only" -> "spr_bench"
    log = data_jt.get("joint_token_only", {}).get("spr_bench", {})
    epochs = log.get("epochs", [])
    train_loss = log.get("losses", {}).get("train", [])
    dev_loss   = log.get("losses", {}).get("dev", [])
    train_pha  = log.get("metrics", {}).get("train_PHA", [])
    dev_pha    = log.get("metrics", {}).get("dev_PHA", [])
    test_metrics = log.get("test_metrics", {})
    
    fig, axs = plt.subplots(1, 3, figsize=(18,5))
    # Loss Curve
    axs[0].plot(epochs, train_loss, label="Train")
    axs[0].plot(epochs, dev_loss, label="Validation")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Joint-Token Only: Loss Curve")
    axs[0].legend()
    style_axes(axs[0])
    
    # PHA Curve
    axs[1].plot(epochs, train_pha, label="Train PHA")
    axs[1].plot(epochs, dev_pha, label="Validation PHA")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("PHA")
    axs[1].set_title("Joint-Token Only: PHA Curve")
    axs[1].legend()
    style_axes(axs[1])
    
    # Test Metrics Bar Chart
    if test_metrics:
        names = list(test_metrics.keys())
        vals = [test_metrics[n] for n in names]
        axs[2].bar(names, vals, color=["tab:blue", "tab:orange", "tab:green"])
        axs[2].set_ylim(0,1)
        axs[2].set_ylabel("Score")
        axs[2].set_title("Joint-Token Only: Test Metrics")
        for i, v in enumerate(vals):
            axs[2].text(i, v+0.02, f"{v:.2f}", ha="center")
    else:
        axs[2].text(0.5, 0.5, "No test metrics", ha="center")
    style_axes(axs[2])
    
    plt.tight_layout()
    save_path = os.path.join("figures", "joint_token_only_combined.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Joint-Token-Only plots to {save_path}")
except Exception as e:
    print("Error in Joint-Token-Only plots:", str(e))
    plt.close()

##################################################################
# 7) No-Early-Stopping (Fixed-Epoch Training) Ablation Plots       #
##################################################################
try:
    nes_path = "experiment_results/experiment_a6c47dccb8884e6db85f19b594fb2d42_proc_335110/experiment_data.npy"
    data_nes = np.load(nes_path, allow_pickle=True).item()
    # This file may contain multiple experiment entries; iterate over them
    # For each experiment in the file, plot Loss and PHA curves in one figure;
    # also plot a separate confusion matrix if prediction data exists.
    for exp_name, dsets in data_nes.items():
        for ds_name, log in dsets.items():
            epochs = log.get("epochs", [])
            losses = log.get("losses", {})
            metrics = log.get("metrics", {})
            preds = np.asarray(log.get("predictions", []))
            gts   = np.asarray(log.get("ground_truth", []))
            # Loss and PHA curves in one figure (2 subplots)
            fig, axs = plt.subplots(1, 2, figsize=(12,5))
            if "train" in losses and "dev" in losses:
                axs[0].plot(epochs, losses["train"], label="Train")
                axs[0].plot(epochs, losses["dev"], label="Validation")
                axs[0].set_xlabel("Epoch")
                axs[0].set_ylabel("Loss")
                axs[0].set_title(f"{ds_name} Loss Curve ({exp_name})")
                axs[0].legend()
                style_axes(axs[0])
            if "train_PHA" in metrics and "dev_PHA" in metrics:
                axs[1].plot(epochs, metrics["train_PHA"], label="Train PHA")
                axs[1].plot(epochs, metrics["dev_PHA"], label="Validation PHA")
                axs[1].set_xlabel("Epoch")
                axs[1].set_ylabel("PHA")
                axs[1].set_title(f"{ds_name} PHA Curve ({exp_name})")
                axs[1].legend()
                style_axes(axs[1])
            plt.tight_layout()
            fname = f"{ds_name}_{exp_name}_loss_and_PHA.png"
            save_path = os.path.join("figures", fname)
            plt.savefig(save_path)
            plt.close()
            print(f"Saved No-Early-Stopping loss/PHA plot to {save_path}")
            
            # Confusion Matrix in a separate figure (if data available)
            if preds.size and gts.size:
                n_cls = int(max(preds.max(), gts.max()))+1
                cm = np.zeros((n_cls, n_cls), dtype=int)
                for p, g in zip(preds, gts):
                    cm[g, p] += 1
                plt.figure(figsize=(6,5))
                im = plt.imshow(cm, cmap="Blues")
                plt.colorbar(im, fraction=0.046, pad=0.04)
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title(f"{ds_name} Confusion Matrix ({exp_name})")
                for i in range(n_cls):
                    for j in range(n_cls):
                        color = "white" if cm[i,j] > cm.max()/2 else "black"
                        plt.text(j, i, str(cm[i,j]), ha="center", va="center", color=color)
                plt.tight_layout()
                fname = f"{ds_name}_{exp_name}_confusion_matrix.png"
                save_path = os.path.join("figures", fname)
                plt.savefig(save_path)
                plt.close()
                print(f"Saved No-Early-Stopping confusion matrix to {save_path}")
except Exception as e:
    print("Error in No-Early-Stopping plots:", str(e))
    plt.close()

print("All plotting complete. Final figures are in the 'figures/' directory.")
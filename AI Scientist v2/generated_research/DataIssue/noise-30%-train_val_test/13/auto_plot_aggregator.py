#!/usr/bin/env python3
"""
Aggregator Script for Final Scientific Figures
This script aggregates and generates final publication‐quality plots
from multiple experiment summaries (Baseline, Research, and selected Ablation studies).
All figures are saved as PNG files in the "figures/" directory.
Each plot is enclosed in its own try-except block so that failure of one plot does not stop the rest.
Data is loaded from existing .npy files (full and exact paths as provided in summaries).
No synthetic or extra data is generated.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score

# Set common style parameters for publication quality
plt.rcParams.update({
    "font.size": 14,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight"
})

# Create figures directory
os.makedirs("figures", exist_ok=True)

# ------------------------
# Helper Functions
# ------------------------

def load_experiment_data(filepath):
    try:
        data = np.load(filepath, allow_pickle=True).item()
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def extract_baseline_metrics(runs):
    # Given the baseline nested dict keyed by "num_layers" (e.g., "nl_1", etc.)
    # We assume each run dict has: 'epochs', 'metrics' with keys "train_f1", "val_f1",
    # "losses" with "train" and "val", and arrays for "predictions" and "ground_truth"
    keys = sorted(runs.keys(), key=lambda x: int(x.split("_")[1]))
    epochs_dict = {}
    train_f1 = {}
    val_f1 = {}
    train_loss = {}
    val_loss = {}
    predictions = {}
    ground_truth = {}
    for key in keys:
        rec = runs[key]
        epochs_dict[key] = rec.get("epochs", [])
        train_f1[key] = np.array(rec.get("metrics", {}).get("train_f1", []))
        val_f1[key] = np.array(rec.get("metrics", {}).get("val_f1", []))
        train_loss[key] = np.array(rec.get("losses", {}).get("train", []))
        val_loss[key] = np.array(rec.get("losses", {}).get("val", []))
        predictions[key] = np.array(rec.get("predictions", []))
        ground_truth[key] = np.array(rec.get("ground_truth", []))
    return keys, epochs_dict, train_f1, val_f1, train_loss, val_loss, predictions, ground_truth

def plot_confusion_matrix(gt, preds, title, fname):
    try:
        cm = confusion_matrix(gt, preds)
        plt.figure(figsize=(6,5))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        # Label each cell
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                         color="white" if cm[i,j] > (cm.max()/2) else "black")
        plt.tight_layout()
        plt.savefig(os.path.join("figures", fname))
        plt.close()
    except Exception as e:
        print(f"Error in confusion matrix plot '{fname}': {e}")
        plt.close()

# ------------------------
# 1. BASELINE EXPERIMENT PLOTS
# ------------------------
baseline_file = "experiment_results/experiment_4b0c80e1250c48a09c7ec4ad5625b93b_proc_3464962/experiment_data.npy"  \
                if os.path.exists("experiment_results/experiment_4b0b80e1250c48a09c7ec4ad5625b93b_proc_3464962/experiment_data.npy") is False \
                else "experiment_results/experiment_4b0b80e1250c48a09c7ec4ad5625b93b_proc_3464962/experiment_data.npy"

baseline_data = load_experiment_data(baseline_file)
if baseline_data is not None and "num_layers" in baseline_data:
    runs = baseline_data["num_layers"]["SPR_BENCH"]
    keys, epochs_dict, train_f1, val_f1, train_loss, val_loss, predictions, ground_truth = extract_baseline_metrics(runs)
    
    # Plot 1: Training vs Validation Macro-F1 Curves (One figure with multiple curves)
    try:
        plt.figure()
        for key in keys:
            epochs = epochs_dict[key]
            plt.plot(epochs, train_f1[key], label=f"Train (layers {key.split('_')[1]})")
            plt.plot(epochs, val_f1[key], linestyle="--", label=f"Validation (layers {key.split('_')[1]})")
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.title("SPR_BENCH: Train vs Validation Macro F1 (Baseline)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "Baseline_MacroF1_Curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting Baseline F1 curves: {e}")
        plt.close()
    
    # Plot 2: Training vs Validation Loss Curves
    try:
        plt.figure()
        for key in keys:
            epochs = epochs_dict[key]
            plt.plot(epochs, train_loss[key], label=f"Train (layers {key.split('_')[1]})")
            plt.plot(epochs, val_loss[key], linestyle="--", label=f"Validation (layers {key.split('_')[1]})")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Train vs Validation Loss (Baseline)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "Baseline_Loss_Curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting Baseline Loss curves: {e}")
        plt.close()
    
    # Plot 3: Bar Chart of Test Macro-F1 Scores (computed from predictions)
    try:
        test_f1_scores = {}
        for key in keys:
            preds = predictions[key]
            gt = ground_truth[key]
            if preds.size and gt.size and preds.shape == gt.shape:
                test_f1_scores[key] = f1_score(gt, preds, average="macro")
        plt.figure()
        sorted_keys = sorted(test_f1_scores.keys(), key=lambda x: int(x.split("_")[1]))
        vals = [test_f1_scores[k] for k in sorted_keys]
        plt.bar([k for k in sorted_keys], vals, color="skyblue")
        plt.ylabel("Macro F1")
        plt.title("SPR_BENCH: Test Macro F1 by Num Layers (Baseline)")
        for i, v in enumerate(vals):
            plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "Baseline_TestF1_Bar.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting Baseline Test F1 bar chart: {e}")
        plt.close()
    
    # Plot 4: Confusion Matrix for Best Model (highest test F1)
    try:
        if test_f1_scores:
            best_key = max(test_f1_scores, key=test_f1_scores.get)
            preds = predictions[best_key]
            gt = ground_truth[best_key]
            plot_confusion_matrix(gt, preds,
                                  title=f"SPR_BENCH Confusion Matrix (Best: layers {best_key.split('_')[1]})",
                                  fname="Baseline_Confusion_Matrix.png")
    except Exception as e:
        print(f"Error plotting Baseline Confusion Matrix: {e}")
else:
    print("Baseline data not found or improperly formatted.")

# ------------------------
# 2. RESEARCH EXPERIMENT PLOTS (Augmented Model with Symbolic Head)
# ------------------------
research_file = "experiment_results/experiment_3eb8bea5b5774f3991a7842748719097_proc_3471777/experiment_data.npy"
research_data = load_experiment_data(research_file)
if research_data is not None:
    # Assume research_data is organized per dataset key (e.g., "SPR_BENCH")
    for dset in research_data:
        rec = research_data[dset]
        epochs = np.array(rec.get("epochs", []))
        metrics = rec.get("metrics", {})
        losses = rec.get("losses", {})
        preds = np.array(rec.get("predictions", []))
        gt = np.array(rec.get("ground_truth", []))
        # Plot 5: F1 Curve for Research experiment
        try:
            if epochs.size and metrics.get("train_f1") is not None and metrics.get("val_f1") is not None:
                plt.figure()
                plt.plot(epochs, metrics["train_f1"], label="Train")
                plt.plot(epochs, metrics["val_f1"], linestyle="--", label="Validation")
                plt.xlabel("Epoch")
                plt.ylabel("Macro F1")
                plt.title(f"{dset}: Train vs Validation Macro F1 (Research)")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join("figures", f"Research_{dset}_F1_Curve.png"))
                plt.close()
        except Exception as e:
            print(f"Error plotting Research F1 curve for {dset}: {e}")
            plt.close()
        # Plot 6: Loss Curve for Research experiment
        try:
            if epochs.size and losses.get("train") is not None and losses.get("val") is not None:
                plt.figure()
                plt.plot(epochs, losses["train"], label="Train")
                plt.plot(epochs, losses["val"], linestyle="--", label="Validation")
                plt.xlabel("Epoch")
                plt.ylabel("Cross-Entropy Loss")
                plt.title(f"{dset}: Train vs Validation Loss (Research)")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join("figures", f"Research_{dset}_Loss_Curve.png"))
                plt.close()
        except Exception as e:
            print(f"Error plotting Research Loss curve for {dset}: {e}")
            plt.close()
        # Plot 7: Confusion Matrix for Research experiment (if predictions available)
        try:
            if preds.size and gt.size and preds.shape == gt.shape:
                plot_confusion_matrix(gt, preds,
                                      title=f"{dset}: Confusion Matrix (Research)",
                                      fname=f"Research_{dset}_Confusion_Matrix.png")
        except Exception as e:
            print(f"Error plotting Research Confusion Matrix for {dset}: {e}")

# ------------------------
# 3. ABLATION EXPERIMENT PLOTS
# Selected Ablation experiments: Symbols-Only Classifier, No Positional Encoding, and CLS-Token Pooling.
# ------------------------

# 3a. Symbols-Only Classifier (Remove Transformer Branch)
symbols_only_file = "experiment_results/experiment_09cbbe63ee7446d0a30d89b8b8466a35_proc_3477729/experiment_data.npy"
symbols_only_data = load_experiment_data(symbols_only_file)
if symbols_only_data is not None:
    # The stored data is nested by (model_name, dataset_name). Iterate over them.
    for model_name in symbols_only_data:
        for dset in symbols_only_data[model_name]:
            rec = symbols_only_data[model_name][dset]
            epochs = np.array(rec.get("epochs", []))
            losses = rec.get("losses", {})
            f1_metrics = rec.get("metrics", {})
            # Plot 8: F1 Curve
            try:
                if epochs.size and f1_metrics.get("train_f1") is not None and f1_metrics.get("val_f1") is not None:
                    plt.figure()
                    plt.plot(epochs, f1_metrics["train_f1"], label="Train F1")
                    plt.plot(epochs, f1_metrics["val_f1"], linestyle="--", label="Validation F1")
                    plt.xlabel("Epoch")
                    plt.ylabel("Macro F1")
                    plt.title(f"{dset} - {model_name}: F1 Curve (Symbols-Only)")
                    plt.legend()
                    plt.tight_layout()
                    fname = f"Ablation_SymbolsOnly_{dset}_F1_Curve.png"
                    plt.savefig(os.path.join("figures", fname))
                    plt.close()
            except Exception as e:
                print(f"Error plotting Symbols-Only F1 curve for {dset}: {e}")
                plt.close()
# 3b. No Positional Encoding Ablation
no_pos_file = "experiment_results/experiment_2b5634daf14a4962a279807a9d422af9_proc_3477730/experiment_data.npy"
no_pos_data = load_experiment_data(no_pos_file)
if no_pos_data is not None and "no_positional_encoding" in no_pos_data:
    rec = no_pos_data["no_positional_encoding"]["SPR_BENCH"]
    epochs = np.array(rec.get("epochs", []))
    losses = rec.get("losses", {})
    f1_metrics = rec.get("metrics", {})
    preds = np.array(rec.get("predictions", []))
    gt = np.array(rec.get("ground_truth", []))
    # Plot 9: Combined Loss and F1 Curves (using subplots)
    try:
        fig, axs = plt.subplots(1, 2, figsize=(12,5))
        axs[0].plot(epochs, losses.get("train", []), label="Train")
        axs[0].plot(epochs, losses.get("val", []), linestyle="--", label="Validation")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("Loss Curve (No Positional Encoding)")
        axs[0].legend()
        axs[1].plot(epochs, f1_metrics.get("train", []), label="Train Macro F1")
        axs[1].plot(epochs, f1_metrics.get("val", []), linestyle="--", label="Validation Macro F1")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Macro F1")
        axs[1].set_title("Macro F1 Curve (No Positional Encoding)")
        axs[1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "Ablation_NoPositional_Combined.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting No Positional Encoding combined curves: {e}")
        plt.close()
    # Plot Confusion Matrix for No Positional Encoding if predictions available
    try:
        if preds.size and gt.size and preds.shape == gt.shape:
            plot_confusion_matrix(gt, preds,
                                  title="No Positional Encoding: Confusion Matrix",
                                  fname="Ablation_NoPositional_Confusion_Matrix.png")
    except Exception as e:
        print(f"Error plotting No Positional Encoding Confusion Matrix: {e}")
# 3c. CLS-Token Pooling Ablation
cls_token_file = "experiment_results/experiment_ad3ef1d6d44c44cbbe3c0458a75b8919_proc_3477729/experiment_data.npy"
cls_token_data = load_experiment_data(cls_token_file)
if cls_token_data is not None:
    for model_name in cls_token_data:
        for dset in cls_token_data[model_name]:
            rec = cls_token_data[model_name][dset]
            epochs = np.array(rec.get("epochs", []))
            losses = rec.get("losses", {})
            f1_metrics = rec.get("metrics", {})
            test_f1_val = f1_metrics.get("test_f1", 0)
            sga = f1_metrics.get("SGA", 0)
            preds = np.array(rec.get("predictions", []))
            gt = np.array(rec.get("ground_truth", []))
            # Plot 10: Loss, F1 and Final Metrics Bar Chart in subplots (2 rows)
            try:
                fig, axs = plt.subplots(1, 2, figsize=(12,5))
                # Left: Loss and F1 curves (use two subplots within left if desired)
                axs[0].plot(epochs, losses.get("train", []), label="Train Loss")
                axs[0].plot(epochs, losses.get("val", []), linestyle="--", label="Val Loss")
                axs[0].set_xlabel("Epoch")
                axs[0].set_ylabel("Loss")
                axs[0].set_title("Loss Curve (CLS-Token Pooling)")
                axs[0].legend()
                axs[1].plot(epochs, f1_metrics.get("train_f1", []), label="Train Macro F1")
                axs[1].plot(epochs, f1_metrics.get("val_f1", []), linestyle="--", label="Val Macro F1")
                axs[1].set_xlabel("Epoch")
                axs[1].set_ylabel("Macro F1")
                axs[1].set_title("F1 Curve (CLS-Token Pooling)")
                axs[1].legend()
                plt.tight_layout()
                plt.savefig(os.path.join("figures", f"Ablation_CLSToken_{dset}_Curves.png"))
                plt.close()
            except Exception as e:
                print(f"Error plotting CLS-Token Pooling curves for {dset}: {e}")
                plt.close()
            # Plot 11: Final Metrics Bar Chart
            try:
                plt.figure()
                bars = ["Train F1 (last)", "Val F1 (last)", "Test F1", "SGA"]
                train_last = f1_metrics.get("train_f1", [0])[-1] if f1_metrics.get("train_f1") else 0
                val_last = f1_metrics.get("val_f1", [0])[-1] if f1_metrics.get("val_f1") else 0
                vals = [train_last, val_last, test_f1_val, sga]
                plt.bar(bars, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
                plt.ylim(0, 1)
                plt.ylabel("Score")
                plt.title(f"{dset}: Final Metrics (CLS-Token Pooling)")
                plt.tight_layout()
                plt.savefig(os.path.join("figures", f"Ablation_CLSToken_{dset}_FinalMetrics.png"))
                plt.close()
            except Exception as e:
                print(f"Error plotting CLS-Token Pooling final metrics for {dset}: {e}")
                plt.close()
            # Plot 12: Confusion Matrix for CLS-Token Pooling
            try:
                if preds.size and gt.size and preds.shape == gt.shape:
                    plot_confusion_matrix(gt, preds,
                                          title=f"{dset}: Confusion Matrix (CLS-Token Pooling)",
                                          fname=f"Ablation_CLSToken_{dset}_Confusion.png")
            except Exception as e:
                print(f"Error plotting CLS-Token Pooling confusion matrix for {dset}: {e}")

print("Final plots generated and saved in the 'figures/' directory.")
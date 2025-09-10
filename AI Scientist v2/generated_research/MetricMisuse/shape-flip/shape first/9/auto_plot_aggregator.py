#!/usr/bin/env python3
"""
Final Aggregator Script for Neural-Symbolic Zero-Shot SPR
This script loads experiment .npy files (from baseline, research, and ablation experiments)
and produces final publishable figures in the 'figures/' directory.
Each figure is created using try-except blocks so that a failure in one plot does not affect the rest.
All plots are styled with larger fonts and saved at high dpi.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Increase font size for publication quality
plt.rcParams.update({"font.size": 14})


# Create figures directory
os.makedirs("figures", exist_ok=True)


def plot_baseline():
    # Baseline experiment .npy file path
    baseline_file = "experiment_results/experiment_a056e7a3fad149fb8bcf2a1977a4bbe6_proc_2797189/experiment_data.npy"
    try:
        data = np.load(baseline_file, allow_pickle=True).item()
        # Data structure: data["batch_size"]["SPR_BENCH"] is a dict keyed by batch size.
        spr_runs = data.get("batch_size", {}).get("SPR_BENCH", {})
        if not spr_runs:
            raise KeyError("Missing 'batch_size' -> 'SPR_BENCH' in baseline data.")
        batch_sizes = sorted(spr_runs.keys(), key=lambda x: int(x))
    except Exception as e:
        print(f"[Baseline] Error loading baseline data: {e}")
        return

    # Plot 1: Loss Curves (Train vs. Val) for Baseline
    try:
        plt.figure(figsize=(6, 4), dpi=300)
        for bs in batch_sizes:
            run = spr_runs[bs]
            epochs = np.arange(1, len(run["losses"]["train"]) + 1)
            plt.plot(epochs, run["losses"]["train"], label=f"Train (batch {bs})")
            plt.plot(epochs, run["losses"]["val"], linestyle="--", label=f"Val (batch {bs})")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Baseline: Loss Curves (Train vs. Val)")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join("figures", "baseline_loss_curves.png")
        plt.savefig(fname)
        plt.close()
        print(f"[Baseline] Saved Loss Curves to {fname}")
    except Exception as e:
        print(f"[Baseline] Error plotting loss curves: {e}")
        plt.close()

    # Plot 2: Validation HWA Curves for Baseline
    try:
        plt.figure(figsize=(6, 4), dpi=300)
        for bs in batch_sizes:
            run = spr_runs[bs]
            epochs = np.arange(1, len(run["metrics"]["val"]) + 1)
            plt.plot(epochs, run["metrics"]["val"], label=f"batch {bs}")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.title("Baseline: Validation HWA over Epochs")
        plt.ylim(0, 1.05)
        plt.legend()
        plt.tight_layout()
        fname = os.path.join("figures", "baseline_val_HWA.png")
        plt.savefig(fname)
        plt.close()
        print(f"[Baseline] Saved Validation HWA plot to {fname}")
    except Exception as e:
        print(f"[Baseline] Error plotting validation HWA: {e}")
        plt.close()

    # Plot 3: Test HWA vs Batch Size (Bar Chart) for Baseline
    try:
        plt.figure(figsize=(5, 3), dpi=300)
        test_hw_array = []
        for bs in batch_sizes:
            run = spr_runs[bs]
            test_hw_array.append(run["test_metrics"])
        plt.bar(range(len(batch_sizes)), test_hw_array, tick_label=batch_sizes)
        plt.xlabel("Batch Size")
        plt.ylabel("Test HWA")
        plt.title("Baseline: Test HWA vs. Batch Size")
        plt.ylim(0, 1.05)
        plt.tight_layout()
        fname = os.path.join("figures", "baseline_test_HWA_vs_batch.png")
        plt.savefig(fname)
        plt.close()
        print(f"[Baseline] Saved Test HWA vs Batch Size plot to {fname}")
    except Exception as e:
        print(f"[Baseline] Error plotting test HWA vs batch size: {e}")
        plt.close()


def plot_research():
    # Research experiment .npy file path
    research_file = "experiment_results/experiment_06791fdb19e546538a08387dc7e74f05_proc_2799577/experiment_data.npy"
    try:
        data = np.load(research_file, allow_pickle=True).item()
        spr = data.get("SPR_BENCH", None)
        if spr is None:
            raise KeyError("Missing 'SPR_BENCH' key.")
        epochs = np.arange(1, len(spr["losses"]["train"]) + 1)
    except Exception as e:
        print(f"[Research] Error loading research data: {e}")
        return

    # Plot 4: Loss Curves for Research
    try:
        plt.figure(figsize=(6, 4), dpi=300)
        plt.plot(epochs, spr["losses"]["train"], label="Train")
        plt.plot(epochs, spr["losses"]["val"], linestyle="--", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Research: SPR_BENCH Loss Curves")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join("figures", "research_loss_curves.png")
        plt.savefig(fname)
        plt.close()
        print(f"[Research] Saved Loss Curves plot to {fname}")
    except Exception as e:
        print(f"[Research] Error plotting loss curves: {e}")
        plt.close()

    # Plot 5: Validation SWA Curve for Research
    try:
        plt.figure(figsize=(6, 4), dpi=300)
        plt.plot(epochs, spr["metrics"]["val"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        plt.title("Research: Validation Shape-Weighted Accuracy")
        plt.ylim(0, 1.05)
        plt.tight_layout()
        fname = os.path.join("figures", "research_val_SWA.png")
        plt.savefig(fname)
        plt.close()
        print(f"[Research] Saved Validation SWA plot to {fname}")
    except Exception as e:
        print(f"[Research] Error plotting validation SWA: {e}")
        plt.close()

    # Plot 6: Confusion Matrix for Research Test Set
    try:
        preds = np.array(spr["predictions"])
        gts = np.array(spr["ground_truth"])
        classes = sorted(set(gts))
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure(figsize=(4, 4), dpi=300)
        im = plt.imshow(cm, cmap="Blues")
        plt.title("Research: Confusion Matrix (Test Set)")
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.colorbar(im)
        plt.xticks(classes)
        plt.yticks(classes)
        for i in range(len(classes)):
            for j in range(len(classes)):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                plt.text(j, i, cm[i, j], ha="center", va="center", color=color)
        plt.tight_layout()
        fname = os.path.join("figures", "research_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
        print(f"[Research] Saved Confusion Matrix plot to {fname}")
    except Exception as e:
        print(f"[Research] Error plotting confusion matrix: {e}")
        plt.close()


def plot_ablation_no_symbolic():
    # Ablation: No-Symbolic-Branch (Neural-Only)
    file_path = "experiment_results/experiment_a644eafe7727401995b793449d9178d0_proc_2801222/experiment_data.npy"
    try:
        data = np.load(file_path, allow_pickle=True).item()
        run = data.get("NO_SYMBOLIC_BRANCH", {}).get("SPR_BENCH", None)
        if run is None:
            raise KeyError("Missing NO_SYMBOLIC_BRANCH -> SPR_BENCH data.")
        epochs = np.arange(1, len(run["losses"]["train"]) + 1)
    except Exception as e:
        print(f"[Ablation Neural-Only] Error loading data: {e}")
        return

    # Figure with 3 subplots: Loss Curves, Val SWA, Confusion Matrix
    try:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

        # Subplot 1: Loss Curves
        axs[0].plot(epochs, run["losses"]["train"], label="Train")
        axs[0].plot(epochs, run["losses"]["val"], label="Validation", linestyle="--")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("Neural-Only: Loss Curves")
        axs[0].legend()

        # Subplot 2: Validation SWA Curve
        axs[1].plot(epochs, run["metrics"]["val"], marker="o", color="green")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("SWA")
        axs[1].set_title("Neural-Only: Validation SWA")

        # Subplot 3: Confusion Matrix (Test)
        preds = np.array(run["predictions"])
        gts = np.array(run["ground_truth"])
        classes = sorted(set(gts))
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        im = axs[2].imshow(cm, cmap="Blues")
        axs[2].set_title("Neural-Only: Confusion Matrix")
        axs[2].set_xlabel("Predicted")
        axs[2].set_ylabel("Ground Truth")
        for i in range(len(classes)):
            for j in range(len(classes)):
                color = "white" if cm[i,j] > cm.max()/2 else "black"
                axs[2].text(j, i, cm[i,j], ha="center", va="center", color=color)
        fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        fname = os.path.join("figures", "ablation_neural_only.png")
        plt.savefig(fname)
        plt.close()
        print(f"[Ablation Neural-Only] Saved aggregated figure to {fname}")
    except Exception as e:
        print(f"[Ablation Neural-Only] Error plotting aggregated figure: {e}")
        plt.close()


def plot_ablation_no_neural():
    # Ablation: No-Neural-Branch (Symbolic-Only)
    file_path = "experiment_results/experiment_8b26ddd74ee94cf094dc9672d840a998_proc_2801223/experiment_data.npy"
    try:
        data = np.load(file_path, allow_pickle=True).item()
        run = data.get("No-Neural-Branch (Symbolic-Only)", {}).get("SPR_BENCH", None)
        if run is None:
            raise KeyError("Missing No-Neural-Branch (Symbolic-Only) -> SPR_BENCH data.")
        epochs = np.arange(1, len(run["losses"]["train"]) + 1)
    except Exception as e:
        print(f"[Ablation Symbolic-Only] Error loading data: {e}")
        return

    # Figure with 3 subplots: Loss Curve, Accuracy Curve, Class Distribution
    try:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

        # Subplot 1: Loss Curves
        axs[0].plot(epochs, run["losses"]["train"], label="Train")
        axs[0].plot(epochs, run["losses"]["val"], label="Val", linestyle="--")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("Symbolic-Only: Loss Curves")
        axs[0].legend()

        # Subplot 2: Accuracy Curve
        axs[1].plot(epochs, run["metrics"]["val"], marker="o", color="orange")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("SWA")
        axs[1].set_title("Symbolic-Only: Validation SWA")

        # Subplot 3: Class Distribution (Bar Plot)
        preds = run["predictions"]
        gts = run["ground_truth"]
        uniq = sorted(set(gts))
        gt_counts = [gts.count(c) for c in uniq]
        pred_counts = [preds.count(c) for c in uniq]
        x = np.arange(len(uniq))
        width = 0.35
        axs[2].bar(x - width/2, gt_counts, width, label="GT")
        axs[2].bar(x + width/2, pred_counts, width, label="Pred")
        axs[2].set_xlabel("Class")
        axs[2].set_title("Symbolic-Only: Class Distribution")
        axs[2].set_xticks(x)
        axs[2].set_xticklabels(uniq)
        axs[2].legend()

        plt.tight_layout()
        fname = os.path.join("figures", "ablation_symbolic_only.png")
        plt.savefig(fname)
        plt.close()
        print(f"[Ablation Symbolic-Only] Saved aggregated figure to {fname}")
    except Exception as e:
        print(f"[Ablation Symbolic-Only] Error plotting aggregated figure: {e}")
        plt.close()


def plot_ablation_no_projection():
    # Ablation: No-Projection-Layers (Direct Feature Concatenation)
    file_path = "experiment_results/experiment_f38611b41506488689b52010e1c090d0_proc_2801224/experiment_data.npy"
    try:
        data = np.load(file_path, allow_pickle=True).item()
        run = data.get("NoProj", {}).get("SPR_BENCH", None)
        if run is None:
            raise KeyError("Missing NoProj -> SPR_BENCH data.")
        epochs = np.arange(1, len(run["losses"]["train"]) + 1)
    except Exception as e:
        print(f"[Ablation No-Projection] Error loading data: {e}")
        return

    # Figure with 3 subplots: Loss, Val Metric Curve, Confusion Matrix
    try:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
        # Subplot 1: Loss Curves
        axs[0].plot(epochs, run["losses"]["train"], label="Train")
        axs[0].plot(epochs, run["losses"]["val"], label="Val", linestyle="--")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("No-Projection: Loss Curves")
        axs[0].legend()

        # Subplot 2: Validation Metric Curve
        axs[1].plot(epochs, run["metrics"]["val"], marker="o", color="magenta")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("SWA")
        axs[1].set_title("No-Projection: Val SWA Curve")

        # Subplot 3: Confusion Matrix
        preds = np.array(run["predictions"])
        gts = np.array(run["ground_truth"])
        uniq = sorted(set(gts))
        cm = np.zeros((len(uniq), len(uniq)), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        im = axs[2].imshow(cm, cmap="Blues")
        axs[2].set_title("No-Projection: Confusion Matrix")
        axs[2].set_xlabel("Predicted")
        axs[2].set_ylabel("GT")
        for i in range(len(uniq)):
            for j in range(len(uniq)):
                color = "white" if cm[i,j]>cm.max()/2 else "black"
                axs[2].text(j, i, cm[i,j], ha="center", va="center", color=color)
        fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        fname = os.path.join("figures", "ablation_no_projection.png")
        plt.savefig(fname)
        plt.close()
        print(f"[Ablation No-Projection] Saved aggregated figure to {fname}")
    except Exception as e:
        print(f"[Ablation No-Projection] Error plotting aggregated figure: {e}")
        plt.close()


def plot_ablation_unidirectional():
    # Ablation: Unidirectional-GRU
    file_path = "experiment_results/experiment_a5bc15b1045746629a2cccf37204e82a_proc_2801225/experiment_data.npy"
    try:
        data = np.load(file_path, allow_pickle=True).item()
        run = data.get("unidirectional_gru", {}).get("SPR_BENCH", None)
        if run is None:
            raise KeyError("Missing unidirectional_gru -> SPR_BENCH data.")
        epochs = np.arange(1, len(run["losses"]["train"]) + 1)
    except Exception as e:
        print(f"[Ablation Unidirectional] Error loading data: {e}")
        return

    # Figure with 3 subplots: Loss, Val SWA, and Class Distribution
    try:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
        # Subplot 1: Loss Curves
        axs[0].plot(epochs, run["losses"]["train"], label="Train")
        axs[0].plot(epochs, run["losses"]["val"], label="Val", linestyle="--")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("Unidirectional-GRU: Loss Curves")
        axs[0].legend()

        # Subplot 2: Validation SWA
        axs[1].plot(epochs, run["metrics"]["val"], marker="o", color="brown")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("SWA")
        axs[1].set_title("Unidirectional-GRU: Validation SWA")

        # Subplot 3: Class Distribution (Bar Plot)
        preds = run.get("predictions", [])
        gts = run.get("ground_truth", [])
        uniq = sorted(set(gts))
        pred_counts = [sum(1 for p in preds if p == c) for c in uniq]
        gt_counts = [sum(1 for t in gts if t == c) for c in uniq]
        x = np.arange(len(uniq))
        width = 0.35
        axs[2].bar(x - width/2, gt_counts, width, label="GT")
        axs[2].bar(x + width/2, pred_counts, width, label="Pred")
        axs[2].set_xlabel("Class")
        axs[2].set_title("Unidirectional-GRU: Class Distribution")
        axs[2].set_xticks(x)
        axs[2].set_xticklabels(uniq)
        axs[2].legend()

        plt.tight_layout()
        fname = os.path.join("figures", "ablation_unidirectional_gru.png")
        plt.savefig(fname)
        plt.close()
        print(f"[Ablation Unidirectional] Saved aggregated figure to {fname}")
    except Exception as e:
        print(f"[Ablation Unidirectional] Error plotting aggregated figure: {e}")
        plt.close()


def plot_ablation_binary_symbolic():
    # Ablation: Binary-Symbolic-Features
    file_path = "experiment_results/experiment_6e4a40bd9174495eb018acfb99736710_proc_2801223/experiment_data.npy"
    try:
        data = np.load(file_path, allow_pickle=True).item()
        exp = data.get("BinarySymFeat", {}).get("SPR_BENCH", None)
        if exp is None:
            raise KeyError("Missing BinarySymFeat -> SPR_BENCH data.")
        epochs = np.arange(1, len(exp["losses"]["train"]) + 1)
    except Exception as e:
        print(f"[Ablation Binary-Symbolic] Error loading data: {e}")
        return

    # Figure with 3 subplots: Loss Curves, Val Accuracy, GT vs Pred Distribution
    try:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
        # Subplot 1: Loss Curves
        axs[0].plot(epochs, exp["losses"]["train"], label="Train")
        axs[0].plot(epochs, exp["losses"]["val"], label="Val", linestyle="--")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("Binary-Symbolic: Loss Curves")
        axs[0].legend()
        
        # Subplot 2: Validation SWA Curve
        vals = exp["metrics"]["val"]
        axs[1].plot(epochs, vals, marker="o", color="teal")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("SWA")
        axs[1].set_title("Binary-Symbolic: Val SWA")
        
        # Subplot 3: GT vs Prediction Distribution (Bar Plot)
        preds = exp.get("predictions", [])
        gts = exp.get("ground_truth", [])
        uniq = sorted(set(gts))
        gt_counts = [gts.count(c) for c in uniq]
        pred_counts = [preds.count(c) for c in uniq]
        x = np.arange(len(uniq))
        width = 0.35
        axs[2].bar(x - width/2, gt_counts, width, label="GT")
        axs[2].bar(x + width/2, pred_counts, width, label="Pred")
        axs[2].set_xlabel("Class")
        axs[2].set_title("Binary-Symbolic: GT vs Pred")
        axs[2].set_xticks(x)
        axs[2].set_xticklabels(uniq)
        axs[2].legend()
        
        plt.tight_layout()
        fname = os.path.join("figures", "ablation_binary_symbolic.png")
        plt.savefig(fname)
        plt.close()
        print(f"[Ablation Binary-Symbolic] Saved aggregated figure to {fname}")
    except Exception as e:
        print(f"[Ablation Binary-Symbolic] Error plotting aggregated figure: {e}")
        plt.close()


def plot_ablation_late_fusion():
    # Ablation: Late-Fusion (Separate Logit Heads Averaged)
    file_path = "experiment_results/experiment_849399ec49544519afc0b1de9a42d650_proc_2801225/experiment_data.npy"
    try:
        data = np.load(file_path, allow_pickle=True).item()
        ed = data.get("LateFusion_LogitsAvg", {}).get("SPR_BENCH", None)
        if ed is None:
            raise KeyError("Missing LateFusion_LogitsAvg -> SPR_BENCH data.")
        epochs = np.arange(1, len(ed["losses"]["train"]) + 1)
    except Exception as e:
        print(f"[Ablation Late-Fusion] Error loading data: {e}")
        return

    # Figure with 3 subplots: Loss Curves, Val SWA, Confusion Matrix
    try:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
        # Subplot 1: Loss Curves
        axs[0].plot(epochs, ed["losses"]["train"], label="Train")
        axs[0].plot(epochs, ed["losses"]["val"], label="Val", linestyle="--")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("Late-Fusion: Loss Curves")
        axs[0].legend()
        
        # Subplot 2: Validation SWA
        axs[1].plot(epochs, ed["metrics"]["val"], marker="o", color="navy")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("SWA")
        axs[1].set_title("Late-Fusion: Val SWA")
        
        # Subplot 3: Confusion Matrix
        preds = np.array(ed["predictions"])
        gts = np.array(ed["ground_truth"])
        uniq = sorted(set(gts))
        cm = np.zeros((len(uniq), len(uniq)), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        im = axs[2].imshow(cm, cmap="Blues")
        axs[2].set_title("Late-Fusion: Confusion Matrix")
        axs[2].set_xlabel("Predicted")
        axs[2].set_ylabel("GT")
        for i in range(len(uniq)):
            for j in range(len(uniq)):
                color = "white" if cm[i,j] > cm.max()/2 else "black"
                axs[2].text(j, i, cm[i,j], ha="center", va="center", color=color)
        fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        fname = os.path.join("figures", "ablation_late_fusion.png")
        plt.savefig(fname)
        plt.close()
        print(f"[Ablation Late-Fusion] Saved aggregated figure to {fname}")
    except Exception as e:
        print(f"[Ablation Late-Fusion] Error plotting aggregated figure: {e}")
        plt.close()


def main():
    print("Generating final figures ...")
    plot_baseline()
    plot_research()
    plot_ablation_no_symbolic()
    plot_ablation_no_neural()
    plot_ablation_no_projection()
    plot_ablation_unidirectional()
    plot_ablation_binary_symbolic()
    plot_ablation_late_fusion()
    print("Figure generation completed. Figures are saved in the 'figures/' directory.")

if __name__ == "__main__":
    main()
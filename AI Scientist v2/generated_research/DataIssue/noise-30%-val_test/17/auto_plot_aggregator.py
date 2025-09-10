#!/usr/bin/env python3
"""
Aggregator script for final scientific plots for
"Developing Robust Algorithms for Symbolic PolyRule Reasoning".
This script loads experiment results from existing .npy files and produces 
final, polished plots saved exclusively in the "figures/" folder.
Each plot is wrapped in try-except so a failure in one does not affect others.

Author: AI Researcher
Date: 2023-10-XX
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix

# Set high-quality fonts for publication
plt.rcParams.update({'font.size': 14})
DPI = 300

# Create figures directory
os.makedirs("figures", exist_ok=True)

# Helper: Remove top/right spines (for a professional look)
def style_axes(ax):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

# Helper to annotate bars
def annotate_bars(ax, values):
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}" if isinstance(v, float) else f"{v}", ha="center")

# ----------------------- Baseline Plots -----------------------
def plot_baseline():
    # Baseline npy file from summary
    npy_path = "experiment_results/experiment_b9a30e0c00cd4e55acef01fc62a846c2_proc_3331034/experiment_data.npy"
    try:
        data = np.load(npy_path, allow_pickle=True).item()
    except Exception as e:
        print(f"Baseline: Failed to load data from {npy_path}: {e}")
        return

    # Iterate over keys in the baseline experiment_data (assume one key)
    for key, rec in data.items():
        # 1) Loss Curves (Baseline)
        try:
            fig, ax = plt.subplots(figsize=(8, 6), dpi=DPI)
            epochs = range(1, len(rec["losses"]["train"]) + 1)
            ax.plot(epochs, rec["losses"]["train"], label="Train")
            ax.plot(epochs, rec["losses"]["val"], label="Validation")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("BCE Loss")
            ax.set_title("Baseline: Training vs Validation Loss")
            ax.legend()
            style_axes(ax)
            fname = os.path.join("figures", "Baseline_Loss_Curves.png")
            fig.tight_layout()
            fig.savefig(fname)
            plt.close(fig)
            print(f"Saved Baseline loss curves to {fname}")
        except Exception as e:
            print(f"Baseline Loss Curves plot error: {e}")

        # 2) Macro-F1 Curves (Baseline)
        try:
            fig, ax = plt.subplots(figsize=(8, 6), dpi=DPI)
            epochs = range(1, len(rec["metrics"]["train"]) + 1)
            ax.plot(epochs, rec["metrics"]["train"], label="Train")
            ax.plot(epochs, rec["metrics"]["val"], label="Validation")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Macro F1")
            ax.set_title("Baseline: Training vs Validation Macro F1")
            ax.legend()
            style_axes(ax)
            fname = os.path.join("figures", "Baseline_Macro_F1_Curves.png")
            fig.tight_layout()
            fig.savefig(fname)
            plt.close(fig)
            print(f"Saved Baseline F1 curves to {fname}")
        except Exception as e:
            print(f"Baseline Macro-F1 Curves plot error: {e}")

        # 3) Test Metrics Bar Chart (Baseline)
        try:
            # Use first test run predictions/ground-truth
            preds = np.array(rec["predictions"][0]).flatten()
            gts = np.array(rec["ground_truth"][0]).flatten()
            test_f1 = f1_score(gts, preds, average="macro")
            test_mcc = matthews_corrcoef(gts, preds)
            fig, ax = plt.subplots(figsize=(6, 6), dpi=DPI)
            ax.bar(["Macro F1", "MCC"], [test_f1, test_mcc], color=["steelblue", "orange"])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Score")
            ax.set_title("Baseline: Test Metrics Comparison")
            annotate_bars(ax, [test_f1, test_mcc])
            style_axes(ax)
            fig.tight_layout()
            fname = os.path.join("figures", "Baseline_Test_Metrics.png")
            fig.savefig(fname)
            plt.close(fig)
            print(f"Saved Baseline test metrics bar chart to {fname}")
        except Exception as e:
            print(f"Baseline Test Metrics plot error: {e}")

# ----------------------- Research Plots -----------------------
def plot_research():
    npy_path = "experiment_results/experiment_200a680eec6d46a79285fcb55b86f436_proc_3335769/experiment_data.npy"
    try:
        data = np.load(npy_path, allow_pickle=True).item()
    except Exception as e:
        print(f"Research: Failed to load data from {npy_path}: {e}")
        return
    
    for key, rec in data.items():
        # Combined Plot: Loss and MCC Curves side-by-side
        try:
            fig, axs = plt.subplots(1, 2, figsize=(14, 6), dpi=DPI)
            epochs = range(1, len(rec["losses"]["train"]) + 1)
            # Loss curves
            axs[0].plot(epochs, rec["losses"]["train"], label="Train Loss")
            axs[0].plot(epochs, rec["losses"]["val"], label="Validation Loss")
            axs[0].set_xlabel("Epoch")
            axs[0].set_ylabel("BCE Loss")
            axs[0].set_title("Research: Loss Curves")
            axs[0].legend()
            style_axes(axs[0])
            # MCC curves (using metrics as MCC in research code)
            axs[1].plot(epochs, rec["metrics"]["train"], label="Train MCC")
            axs[1].plot(epochs, rec["metrics"]["val"], label="Validation MCC")
            axs[1].set_xlabel("Epoch")
            axs[1].set_ylabel("MCC")
            axs[1].set_title("Research: MCC Curves")
            axs[1].legend()
            style_axes(axs[1])
            fig.tight_layout()
            fname = os.path.join("figures", "Research_Loss_and_MCC_Curves.png")
            fig.savefig(fname)
            plt.close(fig)
            print(f"Saved Research combined loss and MCC curves to {fname}")
        except Exception as e:
            print(f"Research combined loss/MCC curves plot error: {e}")

        # Confusion Matrix Heatmap (Research)
        try:
            preds = np.array(rec["predictions"][0]).flatten()
            gts = np.array(rec["ground_truth"][0]).flatten()
            cm = confusion_matrix(gts, preds, labels=[0, 1])
            fig, ax = plt.subplots(figsize=(6, 6), dpi=DPI)
            cax = ax.imshow(cm, cmap="Blues")
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, str(v), ha="center", va="center", color="black")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Pred 0", "Pred 1"])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["True 0", "True 1"])
            ax.set_title("Research: Confusion Matrix")
            fig.colorbar(cax, ax=ax)
            style_axes(ax)
            fig.tight_layout()
            fname = os.path.join("figures", "Research_Confusion_Matrix.png")
            fig.savefig(fname)
            plt.close(fig)
            print(f"Saved Research confusion matrix to {fname}")
        except Exception as e:
            print(f"Research confusion matrix plot error: {e}")

# ----------------------- Ablation Plots: No-Position-Embedding -----------------------
def plot_ablation_no_pos_embedding():
    npy_path = "experiment_results/experiment_84e9deb830144dcdb3a17e460e1c4ef7_proc_3341509/experiment_data.npy"
    try:
        data = np.load(npy_path, allow_pickle=True).item()
        rec = data.get("no_pos_embedding", {}).get("SPR_BENCH", None)
        if rec is None:
            raise ValueError("No data for 'no_pos_embedding' in SPR_BENCH")
    except Exception as e:
        print(f"No-Position-Embedding Ablation: Failed to load data from {npy_path}: {e}")
        return

    # 1) Loss Curves
    try:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=DPI)
        epochs = np.arange(1, len(rec["losses"]["train"]) + 1)
        ax.plot(epochs, rec["losses"]["train"], label="Train")
        ax.plot(epochs, rec["losses"]["val"], label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE Loss")
        ax.set_title("Ablation (No Pos Embedding): Loss Curves")
        ax.legend()
        style_axes(ax)
        fig.tight_layout()
        fname = os.path.join("figures", "Ablation_NoPosEmbedding_Loss_Curves.png")
        fig.savefig(fname)
        plt.close(fig)
        print(f"Saved No-Position-Embedding loss curves to {fname}")
    except Exception as e:
        print(f"No-Position-Embedding Loss Curves plot error: {e}")

    # 2) MCC Curves
    try:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=DPI)
        epochs = np.arange(1, len(rec["metrics"]["train"]) + 1)
        ax.plot(epochs, rec["metrics"]["train"], label="Train")
        ax.plot(epochs, rec["metrics"]["val"], label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MCC")
        ax.set_title("Ablation (No Pos Embedding): MCC Curves")
        ax.legend()
        style_axes(ax)
        fig.tight_layout()
        fname = os.path.join("figures", "Ablation_NoPosEmbedding_MCC_Curves.png")
        fig.savefig(fname)
        plt.close(fig)
        print(f"Saved No-Position-Embedding MCC curves to {fname}")
    except Exception as e:
        print(f"No-Position-Embedding MCC Curves plot error: {e}")

    # 3) Test MCC Bar Chart (Compute per-run MCC using helper function)
    def compute_mcc(y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        denom = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        return 0.0 if denom==0 else (tp*tn - fp*fn)/denom

    try:
        test_mccs = []
        for p, g in zip(rec["predictions"], rec["ground_truth"]):
            test_mccs.append(compute_mcc(np.asarray(g), np.asarray(p)))
        fig, ax = plt.subplots(figsize=(8, 6), dpi=DPI)
        indices = np.arange(len(test_mccs))
        ax.bar(indices, test_mccs, color="skyblue")
        ax.set_xlabel("Run Index")
        ax.set_ylabel("Test MCC")
        ax.set_title("Ablation (No Pos Embedding): Test MCC per Run")
        ax.set_ylim(0, 1)
        annotate_bars(ax, test_mccs)
        style_axes(ax)
        fig.tight_layout()
        fname = os.path.join("figures", "Ablation_NoPosEmbedding_Test_MCC_Bar.png")
        fig.savefig(fname)
        plt.close(fig)
        print(f"Saved No-Position-Embedding test MCC bar chart to {fname}")
    except Exception as e:
        print(f"No-Position-Embedding Test MCC Bar plot error: {e}")

# ----------------------- Ablation Plots: No-Class-Weight Loss -----------------------
def plot_ablation_no_class_weight():
    npy_path = "experiment_results/experiment_7ae571fc5f1348f1ad032752650779ed_proc_3341511/experiment_data.npy"
    try:
        data = np.load(npy_path, allow_pickle=True).item()
        rec = data.get("no_class_weight", {}).get("SPR_BENCH", None)
        if rec is None:
            raise ValueError("No data for 'no_class_weight' in SPR_BENCH")
    except Exception as e:
        print(f"No-Class-Weight Loss Ablation: Failed to load data from {npy_path}: {e}")
        return

    # 1) Loss Curves
    try:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=DPI)
        ax.plot(rec["losses"]["train"], label="Train")
        ax.plot(rec["losses"]["val"], label="Validation")
        ax.set_xlabel("Epoch Index")
        ax.set_ylabel("BCE Loss")
        ax.set_title("Ablation (No Class Weight): Loss Curves")
        ax.legend()
        style_axes(ax)
        fig.tight_layout()
        fname = os.path.join("figures", "Ablation_NoClassWeight_Loss_Curves.png")
        fig.savefig(fname)
        plt.close(fig)
        print(f"Saved No-Class-Weight loss curves to {fname}")
    except Exception as e:
        print(f"No-Class-Weight Loss Curves plot error: {e}")

    # 2) MCC Curves
    try:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=DPI)
        ax.plot(rec["metrics"]["train"], label="Train")
        ax.plot(rec["metrics"]["val"], label="Validation")
        ax.set_xlabel("Epoch Index")
        ax.set_ylabel("MCC")
        ax.set_title("Ablation (No Class Weight): MCC Curves")
        ax.legend()
        style_axes(ax)
        fig.tight_layout()
        fname = os.path.join("figures", "Ablation_NoClassWeight_MCC_Curves.png")
        fig.savefig(fname)
        plt.close(fig)
        print(f"Saved No-Class-Weight MCC curves to {fname}")
    except Exception as e:
        print(f"No-Class-Weight MCC Curves plot error: {e}")

    # 3) Confusion Matrix Bars (for most recent run)
    try:
        preds = rec["predictions"][-1].astype(int)
        gts = rec["ground_truth"][-1].astype(int)
        tp = int(((preds == 1) & (gts == 1)).sum())
        fp = int(((preds == 1) & (gts == 0)).sum())
        tn = int(((preds == 0) & (gts == 0)).sum())
        fn = int(((preds == 0) & (gts == 1)).sum())
        fig, ax = plt.subplots(figsize=(8, 6), dpi=DPI)
        ax.bar(["TP", "FP", "FN", "TN"], [tp, fp, fn, tn], color=["g", "r", "r", "g"])
        ax.set_ylabel("Count")
        ax.set_title("Ablation (No Class Weight): Confusion Counts")
        style_axes(ax)
        fig.tight_layout()
        fname = os.path.join("figures", "Ablation_NoClassWeight_Confusion_Bars.png")
        fig.savefig(fname)
        plt.close(fig)
        print(f"Saved No-Class-Weight confusion matrix bars to {fname}")
    except Exception as e:
        print(f"No-Class-Weight Confusion Matrix Bars plot error: {e}")

# ----------------------- Appendix Plots -----------------------
def plot_appendix_no_weight_decay():
    npy_path = "experiment_results/experiment_a3249688c40740fead4c19c896e8f5c6_proc_3341509/experiment_data.npy"
    try:
        data = np.load(npy_path, allow_pickle=True).item()
        rec = data.get("NoWeightDecay", {}).get("SPR_BENCH", None)
        if rec is None:
            raise ValueError("No data for 'NoWeightDecay' in SPR_BENCH")
    except Exception as e:
        print(f"No Weight Decay Ablation: Failed to load data from {npy_path}: {e}")
        return

    # Confusion Matrix Bars for last run
    try:
        preds = rec["predictions"][-1].astype(int)
        gts = rec["ground_truth"][-1].astype(int)
        tp = int(((preds == 1) & (gts == 1)).sum())
        fp = int(((preds == 1) & (gts == 0)).sum())
        tn = int(((preds == 0) & (gts == 0)).sum())
        fn = int(((preds == 0) & (gts == 1)).sum())
        fig, ax = plt.subplots(figsize=(8, 6), dpi=DPI)
        ax.bar(["TP", "FP", "TN", "FN"], [tp, fp, tn, fn], color=["g", "r", "b", "orange"])
        ax.set_ylabel("Count")
        ax.set_title("Appendix (No Weight Decay): Confusion Bars")
        style_axes(ax)
        fig.tight_layout()
        fname = os.path.join("figures", "Appendix_NoWeightDecay_Confusion_Bars.png")
        fig.savefig(fname)
        plt.close(fig)
        print(f"Saved No Weight Decay confusion bars to {fname}")
    except Exception as e:
        print(f"No Weight Decay Confusion Bars plot error: {e}")

def plot_appendix_fixed_sinusoidal():
    npy_path = "experiment_results/experiment_954f6265a2cc4fe494ca2b528c099270_proc_3341512/experiment_data.npy"
    try:
        data = np.load(npy_path, allow_pickle=True).item()
        # Data structure: iterate over model names and datasets; use first available.
        model_key = list(data.keys())[0]
        dset_key = list(data[model_key].keys())[0]
        rec = data[model_key][dset_key]
    except Exception as e:
        print(f"Fixed-Sinusoidal Ablation: Failed to load data from {npy_path}: {e}")
        return

    # Loss Curves for fixed sinusoidal positional encoding
    try:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=DPI)
        epochs = range(1, len(rec["losses"]["train"]) + 1)
        ax.plot(epochs, rec["losses"]["train"], label="Train Loss")
        ax.plot(epochs, rec["losses"]["val"], label="Val Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE Loss")
        ax.set_title("Appendix (Fixed Sinusoidal): Loss Curves")
        ax.legend()
        style_axes(ax)
        fig.tight_layout()
        fname = os.path.join("figures", "Appendix_FixedSinusoidal_Loss_Curves.png")
        fig.savefig(fname)
        plt.close(fig)
        print(f"Saved Fixed-Sinusoidal loss curves to {fname}")
    except Exception as e:
        print(f"Fixed-Sinusoidal Loss Curves plot error: {e}")

# ----------------------- Main Function -----------------------
def main():
    print("Generating final publication figures...")
    plot_baseline()
    plot_research()
    plot_ablation_no_pos_embedding()
    plot_ablation_no_class_weight()
    # Appendix plots (synthetic/extended results)
    plot_appendix_no_weight_decay()
    plot_appendix_fixed_sinusoidal()
    print("All final plots have been saved in the 'figures/' folder.")

if __name__ == "__main__":
    main()
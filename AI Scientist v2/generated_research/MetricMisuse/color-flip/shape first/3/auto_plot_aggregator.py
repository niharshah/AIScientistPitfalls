#!/usr/bin/env python3
"""
Final Aggregator Script for Publishing Figures

This script loads experimental results (stored as .npy files) from baseline, research,
and selected ablation experiments. It then produces a comprehensive set of final scientific
plots for the paper and additional supplemental plots (in the Appendix).

All final figures are saved in the "figures/" directory.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Set common plotting style parameters
plt.rcParams.update({
    "font.size": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Create output folder for final figures
os.makedirs("figures", exist_ok=True)

# Helper function: try to load npy file and return its object (or None if fails)
def load_experiment_data(npy_path):
    try:
        data = np.load(npy_path, allow_pickle=True).item()
        return data
    except Exception as e:
        print(f"Error loading {npy_path}: {e}")
        return None

# -------------------------------
# MAIN PAPER FIGURES (Approximately 9 Figures)
# -------------------------------

# --- Plot 1: Baseline Loss Curves ---
try:
    # Baseline npy file path as provided in the JSON summary
    baseline_path = "experiment_results/experiment_931bb17225904a44802375f84e37e198_proc_3017221/experiment_data.npy"
    baseline_data = load_experiment_data(baseline_path)
    if baseline_data is not None:
        ds = "SPR_BENCH"
        per_layer = baseline_data["num_layers"][ds]["per_layer"]
        colors = {nl: c for nl, c in zip(sorted(per_layer), ["r", "g", "b", "m", "c"])}
        plt.figure(dpi=300)
        for nl, rec in per_layer.items():
            epochs = rec["epochs"]
            plt.plot(epochs, rec["losses"]["train"], linestyle="--", color=colors[nl], label=f"Train L{nl}")
            plt.plot(epochs, rec["losses"]["val"], linestyle="-", color=colors[nl], label=f"Val L{nl}")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy Loss")
        plt.title("Baseline: Training & Validation Loss vs Epoch")
        plt.legend()
        fn = os.path.join("figures", "FIG_Baseline_Loss_Curves.png")
        plt.savefig(fn)
        print(f"Saved {fn}")
        plt.close()
except Exception as e:
    print("Error in Baseline Loss Curves:", e)
    plt.close()

# --- Plot 2: Baseline Validation SCWA Curves ---
try:
    if baseline_data is not None:
        ds = "SPR_BENCH"
        plt.figure(dpi=300)
        for nl, rec in per_layer.items():
            epochs = rec["epochs"]
            plt.plot(epochs, rec["metrics"]["val"], color=colors[nl], label=f"L{nl}")
        plt.xlabel("Epoch")
        plt.ylabel("Validation SCWA")
        plt.title("Baseline: Validation SCWA vs Epoch")
        plt.legend()
        fn = os.path.join("figures", "FIG_Baseline_Val_SCWA.png")
        plt.savefig(fn)
        print(f"Saved {fn}")
        plt.close()
except Exception as e:
    print("Error in Baseline Validation SCWA Curves:", e)
    plt.close()

# --- Plot 3: Baseline Best SCWA Summary ---
try:
    if baseline_data is not None:
        ds = "SPR_BENCH"
        best_val_scores = [rec["best_val_scwa"] for rec in per_layer.values()]
        layers = list(per_layer.keys())
        best_layer = baseline_data["num_layers"][ds]["best_layer"]
        test_scwa = baseline_data["num_layers"][ds]["test_scwa"]
        x = np.arange(len(layers))
        plt.figure(dpi=300)
        plt.bar(x, best_val_scores, color=[colors[l] for l in layers], alpha=0.7, label="Best Val SCWA")
        # Overlay test score for best layer (highlighted)
        plt.bar(layers.index(best_layer), test_scwa, color="k", alpha=0.4, label="Test SCWA (best layer)")
        plt.xticks(x, [f"L{l}" for l in layers])
        plt.ylabel("SCWA")
        plt.title("Baseline: Best Val SCWA per Depth\n(Test SCWA highlighted)")
        plt.legend()
        fn = os.path.join("figures", "FIG_Baseline_Best_SCWA.png")
        plt.savefig(fn)
        print(f"Saved {fn}")
        plt.close()
except Exception as e:
    print("Error in Baseline Best SCWA Summary:", e)
    plt.close()

# --- Plot 4: Research - Contrastive Pre-training Loss ---
try:
    research_path = "experiment_results/experiment_7f065d85878c4bb997afc614645e7512_proc_3027376/experiment_data.npy"
    research_data = load_experiment_data(research_path)
    if research_data is not None and "SPR_BENCH" in research_data:
        rec = research_data["SPR_BENCH"]
        plt.figure(dpi=300)
        epochs = range(1, len(rec["losses"]["pretrain"]) + 1)
        plt.plot(epochs, rec["losses"]["pretrain"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Contrastive Loss")
        plt.title("Research: Contrastive Pre-training Loss")
        fn = os.path.join("figures", "FIG_Research_Pretrain_Loss.png")
        plt.savefig(fn)
        print(f"Saved {fn}")
        plt.close()
except Exception as e:
    print("Error in Research Pre-training Loss:", e)
    plt.close()

# --- Plot 5: Research - Fine-tuning Loss Curves ---
try:
    if research_data is not None and "SPR_BENCH" in research_data:
        rec = research_data["SPR_BENCH"]
        plt.figure(dpi=300)
        epochs = range(1, len(rec["losses"]["train"]) + 1)
        plt.plot(epochs, rec["losses"]["train"], linestyle="--", label="Train Loss")
        plt.plot(epochs, rec["losses"]["val"], linestyle="-", label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy Loss")
        plt.title("Research: Train vs Validation Loss")
        plt.legend()
        fn = os.path.join("figures", "FIG_Research_Train_Val_Loss.png")
        plt.savefig(fn)
        print(f"Saved {fn}")
        plt.close()
except Exception as e:
    print("Error in Research Fine-tuning Loss:", e)
    plt.close()

# --- Plot 6: Research - Validation Metrics (Accuracy & ACA) ---
try:
    if research_data is not None and "SPR_BENCH" in research_data:
        rec = research_data["SPR_BENCH"]
        plt.figure(dpi=300)
        epochs = range(1, len(rec["metrics"]["val_acc"]) + 1)
        plt.plot(epochs, rec["metrics"]["val_acc"], label="Val Accuracy")
        plt.plot(epochs, rec["metrics"]["val_aca"], label="Val ACA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Research: Validation Metrics")
        plt.legend()
        fn = os.path.join("figures", "FIG_Research_Val_Metrics.png")
        plt.savefig(fn)
        print(f"Saved {fn}")
        plt.close()
except Exception as e:
    print("Error in Research Validation Metrics:", e)
    plt.close()

# --- Plot 7: Research - Test Metrics Summary ---
try:
    if research_data is not None and "SPR_BENCH" in research_data:
        rec = research_data["SPR_BENCH"]
        test_metrics = rec["test"]
        names = ["acc", "swa", "cwa", "aca"]
        scores = [test_metrics[n] for n in names]
        plt.figure(dpi=300)
        plt.bar([n.upper() for n in names], scores, color="skyblue")
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title("Research: Test Metrics Summary\n(Acc, SWA, CWA, ACA)")
        fn = os.path.join("figures", "FIG_Research_Test_Metrics.png")
        plt.savefig(fn)
        print(f"Saved {fn}")
        plt.close()
except Exception as e:
    print("Error in Research Test Metrics Summary:", e)
    plt.close()

# --- Plot 8 (Main): Ablation - No-Pretraining Ablation ---
try:
    np_ablation = "experiment_results/experiment_6557f600264840f4818341e20c0af110_proc_3039482/experiment_data.npy"
    ablation_data = load_experiment_data(np_ablation)
    # Extract the no_pretraining branch for SPR_BENCH
    if ablation_data is not None:
        ed = ablation_data["no_pretraining"]["SPR_BENCH"]
        epochs = range(1, len(ed["losses"]["train"]) + 1)
        # Create a figure with 3 subplots
        fig, axs = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
        # Subplot 1: Loss Curves
        axs[0].plot(epochs, ed["losses"]["train"], label="Train Loss")
        axs[0].plot(epochs, ed["losses"]["val"], label="Validation Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Cross-entropy Loss")
        axs[0].set_title("Loss Curves")
        axs[0].legend()
        # Subplot 2: Accuracy Curves
        axs[1].plot(epochs, ed["metrics"]["train_acc"], label="Train Acc")
        axs[1].plot(epochs, ed["metrics"]["val_acc"], label="Val Acc")
        axs[1].plot(epochs, ed["metrics"]["val_aca"], label="Val ACA")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Accuracy / ACA")
        axs[1].set_title("Accuracy & ACA")
        axs[1].legend()
        # Subplot 3: Test Metrics Bar Chart
        test_metrics = ed["test"]
        names = ["Acc", "SWA", "CWA", "ACA"]
        values = [test_metrics.get("acc", 0), test_metrics.get("swa", 0),
                  test_metrics.get("cwa", 0), test_metrics.get("aca", 0)]
        axs[2].bar(names, values, color="steelblue")
        axs[2].set_ylim(0, 1)
        axs[2].set_title("Test Metrics")
        for i, v in enumerate(values):
            axs[2].text(i, v + 0.02, f"{v:.2f}", ha="center")
        plt.suptitle("Ablation: No-Pretraining Ablation", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fn = os.path.join("figures", "FIG_Ablation_NoPretraining.png")
        plt.savefig(fn)
        print(f"Saved {fn}")
        plt.close(fig)
except Exception as e:
    print("Error in Ablation No-Pretraining plot:", e)
    plt.close()

# --- Plot 9 (Main): Ablation - MLM Pre-training Ablation ---
try:
    mlm_path = "experiment_results/experiment_cfd2f21f107e4c62a05b9617ff257d86_proc_3039481/experiment_data.npy"
    mlm_data = load_experiment_data(mlm_path)
    if mlm_data is not None:
        run_dict = mlm_data.get("MLM_pretrain", {}).get("SPR_BENCH", {})
        # Create 3 subplots: loss curves, validation metrics, test metrics
        fig, axs = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
        # Subplot 1: Loss Curves (combine pretrain and fine-tune)
        pre = run_dict.get("losses", {}).get("pretrain", [])
        train = run_dict.get("losses", {}).get("train", [])
        val = run_dict.get("losses", {}).get("val", [])
        offset = len(pre)
        if pre:
            axs[0].plot(range(1, len(pre)+1), pre, label="Pretrain MLM", marker="o")
        if train:
            axs[0].plot(list(range(offset+1, offset+len(train)+1)), train, label="Fine-tune Train", linestyle="--")
        if val:
            axs[0].plot(list(range(offset+1, offset+len(val)+1)), val, label="Fine-tune Val")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("Loss Curves")
        axs[0].legend()
        # Subplot 2: Validation Metrics
        val_acc = run_dict.get("metrics", {}).get("val_acc", [])
        val_aca = run_dict.get("metrics", {}).get("val_aca", [])
        epochs_ft = range(1, len(val_acc)+1)
        axs[1].plot(epochs_ft, val_acc, label="Val Accuracy")
        axs[1].plot(epochs_ft, val_aca, label="Val ACA")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Score")
        axs[1].set_title("Validation Metrics")
        axs[1].legend()
        # Subplot 3: Test Metrics
        test_res = run_dict.get("test", {})
        keys = ["acc", "swa", "cwa", "aca"]
        vals = [test_res.get(k, 0) for k in keys]
        axs[2].bar([k.upper() for k in keys], vals, color="skyblue")
        axs[2].set_ylim(0, 1)
        axs[2].set_title("Test Metrics")
        for i, v in enumerate(vals):
            axs[2].text(i, v + 0.02, f"{v:.3f}", ha="center")
        plt.suptitle("Ablation: MLM Pre-training Ablation", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fn = os.path.join("figures", "FIG_Ablation_MLM_Pretrain.png")
        plt.savefig(fn)
        print(f"Saved {fn}")
        plt.close(fig)
except Exception as e:
    print("Error in Ablation MLM Pretrain plot:", e)
    plt.close()

# -------------------------------
# APPENDIX FIGURES (Additional, based on synthetic or extra ablation experiments)
# -------------------------------

# Appendix Plot A1: Frozen-Encoder Fine-Tuning Ablation
try:
    frozen_path = "experiment_results/experiment_83927c188e504dbe930b476c6498caf9_proc_3039481/experiment_data.npy"
    frozen_data = load_experiment_data(frozen_path)
    if frozen_data is not None and "SPR_BENCH" in frozen_data.get("full_tune", {}):
        # We'll pick values from the "full_tune" branch.
        fulltune = frozen_data["full_tune"]["SPR_BENCH"]
        epochs = range(1, len(fulltune.get("losses", {}).get("train", [])) + 1)
        fig, axs = plt.subplots(1, 3, figsize=(18,5), dpi=300)
        # Pre-training Loss from full_tune (if exists)
        pre_loss = fulltune.get("losses", {}).get("pretrain", [])
        if pre_loss:
            axs[0].plot(range(1, len(pre_loss)+1), pre_loss, marker="o")
            axs[0].set_title("Pre-training Loss")
        else:
            axs[0].text(0.5, 0.5, "No Pretrain Loss", ha="center")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        # Train vs Val Loss Curves
        axs[1].plot(epochs, fulltune.get("losses", {}).get("train", []), label="Train")
        axs[1].plot(epochs, fulltune.get("losses", {}).get("val", []), label="Val")
        axs[1].set_title("Train vs Val Loss")
        axs[1].set_xlabel("Epoch")
        axs[1].legend()
        # Validation Accuracy Curve
        axs[2].plot(epochs, fulltune.get("metrics", {}).get("val_acc", []), marker="s", color="green", label="Val Acc")
        axs[2].set_title("Validation Accuracy")
        axs[2].set_xlabel("Epoch")
        axs[2].legend()
        plt.suptitle("Ablation (Frozen-Encoder Fine-Tuning)", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fn = os.path.join("figures", "APP_FrozenEncoder_Ablation.png")
        plt.savefig(fn)
        print(f"Saved {fn}")
        plt.close(fig)
except Exception as e:
    print("Error in Frozen-Encoder Ablation Appendix plot:", e)
    plt.close()

# Appendix Plot A2: Masking-Only Augmentation Ablation
try:
    mask_only_path = "experiment_results/experiment_cf0efedcdafb4e39afd1fa8244b1c609_proc_3039482/experiment_data.npy"
    mask_only_data = load_experiment_data(mask_only_path)
    if mask_only_data is not None and "mask_only" in mask_only_data:
        rec = mask_only_data["mask_only"].get("SPR_BENCH", {})
        # Plot fine-tuning train/val loss and test metrics in two subplots
        fig, axs = plt.subplots(1, 2, figsize=(12,5), dpi=300)
        eps = range(1, len(rec.get("losses", {}).get("train", [])) + 1)
        axs[0].plot(eps, rec.get("losses", {}).get("train", []), label="Train Loss")
        axs[0].plot(eps, rec.get("losses", {}).get("val", []), label="Val Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("Fine-tuning Loss")
        axs[0].legend()
        test_vals = rec.get("test", {})
        keys = ["acc", "swa", "cwa", "aca"]
        vals = [test_vals.get(k, 0) for k in keys]
        axs[1].bar([k.upper() for k in keys], vals, color="mediumpurple")
        axs[1].set_ylim(0,1)
        axs[1].set_title("Test Metrics")
        for i, v in enumerate(vals):
            axs[1].text(i, v+0.02, f"{v:.2f}", ha="center")
        plt.suptitle("Ablation (Masking-Only Augmentation)", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fn = os.path.join("figures", "APP_MaskingOnly_Ablation.png")
        plt.savefig(fn)
        print(f"Saved {fn}")
        plt.close(fig)
except Exception as e:
    print("Error in Masking-Only Augmentation Ablation Appendix plot:", e)
    plt.close()

print("All final figures have been saved in the 'figures/' directory.")
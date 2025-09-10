#!/usr/bin/env python3
"""
Final Aggregator Script for Publishing Scientific Figures

This script aggregates experiment results stored in multiple .npy files from
baseline, research, and ablation experiments. It produces final, publication‐
ready figures saved in the "figures/" directory. Each plot is wrapped in its own
try‐except block, ensuring that failures in one do not prevent other figures
from being generated.

IMPORTANT:
• Data is loaded directly from the provided .npy files (using full and exact file paths).
• All figures are saved only under "figures/".
• Font sizes, labels, and styling have been increased for clarity in a final PDF.
• Only unique plots are included. Some additional synthetic or ablation plots may
  be put in the appendix.

Directories and file locations (as provided in JSON summaries):
  Baseline:
    experiment_results/experiment_4ab90ec1d66546d2a1d9db02a5f015ca_proc_2964457/experiment_data.npy
  Research:
    experiment_results/experiment_a1b690b6fe944a1a9b28e6a71b47c431_proc_2967787/experiment_data.npy
  Ablation experiments:
    No Positional Embeddings:
      experiment_results/experiment_678cfcf90bbb418d8e86dedea8f5833b_proc_2971858/experiment_data.npy
    No Projection Head (Encoder-Only Contrastive Learning):
      experiment_results/experiment_e1275fbf08aa435fb6b4779cf11f262c_proc_2971859/experiment_data.npy
    No-View-Augmentation Contrastive Pre-training:
      experiment_results/experiment_91297a1be36341f38f1522d48a9f45a1_proc_2971860/experiment_data.npy
    CLS-Token Representation Instead of Mean-Pooling:
      experiment_results/experiment_674fac0b2890482c80e9496ae173d132_proc_2971857/experiment_data.npy
    Frozen Encoder (Linear-Probe Fine-Tuning):
      experiment_results/experiment_4513fc1921e44032a9fc68742c8f0796_proc_2971858/experiment_data.npy
    Attention-Only Encoder (No Feed-Forward Layers):
      experiment_results/experiment_c6283e1f3efe40898b8784119464649e_proc_2971860/experiment_data.npy
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Set up global style for publication-quality figures
plt.rcParams.update({
    "font.size": 14,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False
})

# Create the final figures directory
os.makedirs("figures", exist_ok=True)

##############################################
# Baseline Figures (3 Figures)
##############################################
def plot_baseline():
    # Load baseline experiment data
    baseline_path = "experiment_results/experiment_4ab90ec1d66546d2a1d9db02a5f015ca_proc_2964457/experiment_data.npy"
    try:
        baseline_data = np.load(baseline_path, allow_pickle=True).item()
        data = baseline_data["emb_dim_tuning"]["SPR_BENCH"]
    except Exception as e:
        print(f"Baseline data load error: {e}")
        return

    # Extract arrays from baseline data
    try:
        train_f1 = np.array(data["metrics"]["train_macroF1"])
        val_f1 = np.array(data["metrics"]["val_macroF1"])
        train_ls = np.array(data["losses"]["train"])
        val_ls = np.array(data["losses"]["val"])
        emb_dims = np.array(data["hyperparams"])
        num_epochs = len(train_f1) // len(emb_dims)
        epoch_idx = np.arange(1, len(train_f1)+1)
    except Exception as e:
        print(f"Baseline data extraction error: {e}")
        return

    # Plot 1: Macro-F1 curves (Train and Val)
    try:
        plt.figure(figsize=(8,6))
        plt.plot(epoch_idx, train_f1, label="Train Macro F1", marker="o")
        plt.plot(epoch_idx, val_f1, label="Validation Macro F1", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1 Score")
        plt.title("SPR_BENCH Macro F1 Over Epochs")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join("figures", "baseline_macro_F1.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved baseline plot: {fname}")
    except Exception as e:
        print(f"Error in baseline F1 curve plot: {e}")
        plt.close()

    # Plot 2: Loss curves (Train and Val)
    try:
        plt.figure(figsize=(8,6))
        plt.plot(epoch_idx, train_ls, label="Train Loss", marker="o")
        plt.plot(epoch_idx, val_ls, label="Validation Loss", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Over Epochs")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join("figures", "baseline_loss_curves.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved baseline plot: {fname}")
    except Exception as e:
        print(f"Error in baseline loss curve plot: {e}")
        plt.close()

    # Plot 3: Final Val Macro F1 vs Embedding Dimension (Bar Plot)
    try:
        finals = val_f1.reshape(len(emb_dims), num_epochs)[:, -1]
        plt.figure(figsize=(8,6))
        plt.bar([str(e) for e in emb_dims], finals, color="skyblue")
        plt.xlabel("Embedding Dimension")
        plt.ylabel("Final Validation Macro F1")
        plt.title("SPR_BENCH Final Validation Macro F1 by Embedding Size")
        plt.tight_layout()
        fname = os.path.join("figures", "baseline_valF1_vs_embedding.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved baseline plot: {fname}")
    except Exception as e:
        print(f"Error in baseline emb-dim bar plot: {e}")
        plt.close()


##############################################
# Research Figures (4 Figures)
##############################################
def plot_research():
    # Load research experiment data
    research_path = "experiment_results/experiment_a1b690b6fe944a1a9b28e6a71b47c431_proc_2967787/experiment_data.npy"
    try:
        research_data = np.load(research_path, allow_pickle=True).item()
        data = research_data["SPR_transformer"]
    except Exception as e:
        print(f"Research data load error: {e}")
        return

    try:
        pre_losses = np.array(data["losses"].get("pretrain", []))
        tr_losses = np.array(data["losses"].get("train", []))
        val_losses = np.array(data["losses"].get("val", []))
        swa = np.array(data["metrics"].get("val_SWA", []))
        cwa = np.array(data["metrics"].get("val_CWA", []))
        scwa_vals = np.array(data["metrics"].get("val_SCWA", []))
        preds = np.array(data.get("predictions", []))
        gts = np.array(data.get("ground_truth", []))
    except Exception as e:
        print(f"Research data extraction error: {e}")
        return

    # Plot 1: Pre-training Loss
    try:
        if pre_losses.size:
            plt.figure(figsize=(8,6))
            epochs = np.arange(1, len(pre_losses)+1)
            plt.plot(epochs, pre_losses, marker="o", color="purple")
            plt.xlabel("Epoch")
            plt.ylabel("NT-Xent Loss")
            plt.title("SPR_transformer Pre-training Loss")
            plt.tight_layout()
            fname = os.path.join("figures", "research_pretrain_loss.png")
            plt.savefig(fname)
            plt.close()
            print(f"Saved research plot: {fname}")
    except Exception as e:
        print(f"Error in research pre-training loss plot: {e}")
        plt.close()

    # Plot 2: Fine-tune Loss (Train vs Val)
    try:
        if tr_losses.size and val_losses.size:
            plt.figure(figsize=(8,6))
            epochs = np.arange(1, len(tr_losses)+1)
            plt.plot(epochs, tr_losses, marker="o", label="Train Loss")
            plt.plot(epochs, val_losses, marker="s", label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title("SPR_transformer Fine-tune Losses")
            plt.legend()
            plt.tight_layout()
            fname = os.path.join("figures", "research_finetune_losses.png")
            plt.savefig(fname)
            plt.close()
            print(f"Saved research plot: {fname}")
    except Exception as e:
        print(f"Error in research fine-tuning loss plot: {e}")
        plt.close()

    # Plot 3: Validation Metrics (SWA, CWA, SCWA)
    try:
        if scwa_vals.size:
            plt.figure(figsize=(8,6))
            epochs = np.arange(1, len(scwa_vals)+1)
            plt.plot(epochs, swa, marker="o", label="SWA")
            plt.plot(epochs, cwa, marker="s", label="CWA")
            plt.plot(epochs, scwa_vals, marker="^", label="SCWA")
            plt.xlabel("Epoch")
            plt.ylabel("Metric Value")
            plt.title("SPR_transformer Validation Metrics")
            plt.legend()
            plt.tight_layout()
            fname = os.path.join("figures", "research_val_metrics.png")
            plt.savefig(fname)
            plt.close()
            print(f"Saved research plot: {fname}")
    except Exception as e:
        print(f"Error in research validation metrics plot: {e}")
        plt.close()
    
    # Plot 4: Confusion Matrix
    try:
        if preds.size and gts.size:
            num_lbl = int(max(preds.max(), gts.max())) + 1
            cm = np.zeros((num_lbl, num_lbl), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure(figsize=(6,5))
            im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title("SPR_transformer Confusion Matrix")
            plt.tight_layout()
            fname = os.path.join("figures", "research_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
            print(f"Saved research plot: {fname}")
    except Exception as e:
        print(f"Error in research confusion matrix plot: {e}")
        plt.close()


##############################################
# Ablation Figures (6 Figures)
##############################################
def plot_ablation():
    # 1. No Positional Embeddings (use key "no_positional" under SPR_BENCH)
    np_path = "experiment_results/experiment_678cfcf90bbb418d8e86dedea8f5833b_proc_2971858/experiment_data.npy"
    try:
        np_data_all = np.load(np_path, allow_pickle=True).item()
        np_data = np_data_all["no_positional"]["SPR_BENCH"]
    except Exception as e:
        print(f"No Positional Embeddings load error: {e}")
        np_data = None

    if np_data:
        try:
            # Create combined figure with Loss curves and Metric curves (2 subplots)
            fig, axs = plt.subplots(1, 2, figsize=(14,6))
            epochs_pre = np.arange(1, len(np_data["losses"]["pretrain"]) + 1)
            epochs_ft = np.arange(1, len(np_data["losses"]["train"]) + 1)
            # Loss curves subplot
            axs[0].plot(epochs_pre, np_data["losses"]["pretrain"], label="Pretrain Loss", marker="o")
            axs[0].plot(epochs_ft, np_data["losses"]["train"], label="Train Loss", marker="s")
            axs[0].plot(epochs_ft, np_data["losses"]["val"], label="Val Loss", marker="^")
            axs[0].set_xlabel("Epoch")
            axs[0].set_ylabel("Loss")
            axs[0].set_title("No Positional Embeddings: Loss Curves")
            axs[0].legend()
            # Metric curves subplot
            epochs = np.arange(1, len(np_data["metrics"]["val_SWA"]) + 1)
            axs[1].plot(epochs, np_data["metrics"]["val_SWA"], label="SWA", marker="o")
            axs[1].plot(epochs, np_data["metrics"]["val_CWA"], label="CWA", marker="s")
            axs[1].plot(epochs, np_data["metrics"]["val_SCWA"], label="SCWA", marker="^")
            axs[1].set_xlabel("Epoch")
            axs[1].set_ylabel("Metric Value")
            axs[1].set_title("No Positional Embeddings: Metrics")
            axs[1].legend()
            plt.tight_layout()
            fname = os.path.join("figures", "ablation_no_positional.png")
            plt.savefig(fname)
            plt.close()
            print(f"Saved ablation plot: {fname}")
        except Exception as e:
            print(f"Error plotting No Positional Embeddings: {e}")
            plt.close()

    # 2. No Projection Head (Encoder-Only Contrastive Learning)
    np_head_path = "experiment_results/experiment_e1275fbf08aa435fb6b4779cf11f262c_proc_2971859/experiment_data.npy"
    try:
        np_head_data_all = np.load(np_head_path, allow_pickle=True).item()
        np_head = np_head_data_all["no_projection_head"]["SPR"]
    except Exception as e:
        print(f"No Projection Head load error: {e}")
        np_head = None

    if np_head:
        try:
            # We produce a single figure: Confusion Matrix from no projection head experiment.
            preds = np.array(np_head["predictions"])
            trues = np.array(np_head["ground_truth"])
            num_lbl = int(max(preds.max(), trues.max())) + 1
            cm = np.zeros((num_lbl, num_lbl), dtype=int)
            for t, p in zip(trues, preds):
                cm[t, p] += 1
            plt.figure(figsize=(6,5))
            im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title("No Projection Head: Confusion Matrix")
            plt.tight_layout()
            fname = os.path.join("figures", "ablation_no_projection_head_confusion.png")
            plt.savefig(fname)
            plt.close()
            print(f"Saved ablation plot: {fname}")
        except Exception as e:
            print(f"Error plotting No Projection Head confusion matrix: {e}")
            plt.close()

    # 3. No-View-Augmentation Contrastive Pre-training
    nva_path = "experiment_results/experiment_91297a1be36341f38f1522d48a9f45a1_proc_2971860/experiment_data.npy"
    try:
        nva_data_all = np.load(nva_path, allow_pickle=True).item()
        nva_data = nva_data_all["no_view_aug"]["SPR_transformer"]
    except Exception as e:
        print(f"No-View-Augmentation load error: {e}")
        nva_data = None

    if nva_data:
        try:
            # Create a combined figure with two subplots: Fine-tuning loss and Validation Metrics.
            fig, axs = plt.subplots(1, 2, figsize=(14,6))
            # Fine-tuning Loss plot
            epochs = np.arange(1, len(nva_data["losses"]["train"]) + 1)
            axs[0].plot(epochs, nva_data["losses"]["train"], label="Train CE Loss", marker="o")
            axs[0].plot(epochs, nva_data["losses"]["val"], label="Val CE Loss", marker="s")
            axs[0].set_xlabel("Epoch")
            axs[0].set_ylabel("Cross-Entropy Loss")
            axs[0].set_title("No-View-Aug Pre-training: Fine-tune Loss")
            axs[0].legend()
            # Validation Metrics plot
            epochs2 = np.arange(1, len(nva_data["metrics"]["val_SCWA"]) + 1)
            axs[1].plot(epochs2, nva_data["metrics"]["val_SWA"], label="SWA", marker="o")
            axs[1].plot(epochs2, nva_data["metrics"]["val_CWA"], label="CWA", marker="s")
            axs[1].plot(epochs2, nva_data["metrics"]["val_SCWA"], label="SCWA", marker="^")
            axs[1].set_xlabel("Epoch")
            axs[1].set_ylabel("Metric Value")
            axs[1].set_title("No-View-Aug Pre-training: Metrics")
            axs[1].legend()
            plt.tight_layout()
            fname = os.path.join("figures", "ablation_no_view_aug.png")
            plt.savefig(fname)
            plt.close()
            print(f"Saved ablation plot: {fname}")
        except Exception as e:
            print(f"Error plotting No-View-Augmentation: {e}")
            plt.close()

    # 4. CLS-Token Representation Instead of Mean-Pooling
    cls_path = "experiment_results/experiment_674fac0b2890482c80e9496ae173d132_proc_2971857/experiment_data.npy"
    try:
        cls_data_all = np.load(cls_path, allow_pickle=True).item()
        cls_data = cls_data_all  # Both strategies are stored as keys in the file.
    except Exception as e:
        print(f"CLS-Token data load error: {e}")
        cls_data = {}

    if cls_data:
        try:
            # We assume two keys: 'mean_pool' and 'cls_token' exist.
            strategies = ["mean_pool", "cls_token"]
            colors = {"mean_pool": "tab:blue", "cls_token": "tab:orange"}
            # Pre-training Loss Comparison (single plot with both curves)
            plt.figure(figsize=(8,6))
            for strat in strategies:
                losses = cls_data.get(strat, {}).get("SPR_BENCH", {}).get("losses", {}).get("pretrain", [])
                if losses:
                    plt.plot(range(1, len(losses)+1), losses, marker="o", label=strat, color=colors.get(strat))
            plt.xlabel("Epoch")
            plt.ylabel("Pre-training Loss")
            plt.title("CLS-Token vs Mean-Pooling: Pre-training Loss")
            plt.legend()
            plt.tight_layout()
            fname = os.path.join("figures", "ablation_cls_token_pretrain.png")
            plt.savefig(fname)
            plt.close()
            print(f"Saved ablation plot: {fname}")
        except Exception as e:
            print(f"Error plotting CLS-Token pretrain loss comparison: {e}")
            plt.close()

    # 5. Frozen Encoder (Linear-Probe Fine-Tuning)
    frozen_path = "experiment_results/experiment_4513fc1921e44032a9fc68742c8f0796_proc_2971858/experiment_data.npy"
    try:
        frozen_data_all = np.load(frozen_path, allow_pickle=True).item()
        frozen_data = frozen_data_all["frozen_encoder"]["SPR_BENCH"]
    except Exception as e:
        print(f"Frozen Encoder load error: {e}")
        frozen_data = None

    if frozen_data:
        try:
            plt.figure(figsize=(8,6))
            # Plot validation metrics curves
            epochs = np.arange(1, len(frozen_data["metrics"]["val_SWA"]) + 1)
            plt.plot(epochs, frozen_data["metrics"]["val_SWA"], label="SWA", marker="o")
            plt.plot(epochs, frozen_data["metrics"]["val_CWA"], label="CWA", marker="s")
            plt.plot(epochs, frozen_data["metrics"]["val_SCWA"], label="SCWA", marker="^")
            plt.xlabel("Epoch")
            plt.ylabel("Metric Value")
            plt.title("Frozen Encoder: Validation Metrics")
            plt.legend()
            plt.tight_layout()
            fname = os.path.join("figures", "ablation_frozen_encoder_metrics.png")
            plt.savefig(fname)
            plt.close()
            print(f"Saved ablation plot: {fname}")
        except Exception as e:
            print(f"Error plotting Frozen Encoder metrics: {e}")
            plt.close()

    # 6. Attention-Only Encoder (No Feed-Forward Layers)
    attention_path = "experiment_results/experiment_c6283e1f3efe40898b8784119464649e_proc_2971860/experiment_data.npy"
    try:
        attn_data_all = np.load(attention_path, allow_pickle=True).item()
        attn_data = attn_data_all["attention_only"]["SPR_BENCH"]
    except Exception as e:
        print(f"Attention-Only data load error: {e}")
        attn_data = None

    if attn_data:
        try:
            # Create a 2x2 grid:
            fig, axs = plt.subplots(2, 2, figsize=(12,10))
            # Subplot 1: Pre-training Loss
            axs[0,0].plot(range(1, len(attn_data.get("losses", {}).get("pretrain", [])) + 1), 
                           attn_data.get("losses", {}).get("pretrain", []), marker="o", color="purple")
            axs[0,0].set_title("Attention-Only: Pre-training Loss")
            axs[0,0].set_xlabel("Epoch")
            axs[0,0].set_ylabel("Loss")
            # Subplot 2: Training vs Validation Loss
            epochs_attn = range(1, len(attn_data.get("losses", {}).get("train", [])) + 1)
            axs[0,1].plot(epochs_attn, attn_data.get("losses", {}).get("train", []), label="Train", marker="o")
            axs[0,1].plot(epochs_attn, attn_data.get("losses", {}).get("val", []), label="Validation", marker="s")
            axs[0,1].set_title("Attention-Only: Train vs Val Loss")
            axs[0,1].set_xlabel("Epoch")
            axs[0,1].legend()
            # Subplot 3: Validation Metrics Curves
            epochs_met = range(1, len(attn_data.get("metrics", {}).get("val_SWA", [])) + 1)
            axs[1,0].plot(epochs_met, attn_data.get("metrics", {}).get("val_SWA", []), label="SWA", marker="o")
            axs[1,0].plot(epochs_met, attn_data.get("metrics", {}).get("val_CWA", []), label="CWA", marker="s")
            axs[1,0].plot(epochs_met, attn_data.get("metrics", {}).get("val_SCWA", []), label="SCWA", marker="^")
            axs[1,0].set_title("Attention-Only: Validation Metrics")
            axs[1,0].set_xlabel("Epoch")
            axs[1,0].legend()
            # Subplot 4: Best Epoch Bar Summary (using best of val_SCWA)
            if attn_data.get("metrics", {}).get("val_SCWA", []):
                best_idx = int(np.argmax(attn_data["metrics"]["val_SCWA"]))
                best_vals = [ attn_data["metrics"]["val_SWA"][best_idx],
                              attn_data["metrics"]["val_CWA"][best_idx],
                              attn_data["metrics"]["val_SCWA"][best_idx] ]
                axs[1,1].bar(["SWA", "CWA", "SCWA"], best_vals, color=["skyblue","salmon","gold"])
                axs[1,1].set_title(f"Attention-Only: Best Epoch ({best_idx+1}) Metrics")
                axs[1,1].set_ylabel("Score")
            plt.tight_layout()
            fname = os.path.join("figures", "ablation_attention_only.png")
            plt.savefig(fname)
            plt.close()
            print(f"Saved ablation plot: {fname}")
        except Exception as e:
            print(f"Error plotting Attention-Only Encoder: {e}")
            plt.close()


##############################################
# Main function: generate all plots
##############################################
def main():
    print("Generating Baseline Figures...")
    plot_baseline()
    print("Generating Research Figures...")
    plot_research()
    print("Generating Ablation Figures...")
    plot_ablation()
    print("All figures have been generated in the 'figures/' directory.")

if __name__ == "__main__":
    main()
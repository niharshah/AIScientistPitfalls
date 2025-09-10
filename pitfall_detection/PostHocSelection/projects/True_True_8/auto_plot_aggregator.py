#!/usr/bin/env python
"""
Final Aggregator for Context-Aware Contrastive Learning Figures
This script aggregates final experimental plots from baseline, research,
and ablation experiments. It loads pre‐computed numpy files containing
experiment data and produces publication–ready figures into the ‘figures/’ folder.
Each figure is built in a try–except block so failures in one plot don’t affect others.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Set global matplotlib parameters for professional appearance
plt.rcParams.update({
    "font.size": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300
})

# Create final figures directory
os.makedirs("figures", exist_ok=True)

# ---------------------- Helper Functions ---------------------------

def unpack(nested, keys):
    """
    Given a nested dict and a tuple 'keys', returns two numpy arrays: epochs and values.
    Each target is expected to be a list of (epoch, value) tuples.
    """
    try:
        data = nested
        for k in keys:
            data = data[k]
        epochs, vals = zip(*data)
        return np.array(epochs), np.array(vals)
    except Exception as e:
        print("Error in unpack with keys", keys, ":", e)
        return np.array([]), np.array([])

def set_labels(ax, xlabel, ylabel, title):
    """Helper to label an axis with clean labels."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

# ---------------------- Load Experiment Data -----------------------------
# Baseline experiment data
try:
    baseline_path = "experiment_results/experiment_accc7227dc7845e69f15b6b88f30bd64_proc_3071487/experiment_data.npy"
    baseline_data = np.load(baseline_path, allow_pickle=True).item()
except Exception as e:
    print("Error loading baseline data:", e)
    baseline_data = {}

# Research experiment data
try:
    research_path = "experiment_results/experiment_a340587537454d749b29abbb3c8920cd_proc_3085139/experiment_data.npy"
    research_data = np.load(research_path, allow_pickle=True).item()
except Exception as e:
    print("Error loading research data:", e)
    research_data = {}

# Ablation experiments:
# No Projector Head in Contrastive Pretraining
try:
    np_projector_path = "experiment_results/experiment_b24acd7987e04558b57ea512a4913293_proc_3087391/experiment_data.npy"
    ablation_no_projector = np.load(np_projector_path, allow_pickle=True).item()
    # Expected structure: ablation_no_projector["no_projector_head"]["spr"]
    ab_no_proj = ablation_no_projector.get("no_projector_head", {}).get("spr", {})
except Exception as e:
    print("Error loading No Projector Head data:", e)
    ab_no_proj = {}

# Freeze Encoder During Fine-Tuning
try:
    freeze_enc_path = "experiment_results/experiment_74df53e3d5384a5cb0bb373ee94ac86e_proc_3087393/experiment_data.npy"
    ablation_freeze_enc = np.load(freeze_enc_path, allow_pickle=True).item()
    # Expected structure: ablation_freeze_enc["freeze_encoder"]["SPR"]
    ab_freeze = ablation_freeze_enc.get("freeze_encoder", {}).get("SPR", {})
except Exception as e:
    print("Error loading Freeze Encoder data:", e)
    ab_freeze = {}

# Uni-Directional Encoder (Remove Bidirectional GRU)
try:
    uni_enc_path = "experiment_results/experiment_0b062d29fe7a4c7bb12c635e40b4799e_proc_3087394/experiment_data.npy"
    ablation_uni_enc = np.load(uni_enc_path, allow_pickle=True).item()
    # Expected structure: ablation_uni_enc["uni_directional_encoder"]["SPR"]
    ab_uni = ablation_uni_enc.get("uni_directional_encoder", {}).get("SPR", {})
except Exception as e:
    print("Error loading Uni-Directional Encoder data:", e)
    ab_uni = {}

# No-Contrastive-Pretraining (Scratch Training)
try:
    no_con_path = "experiment_results/experiment_28a4de22adb54e0ca2b9e7bcb9f7b7b0_proc_3087394/experiment_data.npy"
    ablation_no_con = np.load(no_con_path, allow_pickle=True).item()
    # Expected structure: ablation_no_con["no_contrastive_pretraining"]["spr"]
    ab_no_con = ablation_no_con.get("no_contrastive_pretraining", {}).get("spr", {})
except Exception as e:
    print("Error loading No-Contrastive-Pretraining data:", e)
    ab_no_con = {}

# Bag-of-Tokens Input (Token Order Shuffle)
try:
    bag_tokens_path = "experiment_results/experiment_ad8d84e49adb4c969130e19fdcc3ac53_proc_3087393/experiment_data.npy"
    ablation_bag_tokens = np.load(bag_tokens_path, allow_pickle=True).item()
    # Expected structure: iterate over ablation_bag_tokens; assume key "SPR_BENCH" exists.
    ab_bag = ablation_bag_tokens.get("Bag-of-Tokens Input (Token Order Shuffle)", {})
    # If multiple datasets, pick one — assume "SPR_BENCH" key.
    ab_bag_spr = ab_bag.get("SPR_BENCH", {})
except Exception as e:
    print("Error loading Bag-of-Tokens Input data:", e)
    ab_bag_spr = {}

# ---------------------- Baseline Figures -----------------------------
# 1. Baseline Loss Curves per Embedding (aggregated 3 subplots in one row)
try:
    tuning = baseline_data.get("embed_dim_tuning", {})
    embed_keys = sorted(tuning.keys())  # e.g. ['embed_64', 'embed_128', 'embed_256']
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for i, k in enumerate(embed_keys):
        run = tuning.get(k, {})
        ep_tr, tr_loss = unpack(run, ("losses", "train"))
        ep_va, va_loss = unpack(run, ("losses", "val"))
        ax = axs[i]
        ax.plot(ep_tr, tr_loss, label="Train", marker="o")
        ax.plot(ep_va, va_loss, label="Validation", marker="s")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        emb_dim = k.split("_")[1]
        ax.set_title(f"Embedding dim = {emb_dim}")
        ax.legend()
        ax.grid(True)
    fig.suptitle("Baseline: Loss Curves by Embedding Dimension\n(Data: synthetic SPR)", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join("figures", "baseline_loss_curves.png"))
    plt.close(fig)
except Exception as e:
    print("Error plotting baseline loss curves:", e)
    plt.close()

# 2. Baseline: CoWA vs Epoch across dimensions
try:
    fig, ax = plt.subplots(figsize=(8, 6))
    for k in embed_keys:
        ep, cowa = unpack(tuning.get(k, {}), ("metrics", "CoWA"))
        if ep.size:
            emb_dim = k.split("_")[1]
            ax.plot(ep, cowa, marker="o", label=f"Embedding {emb_dim}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CoWA")
    ax.set_title("Baseline: CoWA over Epochs\n(synthetic SPR)")
    ax.legend(title="Embedding Dim")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "baseline_cowa_epochs.png"))
    plt.close()
except Exception as e:
    print("Error plotting baseline CoWA vs Epoch:", e)
    plt.close()

# 3. Baseline: Final CoWA Bar Chart
try:
    dims, finals = [], []
    for k in embed_keys:
        dims.append(k.split("_")[1])
        final_val = unpack(tuning.get(k, {}), ("metrics", "CoWA"))[1]
        if final_val.size:
            finals.append(final_val[-1])
        else:
            finals.append(0)
    x = np.arange(len(dims))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(x, finals, color="skyblue")
    ax.set_xticks(x)
    ax.set_xticklabels(dims)
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Final CoWA")
    ax.set_title("Baseline: Final CoWA by Embedding Size")
    ax.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "baseline_final_cowa.png"))
    plt.close()
except Exception as e:
    print("Error plotting baseline final CoWA bar chart:", e)
    plt.close()

# ---------------------- Research Figures -----------------------------
# 4. Research: Contrastive Pretraining Loss Plot
try:
    ep, loss = unpack(research_data, ("contrastive_pretrain", "losses"))
    if ep.size:
        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(ep, loss, marker="o", color="purple", label="NT-Xent Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("NT-Xent Loss")
        ax.set_title("Research: Contrastive Pretrain Loss\n(SPR_dataset)")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "research_contrastive_loss.png"))
        plt.close()
except Exception as e:
    print("Error plotting research contrastive loss:", e)
    plt.close()

# 5. Research: Fine-tune Loss Curves Plot
try:
    ep_tr, tr_loss = unpack(research_data, ("fine_tune", "losses", "train"))
    ep_va, va_loss = unpack(research_data, ("fine_tune", "losses", "val"))
    if ep_tr.size and ep_va.size:
        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(ep_tr, tr_loss, marker="o", label="Train")
        ax.plot(ep_va, va_loss, marker="s", label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.set_title("Research: Fine-tune Loss Curves\n(SPR_dataset)")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "research_finetune_loss.png"))
        plt.close()
except Exception as e:
    print("Error plotting research fine-tune loss:", e)
    plt.close()

# 6. Research: Fine-tune Metrics Curves (SWA, CWA, CompWA)
try:
    fig, ax = plt.subplots(figsize=(8,6))
    for key, lab in zip(["SWA", "CWA", "CompWA"],
                        ["Shape-Weighted Acc", "Color-Weighted Acc", "Complexity-Weighted Acc"]):
        ep, vals = unpack(research_data, ("fine_tune", "metrics", key))
        if ep.size:
            ax.plot(ep, vals, marker="o", label=lab)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Research: Weighted Accuracy Metrics\n(SPR_dataset)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "research_metrics_curves.png"))
    plt.close()
except Exception as e:
    print("Error plotting research metrics curves:", e)
    plt.close()

# 7. Research: Final Metrics Text Summary (as a textual figure)
try:
    # Extract final fine-tune values
    final_va = unpack(research_data, ("fine_tune", "losses", "val"))[1]
    final_SWA = unpack(research_data, ("fine_tune", "metrics", "SWA"))[1]
    final_CWA = unpack(research_data, ("fine_tune", "metrics", "CWA"))[1]
    final_CompWA = unpack(research_data, ("fine_tune", "metrics", "CompWA"))[1]
    textstr = (f"Final Validation Loss: {final_va[-1]:.4f}\n"
               f"Final SWA: {final_SWA[-1]:.4f}\n"
               f"Final CWA: {final_CWA[-1]:.4f}\n"
               f"Final CompWA: {final_CompWA[-1]:.4f}")
    fig, ax = plt.subplots(figsize=(6,4))
    ax.axis("off")
    ax.text(0.5, 0.5, textstr, transform=ax.transAxes,
            fontsize=16, verticalalignment='center',
            horizontalalignment='center', bbox=dict(facecolor="lightgrey", alpha=0.5))
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "research_final_metrics_text.png"))
    plt.close()
except Exception as e:
    print("Error creating research final metrics text summary:", e)
    plt.close()

# ---------------------- Ablation Figures -----------------------------
# 7. Ablation: No Projector Head (aggregated 3 subplots)
try:
    fig, axs = plt.subplots(1, 3, figsize=(18,5))
    # Subplot 1: Contrastive Pretrain Loss
    ep_np, loss_np = unpack(ab_no_proj, ("contrastive_pretrain", "losses"))
    axs[0].plot(ep_np, loss_np, marker="o", color="darkred")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("NT-Xent Loss")
    axs[0].set_title("No Projector Head: Pretrain Loss")
    axs[0].grid(True)
    # Subplot 2: Fine-tune Loss Curves
    ep_tr, tr_loss = unpack(ab_no_proj, ("fine_tune", "losses", "train"))
    ep_va, va_loss = unpack(ab_no_proj, ("fine_tune", "losses", "val"))
    axs[1].plot(ep_tr, tr_loss, marker="o", label="Train")
    axs[1].plot(ep_va, va_loss, marker="s", label="Validation")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Cross-Entropy Loss")
    axs[1].set_title("No Projector Head: Fine-tune Loss")
    axs[1].legend()
    axs[1].grid(True)
    # Subplot 3: Metrics Curves
    for key, lab in zip(["SWA", "CWA", "CompWA"],
                        ["Shape-Weighted Acc", "Color-Weighted Acc", "Complexity-Weighted Acc"]):
        ep, vals = unpack(ab_no_proj, ("fine_tune", "metrics", key))
        if ep.size:
            axs[2].plot(ep, vals, marker="o", label=lab)
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Accuracy")
    axs[2].set_title("No Projector Head: Metrics")
    axs[2].legend()
    axs[2].grid(True)
    fig.suptitle("Ablation: No Projector Head in Contrastive Pretraining")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join("figures", "ablation_no_projector_head.png"))
    plt.close()
except Exception as e:
    print("Error plotting ablation No Projector Head:", e)
    plt.close()

# 8. Ablation: Freeze Encoder During Fine-Tuning (aggregated 3 subplots)
try:
    fig, axs = plt.subplots(1, 3, figsize=(18,5))
    # Subplot 1: Contrastive Pretrain Loss
    ep_fr, loss_fr = unpack(ab_freeze, ("contrastive_pretrain", "losses"))
    axs[0].plot(ep_fr, loss_fr, marker="o", color="darkgreen")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Freeze Encoder: Pretrain Loss")
    axs[0].grid(True)
    # Subplot 2: Fine-tune Loss (Train/Val)
    ep_tr, tr_loss = unpack(ab_freeze, ("fine_tune", "losses", "train"))
    ep_va, va_loss = unpack(ab_freeze, ("fine_tune", "losses", "val"))
    axs[1].plot(ep_tr, tr_loss, marker="o", label="Train")
    axs[1].plot(ep_va, va_loss, marker="s", label="Validation")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].set_title("Freeze Encoder: Fine-tune Loss")
    axs[1].legend()
    axs[1].grid(True)
    # Subplot 3: Metrics Curves
    for key, lab in zip(["SWA", "CWA", "CompWA"],
                        ["Shape-Weighted Acc", "Color-Weighted Acc", "Complexity-Weighted Acc"]):
        ep, vals = unpack(ab_freeze, ("fine_tune", "metrics", key))
        if ep.size:
            axs[2].plot(ep, vals, marker="o", label=lab)
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Accuracy")
    axs[2].set_title("Freeze Encoder: Metrics")
    axs[2].legend()
    axs[2].grid(True)
    fig.suptitle("Ablation: Freeze Encoder During Fine-Tuning")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join("figures", "ablation_freeze_encoder.png"))
    plt.close()
except Exception as e:
    print("Error plotting ablation Freeze Encoder:", e)
    plt.close()

# 9. Ablation: Uni-Directional Encoder (aggregated 3 subplots)
try:
    fig, axs = plt.subplots(1, 3, figsize=(18,5))
    # Subplot 1: Contrastive Loss (from uni-directional encoder arrays)
    if "contrastive_losses" in ab_uni and len(ab_uni["contrastive_losses"]) > 0:
        contr_arr = np.array(ab_uni["contrastive_losses"])  # expected shape (N,2)
        axs[0].plot(contr_arr[:,0], contr_arr[:,1], marker="o", color="brown")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("NT-Xent Loss")
    axs[0].set_title("Uni-Directional: Pretrain Loss")
    axs[0].grid(True)
    # Subplot 2: Fine-tune Loss (Train & Validation)
    if "losses" in ab_uni:
        tr_arr = np.array(ab_uni["losses"].get("train", []))
        va_arr = np.array(ab_uni["losses"].get("val", []))
        if tr_arr.size:
            axs[1].plot(tr_arr[:,0], tr_arr[:,1], label="Train", marker="o")
        if va_arr.size:
            axs[1].plot(va_arr[:,0], va_arr[:,1], label="Validation", marker="s")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Cross-Entropy Loss")
    axs[1].set_title("Uni-Directional: Fine-tune Loss")
    axs[1].legend()
    axs[1].grid(True)
    # Subplot 3: Metrics Curves
    for key, lab in zip(["SWA", "CWA", "CompWA"],
                        ["Shape-Weighted Acc", "Color-Weighted Acc", "Complexity-Weighted Acc"]):
        if key in ab_uni.get("metrics", {}):
            arr = np.array(ab_uni["metrics"][key])
            if arr.size:
                axs[2].plot(arr[:,0], arr[:,1], marker="o", label=lab)
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Accuracy")
    axs[2].set_title("Uni-Directional: Metrics")
    axs[2].legend()
    axs[2].grid(True)
    fig.suptitle("Ablation: Uni-Directional Encoder (Remove Bi-GRU)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join("figures", "ablation_uni_directional_encoder.png"))
    plt.close()
except Exception as e:
    print("Error plotting ablation Uni-Directional Encoder:", e)
    plt.close()

# 10. Ablation: No-Contrastive-Pretraining (aggregated 2 subplots)
try:
    fig, axs = plt.subplots(1, 2, figsize=(14,5))
    # Subplot 1: Loss Curves
    ep_tr, tr_loss = unpack(ab_no_con, ("losses", "train"))
    ep_va, va_loss = unpack(ab_no_con, ("losses", "val"))
    axs[0].plot(ep_tr, tr_loss, marker="o", label="Train")
    axs[0].plot(ep_va, va_loss, marker="s", label="Validation")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("No-Contrastive-Pretraining: Loss Curves")
    axs[0].legend()
    axs[0].grid(True)
    # Subplot 2: Metrics Curves
    for key, lab in zip(["SWA", "CWA", "CompWA"],
                        ["Shape-Weighted Acc", "Color-Weighted Acc", "Complexity-Weighted Acc"]):
        ep, vals = unpack(ab_no_con, ("metrics", key))
        if ep.size:
            axs[1].plot(ep, vals, marker="o", label=lab)
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_title("No-Contrastive-Pretraining: Metrics")
    axs[1].legend()
    axs[1].grid(True)
    fig.suptitle("Ablation: No-Contrastive-Pretraining (Scratch Training)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join("figures", "ablation_no_contrastive_pretraining.png"))
    plt.close()
except Exception as e:
    print("Error plotting ablation No-Contrastive-Pretraining:", e)
    plt.close()

# 11. Ablation: Bag-of-Tokens Input (Token Order Shuffle) -- aggregated 2x2 subplots
try:
    fig, axs = plt.subplots(2, 2, figsize=(14,10))
    # Subplot (0,0): Contrastive Pretrain Loss
    arr = np.array(ab_bag_spr.get("contrastive_pretrain", {}).get("losses", []))
    if arr.size:
        axs[0,0].plot(arr[:,0], arr[:,1], marker="o", color="navy")
    axs[0,0].set_xlabel("Epoch")
    axs[0,0].set_ylabel("NT-Xent Loss")
    axs[0,0].set_title("Bag-of-Tokens: Pretrain Loss")
    axs[0,0].grid(True)
    # Subplot (0,1): Fine-tune Loss Curves
    tr_arr = np.array(ab_bag_spr.get("fine_tune", {}).get("losses", {}).get("train", []))
    va_arr = np.array(ab_bag_spr.get("fine_tune", {}).get("losses", {}).get("val", []))
    if tr_arr.size:
        axs[0,1].plot(tr_arr[:,0], tr_arr[:,1], label="Train", marker="o")
    if va_arr.size:
        axs[0,1].plot(va_arr[:,0], va_arr[:,1], label="Validation", marker="s")
    axs[0,1].set_xlabel("Epoch")
    axs[0,1].set_ylabel("Loss")
    axs[0,1].set_title("Bag-of-Tokens: Fine-tune Loss")
    axs[0,1].legend()
    axs[0,1].grid(True)
    # Subplot (1,0): Metrics Curves
    for key, lab in zip(["SWA", "CWA", "CompWA"],
                        ["Shape-Weighted Acc", "Color-Weighted Acc", "Comp Weighted Acc"]):
        arr_metric = np.array(ab_bag_spr.get("fine_tune", {}).get("metrics", {}).get(key, []))
        if arr_metric.size:
            axs[1,0].plot(arr_metric[:,0], arr_metric[:,1], marker="o", label=lab)
    axs[1,0].set_xlabel("Epoch")
    axs[1,0].set_ylabel("Accuracy")
    axs[1,0].set_title("Bag-of-Tokens: Metrics Curves")
    axs[1,0].legend()
    axs[1,0].grid(True)
    # Subplot (1,1): Final Metrics Bar Chart
    try:
        # For each metric, get final value from the metric curve
        final_vals = []
        labels = ["SWA", "CWA", "CompWA"]
        for key in labels:
            arr_metric = np.array(ab_bag_spr.get("fine_tune", {}).get("metrics", {}).get(key, []))
            if arr_metric.size:
                final_vals.append(arr_metric[-1,1])
            else:
                final_vals.append(0)
        axs[1,1].bar(labels, final_vals, color=["tab:blue", "tab:orange", "tab:green"])
        axs[1,1].set_ylim(0, 1)
        axs[1,1].set_title("Bag-of-Tokens: Final Metrics")
        axs[1,1].grid(True, axis="y")
    except Exception as inner_e:
        print("Error in Bag-of-Tokens final metric bar:", inner_e)
    fig.suptitle("Ablation: Bag-of-Tokens Input (Token Order Shuffle)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join("figures", "ablation_bag_of_tokens.png"))
    plt.close()
except Exception as e:
    print("Error plotting ablation Bag-of-Tokens Input:", e)
    plt.close()

print("Final plots have been saved in the 'figures/' directory.")
#!/usr/bin/env python3
"""
Aggregator script for final scientific plots.
This script loads experiment data from the provided .npy files (exact full paths)
and generates final publication-quality figures stored in the "figures/" directory.
Each plot is wrapped in a try-except block so that a failure in one does not
prevent the rest of the plots from being generated.

The script covers three experiment summaries:
  • BASELINE_SUMMARY
  • RESEARCH_SUMMARY
  • ABLATION_SUMMARY
  
For the ABLATION_SUMMARY, two branches are plotted:
  (a) Uni-Directional Encoder (No Bidirectionality)
  (b) No-Recurrent Mean-Pooling Encoder

All figures are saved with a dpi of 300 and use an increased font size for clarity.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Increase global font size for readability in the final paper
plt.rcParams.update({'font.size': 14})
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Create final figures directory
os.makedirs("figures", exist_ok=True)

# Helper: cyclic style for plots
def _style(idx):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return colors[idx % len(colors)], "-" if idx < len(colors) else "--"

############################################################
# BASELINE_SUMMARY plots
############################################################
# Load baseline data from the provided .npy file
baseline_npy = "experiment_results/experiment_479d9bad55604ae6b69aad9d4d40ac72_proc_2989519/experiment_data.npy"
try:
    baseline_data = np.load(baseline_npy, allow_pickle=True).item()
    runs_dict = baseline_data.get("pretrain_epochs", {}).get("SPR_BENCH", {})
    run_keys = sorted(runs_dict.keys(), key=lambda x: int(x))
except Exception as e:
    print(f"Error loading baseline data from {baseline_npy}: {e}")
    runs_dict = {}
    run_keys = []

# 1. Baseline: Pre-training Loss Curves
try:
    plt.figure(figsize=(8, 6))
    for i, k in enumerate(run_keys):
        losses = runs_dict[k]["losses"].get("pretrain", [])
        if losses:
            c, ls = _style(i)
            plt.plot(range(1, len(losses)+1), losses, label=f"PT = {k}", color=c, linestyle=ls)
    plt.title("SPR BENCH: Pre-training Loss vs Epochs")
    plt.xlabel("Pre-training Epoch")
    plt.ylabel("Loss")
    plt.legend()
    fname = os.path.join("figures", "baseline_pretrain_loss.png")
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating baseline pre-training loss plot: {e}")
    plt.close()

# 2. Baseline: Fine-tuning Loss (Train vs Val)
try:
    plt.figure(figsize=(8, 6))
    for i, k in enumerate(run_keys):
        losses_train = runs_dict[k]["losses"].get("train", [])
        losses_val   = runs_dict[k]["losses"].get("val", [])
        if losses_train or losses_val:
            c, _ = _style(i)
            if losses_train:
                plt.plot(range(1, len(losses_train)+1), losses_train, color=c, linestyle="-", label=f"Train (PT = {k})")
            if losses_val:
                plt.plot(range(1, len(losses_val)+1), losses_val, color=c, linestyle="--", label=f"Val (PT = {k})")
    plt.title("SPR BENCH: Fine-tuning Loss (Train vs Val)")
    plt.xlabel("Fine-tuning Epoch")
    plt.ylabel("Loss")
    plt.legend(ncol=2)
    fname = os.path.join("figures", "baseline_finetune_loss.png")
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating baseline fine-tuning loss plot: {e}")
    plt.close()

# 3. Baseline: SWA, 4. CWA, 5. SCHM curves (one plot per metric)
def plot_metric_baseline(metric_name, file_suffix):
    try:
        plt.figure(figsize=(8,6))
        for i, k in enumerate(run_keys):
            vals = runs_dict[k]["metrics"].get(metric_name, [])
            if vals:
                c, ls = _style(i)
                plt.plot(range(1, len(vals)+1), vals, label=f"{metric_name} (PT = {k})", color=c, linestyle=ls)
        plt.title(f"SPR BENCH: {metric_name} across Fine-tuning Epochs")
        plt.xlabel("Fine-tuning Epoch")
        plt.ylabel(metric_name)
        plt.legend()
        fname = os.path.join("figures", f"baseline_{file_suffix}.png")
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating baseline {metric_name} plot: {e}")
        plt.close()

plot_metric_baseline("SWA", "SWA_curve")
plot_metric_baseline("CWA", "CWA_curve")
plot_metric_baseline("SCHM", "SCHM_curve")

############################################################
# RESEARCH_SUMMARY plots
############################################################
# Load research data from the provided .npy file
research_npy = "experiment_results/experiment_3d06141ae8b047d08d4ff80712400b32_proc_2997847/experiment_data.npy"
try:
    research_data = np.load(research_npy, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading research data from {research_npy}: {e}")
    research_data = {}

# For research summary, assume the key is "SPR_BENCH"
research_key = "SPR_BENCH"
if research_key in research_data:
    ds_data = research_data[research_key]
    # From research summary, the structure is expected to have:
    #   - losses: { "pretrain": [...], "train": [...], "val": [...] }
    #   - metrics: { "val": list of tuples, with SWA, CWA, CompWA }  
    # We aggregate the plotting similar to baseline but for this single branch.
    # 1. Pre-training Loss
    try:
        if "losses" in ds_data and ds_data["losses"].get("pretrain"):
            plt.figure(figsize=(8,6))
            plt.plot(range(1, len(ds_data["losses"]["pretrain"])+1),
                     ds_data["losses"]["pretrain"],
                     label="Pre-train", color="C0")
            plt.title(f"{research_key}: Pre-training Loss vs Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            fname = os.path.join("figures", "research_pretrain_loss.png")
            plt.savefig(fname, dpi=300)
            plt.close()
            print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating research pre-training loss plot: {e}")
        plt.close()

    # 2. Fine-tuning Loss (Train vs Val)
    try:
        if "losses" in ds_data and (ds_data["losses"].get("train") or ds_data["losses"].get("val")):
            plt.figure(figsize=(8,6))
            if ds_data["losses"].get("train"):
                plt.plot(range(1, len(ds_data["losses"]["train"])+1),
                         ds_data["losses"]["train"],
                         label="Train", color="C0", linestyle="-")
            if ds_data["losses"].get("val"):
                plt.plot(range(1, len(ds_data["losses"]["val"])+1),
                         ds_data["losses"]["val"],
                         label="Val", color="C0", linestyle="--")
            plt.title(f"{research_key}: Fine-tuning Loss (Train vs Val)")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            fname = os.path.join("figures", "research_finetune_loss.png")
            plt.savefig(fname, dpi=300)
            plt.close()
            print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating research fine-tuning loss plot: {e}")
        plt.close()

    # 3. Metrics curves: SWA, CWA, CompWA from research_data["metrics"]["val"]
    try:
        metrics_val = ds_data["metrics"].get("val", [])
        # Extract separate lists if available
        swa_vals = [t[0] for t in metrics_val] if metrics_val else []
        cwa_vals = [t[1] for t in metrics_val] if metrics_val else []
        compwa_vals = [t[2] for t in metrics_val] if metrics_val else []
        
        # SWA
        if swa_vals:
            plt.figure(figsize=(8,6))
            plt.plot(range(1, len(swa_vals)+1), swa_vals, marker="o", label="SWA", color="C0")
            plt.title(f"{research_key}: SWA across Fine-tuning Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("SWA")
            plt.legend()
            fname = os.path.join("figures", "research_SWA_curve.png")
            plt.savefig(fname, dpi=300)
            plt.close()
            print(f"Saved {fname}")
        # CWA
        if cwa_vals:
            plt.figure(figsize=(8,6))
            plt.plot(range(1, len(cwa_vals)+1), cwa_vals, marker="s", label="CWA", color="C1")
            plt.title(f"{research_key}: CWA across Fine-tuning Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("CWA")
            plt.legend()
            fname = os.path.join("figures", "research_CWA_curve.png")
            plt.savefig(fname, dpi=300)
            plt.close()
            print(f"Saved {fname}")
        # CompWA
        if compwa_vals:
            plt.figure(figsize=(8,6))
            plt.plot(range(1, len(compwa_vals)+1), compwa_vals, marker="^", label="CompWA", color="C2")
            plt.title(f"{research_key}: CompWA across Fine-tuning Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("CompWA")
            plt.legend()
            fname = os.path.join("figures", "research_CompWA_curve.png")
            plt.savefig(fname, dpi=300)
            plt.close()
            print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating research metrics plots: {e}")
        plt.close()
else:
    print(f"Key {research_key} not found in research data.")

############################################################
# ABLATION_SUMMARY plots
############################################################

# (a) Uni-Directional Encoder (No Bidirectionality)
uni_npy = "experiment_results/experiment_c2ddaef5d0404c108f446778a857630d_proc_3004940/experiment_data.npy"
try:
    uni_data = np.load(uni_npy, allow_pickle=True).item()
    uni_branch = uni_data.get("uni_directional_encoder", {}).get("SPR_BENCH", {})
    if not uni_branch:
        print("No data found for Uni-Directional Encoder / SPR_BENCH")
        uni_epochs_list = []
    else:
        uni_epochs_list = []
        uni_swa_vals = []
        uni_cwa_vals = []
        uni_schm_vals = []
        for ep_str, run_dict in sorted(uni_branch.items(), key=lambda x: int(x[0])):
            uni_epochs_list.append(int(ep_str))
            # Assuming metrics are stored as lists; take the final metric value from each run.
            uni_swa_vals.append(run_dict["metrics"]["SWA"][-1])
            uni_cwa_vals.append(run_dict["metrics"]["CWA"][-1])
            uni_schm_vals.append(run_dict["metrics"]["SCHM"][-1])
except Exception as e:
    print(f"Error loading uni-directional encoder data: {e}")
    uni_epochs_list, uni_swa_vals, uni_cwa_vals, uni_schm_vals = [], [], [], []

# Uni: Plot SWA vs Pre-training Epochs
try:
    if uni_epochs_list and uni_swa_vals:
        plt.figure(figsize=(8,6))
        plt.plot(uni_epochs_list, uni_swa_vals, marker="o", color="C0")
        plt.title("SPR BENCH: SWA vs Pre-training Epochs\n(uni-directional encoder)")
        plt.xlabel("Pre-training Epochs")
        plt.ylabel("SWA")
        plt.grid(alpha=0.3)
        fname = os.path.join("figures", "ablation_uni_SWA_vs_pretrain_epochs.png")
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating uni-directional SWA plot: {e}")
    plt.close()

# Uni: Plot CWA vs Pre-training Epochs
try:
    if uni_epochs_list and uni_cwa_vals:
        plt.figure(figsize=(8,6))
        plt.plot(uni_epochs_list, uni_cwa_vals, marker="s", color="C1")
        plt.title("SPR BENCH: CWA vs Pre-training Epochs\n(uni-directional encoder)")
        plt.xlabel("Pre-training Epochs")
        plt.ylabel("CWA")
        plt.grid(alpha=0.3)
        fname = os.path.join("figures", "ablation_uni_CWA_vs_pretrain_epochs.png")
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating uni-directional CWA plot: {e}")
    plt.close()

# Uni: Plot SCHM vs Pre-training Epochs
try:
    if uni_epochs_list and uni_schm_vals:
        plt.figure(figsize=(8,6))
        plt.plot(uni_epochs_list, uni_schm_vals, marker="^", color="C2")
        plt.title("SPR BENCH: SCHM vs Pre-training Epochs\n(uni-directional encoder)")
        plt.xlabel("Pre-training Epochs")
        plt.ylabel("SCHM")
        plt.grid(alpha=0.3)
        fname = os.path.join("figures", "ablation_uni_SCHM_vs_pretrain_epochs.png")
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating uni-directional SCHM plot: {e}")
    plt.close()

# (b) No-Recurrent Mean-Pooling Encoder
meanpool_npy = "experiment_results/experiment_95b8530ac46546e88e33b99c1f400ce6_proc_3004939/experiment_data.npy"
try:
    meanpool_data = np.load(meanpool_npy, allow_pickle=True).item()
    mp_branch = meanpool_data.get("no_recurrent_mean_pool", {}).get("SPR_BENCH", {})
    if not mp_branch:
        print("No runs found for no-recurrent mean-pooling encoder in SPR_BENCH")
        mp_sorted_epochs = []
    else:
        mp_sorted_epochs = sorted(mp_branch.keys(), key=lambda x: int(x))
except Exception as e:
    print(f"Error loading no-recurrent mean-pooling data: {e}")
    mp_sorted_epochs = []

# For a subset (up to 3 runs) produce per-run subplots with loss curves and metrics.
if mp_sorted_epochs:
    import numpy as np
    indices = np.linspace(0, len(mp_sorted_epochs)-1, num=min(3, len(mp_sorted_epochs)), dtype=int)
    chosen_runs = [mp_sorted_epochs[i] for i in indices]
    for ep in chosen_runs:
        try:
            r = mp_branch[ep]
            losses = r["losses"]
            metrics = r["metrics"]
            epochs_range = np.arange(1, len(losses.get("train", [])) + 1)
            
            fig, axs = plt.subplots(1, 2, figsize=(12,5))
            # Left subplot: Fine-tuning Loss curves
            if "train" in losses and "val" in losses:
                axs[0].plot(epochs_range, losses["train"], label="Train", color="C0")
                axs[0].plot(epochs_range, losses["val"], label="Val", color="C0", linestyle="--")
            axs[0].set_title("Fine-tune Loss")
            axs[0].set_xlabel("Epoch")
            axs[0].set_ylabel("Loss")
            axs[0].legend()
            
            # Right subplot: Metrics (SWA, CWA, SCHM)
            if "SWA" in metrics and "CWA" in metrics and "SCHM" in metrics:
                axs[1].plot(epochs_range, metrics["SWA"], label="SWA", color="C1")
                axs[1].plot(epochs_range, metrics["CWA"], label="CWA", color="C2")
                axs[1].plot(epochs_range, metrics["SCHM"], label="SCHM", color="C3")
            axs[1].set_title("Metrics")
            axs[1].set_xlabel("Epoch")
            axs[1].set_ylabel("Score")
            axs[1].legend()
            
            fig.suptitle(f"SPR BENCH | Pretrain Epochs = {ep}\n(no-recurrent mean-pooling encoder)")
            fname = os.path.join("figures", f"ablation_meanpool_pretrain{ep}.png")
            plt.savefig(fname, dpi=300)
            plt.close(fig)
            print(f"Saved {fname}")
        except Exception as e:
            print(f"Error plotting no-recurrent mean-pooling run {ep}: {e}")
            plt.close()
            
    # Create aggregate plot: final metrics vs pre-training epochs
    try:
        final_SWA = [mp_branch[ep]["metrics"]["SWA"][-1] for ep in mp_sorted_epochs]
        final_CWA = [mp_branch[ep]["metrics"]["CWA"][-1] for ep in mp_sorted_epochs]
        final_SCHM = [mp_branch[ep]["metrics"]["SCHM"][-1] for ep in mp_sorted_epochs]
        x_epochs = [int(ep) for ep in mp_sorted_epochs]
        plt.figure(figsize=(8,6))
        plt.plot(x_epochs, final_SWA, "o-", label="SWA", color="C1")
        plt.plot(x_epochs, final_CWA, "s-", label="CWA", color="C2")
        plt.plot(x_epochs, final_SCHM, "^-", label="SCHM", color="C3")
        plt.title("Final Metrics vs Pre-training Epochs\n(no-recurrent mean-pooling encoder)")
        plt.xlabel("Pre-training Epochs")
        plt.ylabel("Score")
        plt.legend()
        fname = os.path.join("figures", "ablation_meanpool_final_metrics_vs_pretrain.png")
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating aggregate plot for no-recurrent mean-pooling encoder: {e}")
        plt.close()
else:
    print("Skipping no-recurrent mean-pooling encoder plots because no runs were found.")

print("All plots generated and saved in the 'figures/' directory.")
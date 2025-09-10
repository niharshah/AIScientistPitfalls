#!/usr/bin/env python3
"""
Aggregator Script: Final Figures for Context-aware Contrastive Learning Paper

This script loads experimental .npy files (from the baseline, research, and ablation experiments)
and generates a set of final, publishable figures. All figures are saved under the "figures/" directory.
Each plotting block is wrapped in a try-except so that failure in one does not prevent others from running.

File paths (exact, as provided in the summaries):
  - Baseline:    "experiment_results/experiment_3672c0043ce94320857b974d017aa64d_proc_3099457/experiment_data.npy"
  - Research:    "experiment_results/experiment_ed93b5240f074db1b3b8551e7df1174c_proc_3099457/experiment_data.npy"
  - Cross-Dataset Generalization Ablation: "experiment_results/experiment_1c79a16145454f80a31543216a6f7935_proc_3110949/experiment_data.npy"
  - Bag-of-Words Ablation:    "experiment_results/experiment_6c105040a190431f86e62c8c6a0caa9a_proc_3110951/experiment_data.npy"
  - Static Embedding Ablation: "experiment_results/experiment_782371f836704e739e4f801b369d5bd7_proc_3110952/experiment_data.npy"
  - Layer-Depth Ablation:   "experiment_results/experiment_c726533f40bc4508aba3f4adb68345b7_proc_3110950/experiment_data.npy"
  - Aggregation-Head Ablation: "experiment_results/experiment_43c64d00fbc643b99cf9176208481166_proc_3110952/experiment_data.npy"

The figures herein have been carefully selected (about 12 unique plots) to illustrate the main findings.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Increase font sizes for publication quality
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 14})

# Create figures directory
os.makedirs("figures", exist_ok=True)

# Define file paths (use the full and exact paths as in the summaries)
baseline_file   = "experiment_results/experiment_3672c0043ce94320857b974d017aa64d_proc_3099457/experiment_data.npy"
research_file   = "experiment_results/experiment_ed93b5240f074db1b3b8551e7df1174c_proc_3099457/experiment_data.npy"
cross_dataset_file = "experiment_results/experiment_1c79a16145454f80a31543216a6f7935_proc_3110949/experiment_data.npy"
bow_file        = "experiment_results/experiment_6c105040a190431f86e62c8c6a0caa9a_proc_3110951/experiment_data.npy"
static_emb_file = "experiment_results/experiment_782371f836704e739e4f801b369d5bd7_proc_3110952/experiment_data.npy"
layer_depth_file= "experiment_results/experiment_c726533f40bc4508aba3f4adb68345b7_proc_3110950/experiment_data.npy"
agg_head_file   = "experiment_results/experiment_43c64d00fbc643b99cf9176208481166_proc_3110952/experiment_data.npy"

#############################################
# 1. BASELINE: HWA Comparison Across Learning Rates
#############################################
try:
    baseline_data = np.load(baseline_file, allow_pickle=True).item()
    lr_data = baseline_data.get("learning_rate", {})
    plt.figure(figsize=(6,4))
    for lr in sorted(lr_data, key=lambda x: float(x)):
        # Each lr block holds metrics under key "metrics" -> "val": list of tuples (epoch, swa, cwa, hwa)
        epochs, _, _, hwa = zip(*lr_data[lr]["metrics"]["val"])
        plt.plot(epochs, hwa, marker='o', label="lr=" + lr)
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.title("Baseline: HWA Comparison Across Learning Rates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Baseline_HWA_Comparison.png"), dpi=300)
    plt.close()
except Exception as e:
    print("Error in Baseline HWA Comparison:", e)

#############################################
# 2. BASELINE: Best HWA Bar Chart
#############################################
try:
    best_hwa = {}
    for lr in sorted(lr_data, key=lambda x: float(x)):
        hwa_values = [entry[3] for entry in lr_data[lr]["metrics"]["val"]]
        best_hwa[lr] = max(hwa_values) if hwa_values else 0.0
    plt.figure(figsize=(5,4))
    plt.bar(list(best_hwa.keys()), list(best_hwa.values()), color="skyblue")
    plt.xlabel("Learning Rate")
    plt.ylabel("Best HWA")
    plt.title("Baseline: Best HWA per Learning Rate")
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Baseline_Best_HWA.png"), dpi=300)
    plt.close()
except Exception as e:
    print("Error in Baseline Best HWA Bar Chart:", e)

#############################################
# 3. RESEARCH: Training Loss Curves across Hidden Sizes
#############################################
try:
    research_data = np.load(research_file, allow_pickle=True).item()
    hs_data = research_data.get("hidden_size", {})
    plt.figure(figsize=(6,4))
    for hs in sorted(hs_data, key=lambda x: int(x)):
        # Data stored under key "SPR_BENCH" -> "losses" -> "train": list of (epoch, loss)
        train_losses = hs_data[hs]["SPR_BENCH"]["losses"]["train"]
        epochs, losses = zip(*train_losses)
        plt.plot(epochs, losses, marker='o', label="Hidden = " + str(hs))
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Research: Training Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Research_Train_Loss_Curves.png"), dpi=300)
    plt.close()
except Exception as e:
    print("Error in Research Training Loss Curves:", e)

#############################################
# 4. RESEARCH: HWA Curves Across Hidden Sizes
#############################################
try:
    plt.figure(figsize=(6,4))
    for hs in sorted(hs_data, key=lambda x: int(x)):
        metrics = hs_data[hs]["SPR_BENCH"]["metrics"]["val"]
        epochs = [met[0] for met in metrics]
        hwa   = [met[3] for met in metrics]
        plt.plot(epochs, hwa, marker='o', label="Hidden = " + str(hs))
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.title("Research: HWA Curves Across Hidden Sizes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Research_HWA_Curves.png"), dpi=300)
    plt.close()
except Exception as e:
    print("Error in Research HWA Curves:", e)

#############################################
# 5. RESEARCH: Final HWA by Hidden Size (Bar Chart)
#############################################
try:
    final_hwa = {}
    for hs in sorted(hs_data, key=lambda x: int(x)):
        metrics = hs_data[hs]["SPR_BENCH"]["metrics"]["val"]
        hwa_vals = [met[3] for met in metrics]
        final_hwa[hs] = hwa_vals[-1] if hwa_vals else 0.0
    plt.figure(figsize=(5,4))
    plt.bar(list(final_hwa.keys()), list(final_hwa.values()), color="lightgreen")
    plt.xlabel("Hidden Size")
    plt.ylabel("Final HWA")
    plt.title("Research: Final HWA by Hidden Size")
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Research_Final_HWA.png"), dpi=300)
    plt.close()
except Exception as e:
    print("Error in Research Final HWA Bar Chart:", e)

#############################################
# 6. ABLATION - Cross-Dataset Generalization: Loss and HWA for Hidden Size 256
#############################################
try:
    cd_data = np.load(cross_dataset_file, allow_pickle=True).item()
    # Use key "cross_dataset_generalization" with hidden size 256
    cd_256 = cd_data.get("cross_dataset_generalization", {}).get(256, {})
    methods = ["SPR_BENCH", "SPR+SHAPE+COLOR"]
    plt.figure(figsize=(10,4))
    # Subplot 1: Loss curves
    ax1 = plt.subplot(1,2,1)
    for m in methods:
        arr_train = np.array(cd_256[m]["losses"]["train"])  # shape: (E,2)
        arr_val   = np.array(cd_256[m]["losses"]["val"])
        ax1.plot(arr_train[:,0], arr_train[:,1], marker='o', label=f"{m} Train")
        ax1.plot(arr_val[:,0], arr_val[:,1], linestyle="--", marker='o', label=f"{m} Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("CD Generalization: Loss Curves (Hidden=256)")
    ax1.legend(fontsize=10)
    # Subplot 2: HWA curves
    ax2 = plt.subplot(1,2,2)
    for m in methods:
        hw_arr = np.array(cd_256[m]["metrics"]["val"])
        ax2.plot(hw_arr[:,0], hw_arr[:,3], marker='o', label=m)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("HWA")
    ax2.set_title("CD Generalization: HWA Curves (Hidden=256)")
    ax2.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Abla_CD_Generalization_256.png"), dpi=300)
    plt.close()
except Exception as e:
    print("Error in Cross-Dataset Generalization Ablation (256):", e)

#############################################
# 7. ABLATION - Cross-Dataset Generalization: Final HWA Bar Chart Across Hidden Sizes
#############################################
try:
    cd_gen = cd_data.get("cross_dataset_generalization", {})
    h_sizes = sorted(cd_gen.keys())
    methods = ["SPR_BENCH", "SPR+SHAPE+COLOR"]
    hwa_final = {m: [] for m in methods}
    for h in h_sizes:
        for m in methods:
            hw_arr = np.array(cd_gen[h][m]["metrics"]["val"])
            hwa_final[m].append(hw_arr[-1,3])
    x = np.arange(len(h_sizes))
    width = 0.35
    plt.figure(figsize=(6,4))
    plt.bar(x - width/2, hwa_final["SPR_BENCH"], width, label="SPR_BENCH")
    plt.bar(x + width/2, hwa_final["SPR+SHAPE+COLOR"], width, label="SPR+SHAPE+COLOR")
    plt.xlabel("Hidden Size")
    plt.ylabel("Final HWA")
    plt.title("CD Gen: Final HWA by Hidden Size")
    plt.xticks(x, h_sizes)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Abla_CD_Generalization_Final_HWA.png"), dpi=300)
    plt.close()
except Exception as e:
    print("Error in CD Generalization Final HWA Bar Chart:", e)

#############################################
# 8. ABLATION - Bag-of-Words: Loss and Accuracy Curves
#############################################
try:
    bow_data = np.load(bow_file, allow_pickle=True).item()
    bow_run = bow_data.get("bag_of_words", {}).get("SPR_BENCH", {})
    if bow_run:
        # Unpack training and validation losses as well as weighted accuracies
        loss_tr = list(zip(*bow_run["losses"]["train"])) if bow_run["losses"]["train"] else ([], [])
        loss_va = list(zip(*bow_run["losses"]["val"])) if bow_run["losses"]["val"] else ([], [])
        metrics = list(zip(*bow_run["metrics"]["val"])) if bow_run["metrics"]["val"] else ([], [], [], [])
        plt.figure(figsize=(10,4))
        # Subplot 1: Loss Curves
        plt.subplot(1,2,1)
        plt.plot(loss_tr[0], loss_tr[1], marker='o', label="Train Loss")
        plt.plot(loss_va[0], loss_va[1], marker='o', linestyle="--", label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("BoW: Loss Curves")
        plt.legend()
        # Subplot 2: Weighted Accuracy Curves
        plt.subplot(1,2,2)
        plt.plot(metrics[0], metrics[1], marker='o', label="SWA")
        plt.plot(metrics[0], metrics[2], marker='o', label="CWA")
        plt.plot(metrics[0], metrics[3], marker='o', label="HWA")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.title("BoW: Accuracy Curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "Abla_BoW.png"), dpi=300)
        plt.close()
except Exception as e:
    print("Error in Bag-of-Words Ablation plots:", e)

#############################################
# 9. ABLATION - Static Embedding: HWA Curves Across Hidden Sizes
#############################################
try:
    se_data = np.load(static_emb_file, allow_pickle=True).item()
    static_run = se_data.get("static_embedding", {}).get("SPR_BENCH", {})
    plt.figure(figsize=(6,4))
    for hs in sorted(static_run.keys(), key=lambda x: int(x)):
        run = static_run[hs]
        epochs = [e for e, _ in run["losses"]["train"]]
        hwa   = [entry[3] for entry in run["metrics"]["val"]]
        plt.plot(epochs, hwa, marker='o', label="hs=" + str(hs))
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.title("Static Embedding: HWA over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Abla_StaticEmbedding_HWA.png"), dpi=300)
    plt.close()
except Exception as e:
    print("Error in Static Embedding Ablation plot:", e)

#############################################
# 10. ABLATION - Layer-Depth: Loss and HWA Curves (Combined)
#############################################
try:
    ld_data = np.load(layer_depth_file, allow_pickle=True).item()
    depths = sorted(ld_data.get("layer_depth", {}).keys(), key=lambda x: int(x))
    plt.figure(figsize=(10,4))
    # Subplot 1: Loss Curves
    plt.subplot(1,2,1)
    for d in depths:
        train = ld_data["layer_depth"][d]["SPR_BENCH"]["losses"]["train"]
        val   = ld_data["layer_depth"][d]["SPR_BENCH"]["losses"]["val"]
        plt.plot(*zip(*train), marker='o', label=f"depth {d} train")
        plt.plot(*zip(*val), linestyle="--", marker='o', label=f"depth {d} val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Layer-Depth: Loss Curves")
    plt.legend(fontsize=8)
    # Subplot 2: HWA Curves
    plt.subplot(1,2,2)
    for d in depths:
        met = ld_data["layer_depth"][d]["SPR_BENCH"]["metrics"]["val"]
        epochs = [entry[0] for entry in met]
        hwa_vals = [entry[3] for entry in met]
        plt.plot(epochs, hwa_vals, marker='o', label=f"depth {d}")
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.title("Layer-Depth: HWA Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Abla_LayerDepth.png"), dpi=300)
    plt.close()
except Exception as e:
    print("Error in Layer-Depth Ablation plots:", e)

#############################################
# 11. ABLATION - Aggregation-Head: Final Epoch HWA Bar Chart
#############################################
try:
    agg_data = np.load(agg_head_file, allow_pickle=True).item()
    # Get hidden sizes from one variant (they should match for both)
    hs_keys = sorted(agg_data.get("aggregation_head_final_state", {}) \
                      .get("SPR_BENCH", {}) \
                      .get("hidden_size", {}).keys(), key=lambda x: int(x))
    fs_final = []
    mp_final = []
    for hs in hs_keys:
        met_fs = agg_data["aggregation_head_final_state"]["SPR_BENCH"]["hidden_size"][hs]["metrics"]["val"]
        met_mp = agg_data["aggregation_head_mean_pool"]["SPR_BENCH"]["hidden_size"][hs]["metrics"]["val"]
        fs_final.append(met_fs[-1][3])
        mp_final.append(met_mp[-1][3])
    x = np.arange(len(hs_keys))
    width = 0.35
    plt.figure(figsize=(6,4))
    plt.bar(x - width/2, fs_final, width, label="Final-State")
    plt.bar(x + width/2, mp_final, width, label="Mean-Pool")
    plt.xlabel("Hidden Size")
    plt.ylabel("Final Epoch HWA")
    plt.title("Aggregation-Head: Final HWA Comparison")
    plt.xticks(x, hs_keys)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Abla_AggHead_FinalHWA.png"), dpi=300)
    plt.close()
except Exception as e:
    print("Error in Aggregation-Head Final HWA Bar Chart:", e)

#############################################
# 12. ABLATION - Aggregation-Head: Combined HWA Curves for Both Variants
#############################################
try:
    plt.figure(figsize=(6,4))
    for hs in hs_keys:
        met_fs = agg_data["aggregation_head_final_state"]["SPR_BENCH"]["hidden_size"][hs]["metrics"]["val"]
        met_mp = agg_data["aggregation_head_mean_pool"]["SPR_BENCH"]["hidden_size"][hs]["metrics"]["val"]
        epochs_fs = [entry[0] for entry in met_fs]
        hwa_fs    = [entry[3] for entry in met_fs]
        epochs_mp = [entry[0] for entry in met_mp]
        hwa_mp    = [entry[3] for entry in met_mp]
        plt.plot(epochs_fs, hwa_fs, marker='o', linestyle="-", label=f"Final-State hs={hs}")
        plt.plot(epochs_mp, hwa_mp, marker='o', linestyle="--", label=f"Mean-Pool hs={hs}")
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.title("Aggregation-Head: HWA Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Abla_AggHead_HWA_Curves.png"), dpi=300)
    plt.close()
except Exception as e:
    print("Error in Aggregation-Head HWA Curves:", e)

print("All aggregated final figures have been saved in the 'figures/' directory.")
#!/usr/bin/env python3
"""
Final Aggregator Script for SPR Experiments
Aggregates final figures for baseline, research, and ablation experiments.
All figures are saved into the "figures/" directory.
Each plot is wrapped in its own try/except block.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Set a larger font size and remove top/right spines for all plots
plt.rcParams.update({'font.size': 14})
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Create figures directory
os.makedirs("figures", exist_ok=True)

#########################################
# 1. BASELINE EXPERIMENTS (CharBiGRU with dropout tuning)
#########################################

# Baseline file: experiment_results/experiment_96255ae056f642059702c07062aaf285_proc_3330989/experiment_data.npy
baseline_file = "experiment_results/experiment_96255ae056f642059702c07062aaf285_proc_3330989/experiment_data.npy"
try:
    baseline_data = np.load(baseline_file, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading baseline data from {baseline_file}: {e}")
    baseline_data = {}

if baseline_data:
    drop_dict = baseline_data.get("dropout_rate", {})
    dropouts = sorted(drop_dict.keys(), key=lambda x: float(x))
    # Plot BL1: Bar chart of test Macro-F1 vs dropout rate
    try:
        plt.figure(figsize=(6,4))
        test_scores = [drop_dict[r]["test_macro_f1"] for r in dropouts]
        plt.bar([str(r) for r in dropouts], test_scores, color="skyblue")
        plt.xlabel("Dropout rate")
        plt.ylabel("Macro-F1")
        plt.title("Baseline SPR: Test Macro-F1 vs Dropout")
        for i, v in enumerate(test_scores):
            plt.text(i, v+0.01, f"{v:.2f}", ha="center")
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "baseline_test_macroF1_bar.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error creating baseline bar plot: {e}")
        plt.close()

    # Plot BL2: Line plot for one representative dropout curve (choose dropout=0.3 if exists, else first)
    try:
        target_dropout = "0.3" if "0.3" in drop_dict else dropouts[0]
        rec = drop_dict[target_dropout]
        epochs = rec["epochs"]
        tr_f1 = rec["metrics"]["train_macro_f1"]
        val_f1 = rec["metrics"]["val_macro_f1"]
        plt.figure(figsize=(6,4))
        plt.plot(epochs, tr_f1, label="Train")
        plt.plot(epochs, val_f1, label="Validation", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title(f"Baseline SPR: Macro-F1 (dropout={target_dropout})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("figures", f"baseline_macroF1_curve_dropout_{target_dropout}.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error creating baseline dropout curve for dropout={target_dropout}: {e}")
        plt.close()

#########################################
# 2. RESEARCH EXPERIMENTS (Transformer based)
#########################################

# Research file: experiment_results/experiment_8af726a4e254464ab5b3062661aa2bbd_proc_3336029/experiment_data.npy
research_file = "experiment_results/experiment_8af726a4e254464ab5b3062661aa2bbd_proc_3336029/experiment_data.npy"
try:
    research_data = np.load(research_file, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading research data from {research_file}: {e}")
    research_data = {}

if research_data:
    # For research experiments, assume transformer logs are stored under key "transformer".
    logs = research_data.get("transformer", {})
    # Build aggregated data by dropout (similar structure from provided code)
    by_dp = {}
    epochs_info = logs.get("epochs", [])
    tr_loss_all = logs.get("losses", {}).get("train", [])
    val_loss_all = logs.get("losses", {}).get("val", [])
    tr_mcc_all = logs.get("metrics", {}).get("train_MCC", [])
    val_mcc_all = logs.get("metrics", {}).get("val_MCC", [])
    # Organize the records according to dropout value per entry in epochs_info.
    for i, (dp, ep) in enumerate(epochs_info):
        d = by_dp.setdefault(dp, {"epoch": [], "tr_loss": [], "val_loss": [], "tr_mcc": [], "val_mcc": []})
        d["epoch"].append(ep)
        d["tr_loss"].append(tr_loss_all[i] if i < len(tr_loss_all) else np.nan)
        d["val_loss"].append(val_loss_all[i] if i < len(val_loss_all) else np.nan)
        d["tr_mcc"].append(tr_mcc_all[i] if i < len(tr_mcc_all) else np.nan)
        d["val_mcc"].append(val_mcc_all[i] if i < len(val_mcc_all) else np.nan)
    # Plot R: Create one figure with 3 subplots: Loss curves, MCC curves, and Test Metrics bar.
    try:
        fig, axs = plt.subplots(1, 3, figsize=(18,5))
        # (a) Loss curves for each dropout value
        for dp, d in by_dp.items():
            axs[0].plot(d["epoch"], d["tr_loss"], label=f"Train dp={dp}")
            axs[0].plot(d["epoch"], d["val_loss"], linestyle="--", label=f"Val dp={dp}")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("BCE Loss")
        axs[0].set_title("Transformer Loss Curves")
        axs[0].legend()
        # (b) MCC curves
        for dp, d in by_dp.items():
            axs[1].plot(d["epoch"], d["tr_mcc"], label=f"Train dp={dp}")
            axs[1].plot(d["epoch"], d["val_mcc"], linestyle="--", label=f"Val dp={dp}")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("MCC")
        axs[1].set_title("Transformer MCC Curves")
        axs[1].legend()
        # (c) Test Metrics Bar Chart (assuming test_MCC and test_F1 exist)
        test_mcc = logs.get("test_MCC", np.nan)
        test_f1 = logs.get("test_F1", np.nan)
        axs[2].bar(["MCC", "Macro-F1"], [test_mcc, test_f1], color=["salmon", "seagreen"])
        axs[2].set_ylim(0, 1)
        axs[2].set_title("Transformer Test Metrics")
        for i, v in enumerate([test_mcc, test_f1]):
            axs[2].text(i, v+0.02, f"{v:.2f}", ha="center")
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "research_transformer_aggregated.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error creating research aggregated plot: {e}")
        plt.close()

#########################################
# 3. ABLATION STUDIES
#########################################

# (A1) Remove Positional Encoding (No-PE)
# File: experiment_results/experiment_06227a0558ba414888e9901088a64960_proc_3341727/experiment_data.npy
nope_file = "experiment_results/experiment_06227a0558ba414888e9901088a64960_proc_3341727/experiment_data.npy"
try:
    nope_data = np.load(nope_file, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading No-PE data from {nope_file}: {e}")
    nope_data = {}

if nope_data:
    # Assume data stored under key "SPR"
    d = nope_data.get("SPR", {})
    epochs_arr = list(range(1, len(d.get("losses", {}).get("train", [])) + 1))
    try:
        fig, axs = plt.subplots(1, 2, figsize=(12,5))
        # Loss curve
        axs[0].plot(epochs_arr, d.get("losses", {}).get("train", []), label="Train")
        axs[0].plot(epochs_arr, d.get("losses", {}).get("val", []), label="Validation", linestyle="--")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("BCE Loss")
        axs[0].set_title("No-PE: Loss Curve")
        axs[0].legend()
        # Confusion matrix (build from predictions and g.t.)
        preds = np.array(d.get("predictions", []))
        gts = np.array(d.get("ground_truth", []))
        if preds.size and gts.size:
            cm = np.zeros((2,2), dtype=int)
            for gt, pr in zip(gts, preds):
                cm[int(gt), int(pr)] += 1
            im = axs[1].imshow(cm, cmap="Blues")
            axs[1].set_xticks([0,1])
            axs[1].set_xticklabels(["Pred 0","Pred 1"])
            axs[1].set_yticks([0,1])
            axs[1].set_yticklabels(["True 0","True 1"])
            axs[1].set_title("No-PE: Confusion Matrix")
            for i in range(2):
                for j in range(2):
                    axs[1].text(j, i, cm[i,j], ha="center", va="center", color="black")
        else:
            axs[1].text(0.5,0.5,"No data", ha="center")
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "ablation_noPE_loss_and_CM.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error plotting Remove Positional Encoding figures: {e}")
        plt.close()

# (A2) No-Transformer / Bag-of-Embeddings
# File: experiment_results/experiment_cf6c600544354624af8d8a1860baf831_proc_3341728/experiment_data.npy
nobe_file = "experiment_results/experiment_cf6c600544354624af8d8a1860baf831_proc_3341728/experiment_data.npy"
try:
    nobe_data = np.load(nobe_file, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading Bag-of-Embeddings data from {nobe_file}: {e}")
    nobe_data = {}

if nobe_data:
    models_keys = list(nobe_data.keys())
    # Create aggregated figure with two subplots: one for MCC curves and one for final test MCC bar.
    try:
        fig, axs = plt.subplots(1, 2, figsize=(12,5))
        # Left: MCC curves for each model from the bag-of-embeddings ablation.
        for model in models_keys:
            mcc_train = nobe_data[model]["metrics"].get("train_MCC", [])
            mcc_val = nobe_data[model]["metrics"].get("val_MCC", [])
            axs[0].plot(mcc_train, label=f"{model} Train")
            axs[0].plot(mcc_val, linestyle="--", label=f"{model} Val")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("MCC")
        axs[0].set_title("Bag-of-Embeddings: MCC Curves")
        axs[0].legend()
        # Right: Bar chart comparing final test MCC for each model.
        test_mcc_scores = [nobe_data[m].get("test_MCC", np.nan) for m in models_keys]
        axs[1].bar(models_keys, test_mcc_scores, color=["tab:blue", "tab:orange"])
        axs[1].set_ylabel("Test MCC")
        axs[1].set_title("Bag-of-Embeddings: Test MCC Comparison")
        for i, v in enumerate(test_mcc_scores):
            axs[1].text(i, v+0.02, f"{v:.2f}", ha="center")
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "ablation_bagofembeddings.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error plotting Bag-of-Embeddings ablation: {e}")
        plt.close()

# (A3) Max-Pool Aggregation (Mean→Max)
# File: experiment_results/experiment_f2eba17077454b8ea1998869cbe13395_proc_3341729/experiment_data.npy
maxpool_file = "experiment_results/experiment_f2eba17077454b8ea1998869cbe13395_proc_3341729/experiment_data.npy"
try:
    maxpool_data = np.load(maxpool_file, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading Max-Pool data from {maxpool_file}: {e}")
    maxpool_data = {}

if maxpool_data:
    ds_key = "SPR_BENCH"
    mdl_key = "max_pool"
    # Navigate safely into the nested dict structure
    ed = maxpool_data.get(mdl_key, {}).get(ds_key, {})
    if ed:
        epochs = list(range(1, len(ed.get("losses", {}).get("train", [])) + 1))
        try:
            fig, axs = plt.subplots(1, 2, figsize=(12,5))
            # Loss curves
            axs[0].plot(epochs, ed.get("losses", {}).get("train", []), label="Train")
            axs[0].plot(epochs, ed.get("losses", {}).get("val", []), label="Validation", linestyle="--")
            axs[0].set_xlabel("Epoch")
            axs[0].set_ylabel("BCE Loss")
            axs[0].set_title(f"{ds_key}: Loss (Transformer MaxPool)")
            axs[0].legend()
            # MCC curves
            axs[1].plot(epochs, ed.get("metrics", {}).get("train_MCC", []), label="Train MCC")
            axs[1].plot(epochs, ed.get("metrics", {}).get("val_MCC", []), label="Val MCC", linestyle="--")
            axs[1].set_xlabel("Epoch")
            axs[1].set_ylabel("MCC")
            axs[1].set_title(f"{ds_key}: MCC (Transformer MaxPool)")
            axs[1].legend()
            plt.tight_layout()
            plt.savefig(os.path.join("figures", "ablation_maxpool_loss_MCC.png"), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Error plotting Max-Pool Aggregation figures: {e}")
            plt.close()

# (A4) No-PadMask
# File: experiment_results/experiment_f9639073656545ef9eefde1a358ef47c_proc_3341730/experiment_data.npy
nopad_file = "experiment_results/experiment_f9639073656545ef9eefde1a358ef47c_proc_3341730/experiment_data.npy"
try:
    nopad_data = np.load(nopad_file, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading No-PadMask data from {nopad_file}: {e}")
    nopad_data = {}

if nopad_data:
    ed = nopad_data.get("no_padmask", {}).get("spr", {})
    if ed:
        epochs = np.arange(1, len(ed.get("losses", {}).get("train", [])) + 1)
        try:
            fig, axs = plt.subplots(1, 2, figsize=(12,5))
            # Loss curve
            axs[0].plot(epochs, ed.get("losses", {}).get("train", []), label="Train")
            axs[0].plot(epochs, ed.get("losses", {}).get("val", []), label="Validation", linestyle="--")
            axs[0].set_xlabel("Epoch")
            axs[0].set_ylabel("BCE Loss")
            axs[0].set_title("No-PadMask: Loss Curve")
            axs[0].legend()
            # MCC curve
            axs[1].plot(epochs, ed.get("metrics", {}).get("train_MCC", []), label="Train")
            axs[1].plot(epochs, ed.get("metrics", {}).get("val_MCC", []), label="Validation", linestyle="--")
            axs[1].set_xlabel("Epoch")
            axs[1].set_ylabel("MCC")
            axs[1].set_title("No-PadMask: MCC Curve")
            axs[1].legend()
            plt.tight_layout()
            plt.savefig(os.path.join("figures", "ablation_nopadmask_loss_MCC.png"), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Error plotting No-PadMask figures: {e}")
            plt.close()
        # Additionally, create a separate plot for test metrics and confusion matrix
        try:
            plt.figure(figsize=(6,5))
            plt.bar(["Test MCC", "Test F1"], [ed.get("test_MCC", 0), ed.get("test_F1", 0)], color=["skyblue", "salmon"])
            for i, v in enumerate([ed.get("test_MCC", 0), ed.get("test_F1", 0)]):
                plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
            plt.ylim(0, 1)
            plt.title("No-PadMask: Test Metrics")
            plt.tight_layout()
            plt.savefig(os.path.join("figures", "ablation_nopadmask_test_metrics.png"), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Error plotting No-PadMask test metrics: {e}")
            plt.close()

# (A5) Learned Positional Embedding (Sinusoidal vs Learned)
# File: experiment_results/experiment_3c09a4163dfc413ebf953d8a4a75308b_proc_3341727/experiment_data.npy
learned_file = "experiment_results/experiment_3c09a4163dfc413ebf953d8a4a75308b_proc_3341727/experiment_data.npy"
try:
    learned_data = np.load(learned_file, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading Learned Positional Embedding data from {learned_file}: {e}")
    learned_data = {}

if learned_data:
    # Expect keys "positional_encoding" with subkeys "sinusoidal" and "learned"
    pe_data = learned_data.get("positional_encoding", {})
    try:
        fig, axs = plt.subplots(1, 2, figsize=(12,5))
        # Left: Loss curves for both variants
        for variant in ["sinusoidal", "learned"]:
            exp = pe_data.get(variant, {})
            if exp:
                axs[0].plot(exp.get("losses", {}).get("train", []), label=f"{variant.capitalize()} Train")
                axs[0].plot(exp.get("losses", {}).get("val", []), linestyle="--", label=f"{variant.capitalize()} Val")
        axs[0].set_xlabel("Epoch Update")
        axs[0].set_ylabel("BCE Loss")
        axs[0].set_title("Positional Embedding: Loss Curves")
        axs[0].legend()
        # Right: Bar plot comparing test MCC and test F1 for both variants (side-by-side)
        metrics = ["test_MCC", "test_F1"]
        x = np.arange(len(metrics))
        width = 0.35
        # For each variant, get metric values
        sinusoidal_vals = [pe_data.get("sinusoidal", {}).get(m, 0) for m in metrics]
        learned_vals = [pe_data.get("learned", {}).get(m, 0) for m in metrics]
        axs[1].bar(x - width/2, sinusoidal_vals, width, label="Sinusoidal", color="dodgerblue")
        axs[1].bar(x + width/2, learned_vals, width, label="Learned", color="darkorange")
        axs[1].set_xticks(x)
        axs[1].set_xticklabels([m.replace("_", " ") for m in metrics])
        axs[1].set_ylabel("Metric Value")
        axs[1].set_title("Positional Embedding: Test Metrics")
        axs[1].legend()
        for i in range(len(metrics)):
            axs[1].text(i - width/2, sinusoidal_vals[i] + 0.02, f"{sinusoidal_vals[i]:.2f}", ha="center")
            axs[1].text(i + width/2, learned_vals[i] + 0.02, f"{learned_vals[i]:.2f}", ha="center")
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "ablation_learnedPE.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error plotting Learned Positional Embedding figures: {e}")
        plt.close()

# (A6) No-FFN (Attention-Only Transformer)
# File: experiment_results/experiment_c78a45dfa87049ea9220a2bb10acaf7b_proc_3341730/experiment_data.npy
noffn_file = "experiment_results/experiment_c78a45dfa87049ea9220a2bb10acaf7b_proc_3341730/experiment_data.npy"
try:
    noffn_data = np.load(noffn_file, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading No-FFN data from {noffn_file}: {e}")
    noffn_data = {}

if noffn_data:
    exp = noffn_data.get("no_ffn", {}).get("SPR", {})
    if exp:
        try:
            fig, axs = plt.subplots(1, 3, figsize=(18,5))
            # Loss curves
            axs[0].plot(exp.get("losses", {}).get("train", []), label="Train")
            axs[0].plot(exp.get("losses", {}).get("val", []), label="Validation", linestyle="--")
            axs[0].set_xlabel("Epoch")
            axs[0].set_ylabel("BCE Loss")
            axs[0].set_title("No-FFN: Loss Curve")
            axs[0].legend()
            # MCC curves
            axs[1].plot(exp.get("metrics", {}).get("train_MCC", []), label="Train")
            axs[1].plot(exp.get("metrics", {}).get("val_MCC", []), label="Validation", linestyle="--")
            axs[1].set_xlabel("Epoch")
            axs[1].set_ylabel("MCC")
            axs[1].set_title("No-FFN: MCC Curve")
            axs[1].legend()
            # Confusion Matrix
            preds = np.array(exp.get("predictions", []), dtype=int)
            gts = np.array(exp.get("ground_truth", []), dtype=int)
            if preds.size and gts.size:
                cm = np.zeros((2, 2), dtype=int)
                for p, g in zip(preds, gts):
                    cm[g, p] += 1
                im = axs[2].imshow(cm, cmap="Blues")
                axs[2].set_xticks([0,1])
                axs[2].set_xticklabels(["Pred 0", "Pred 1"])
                axs[2].set_yticks([0,1])
                axs[2].set_yticklabels(["True 0", "True 1"])
                axs[2].set_title("No-FFN: Confusion Matrix")
                for i in range(2):
                    for j in range(2):
                        axs[2].text(j, i, cm[i,j], ha="center", va="center", color="white" if cm[i,j] > cm.max()/2 else "black")
            else:
                axs[2].text(0.5, 0.5, "No Data", ha="center")
            plt.tight_layout()
            plt.savefig(os.path.join("figures", "ablation_noFFN.png"), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Error plotting No-FFN figures: {e}")
            plt.close()

print("Final aggregated figures saved in the 'figures/' directory.")
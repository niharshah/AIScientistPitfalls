#!/usr/bin/env python3
"""
Final Aggregator Script for Contextual Embedding-based SPR Experiments

This script loads experiment data from multiple .npy files (using full, exact file paths
from the experiment summaries) and produces publishable final plots.
All final figures are saved in the "figures/" folder.
Each plotting block is wrapped in its own try-except so one plot’s failure does not stop others.
Plots include:
  • Baseline results (accuracy curves, loss curves, test accuracy bar chart, confusion matrix)
  • Ablation: RemovePositionalEmbeddings (aggregated test accuracy vs nheads + sample training curves)
  • Ablation: MultiSyntheticDatasets (EvenParity, MajoritySymbol, CyclicShift – each in one figure)
  • Ablation: NoTransformerEncoder (accuracy and loss curves in one figure)
  • Ablation: SingleTransformerLayer (accuracy curves, loss curves, test accuracy bar)
  
All plots use an increased font size for readability.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Increase font size for publication quality
plt.rcParams.update({'font.size': 14})
for spine in ["top", "right"]:
    plt.rcParams['axes.spines.' + spine] = False

# Create output folder for figures
os.makedirs("figures", exist_ok=True)

###############################
# Section 1: Baseline Results #
###############################
# File path from summary
baseline_npy = "experiment_results/experiment_1a788b2273c9434aa4a5f57864d9af39_proc_3161120/experiment_data.npy"
try:
    data_baseline = np.load(baseline_npy, allow_pickle=True).item()
    results_baseline = data_baseline.get("nhead_tuning", {}).get("SPR_BENCH", {}).get("results", {})
    if not results_baseline:
        raise ValueError("Baseline results not found")
except Exception as e:
    print("Error loading Baseline experiment data:", e)
    results_baseline = {}

# Determine best nhead (for confusion matrix)
best_nhead = None
best_test_acc = -1
for nhead, d in results_baseline.items():
    if d.get("test_acc", 0) > best_test_acc:
        best_test_acc = d.get("test_acc", 0)
        best_nhead = nhead

# Plot 1: Baseline Accuracy Curves
try:
    plt.figure(figsize=(8,6))
    for nhead, d in results_baseline.items():
        epochs = range(1, len(d["metrics"].get("train_acc", [])) + 1)
        plt.plot(epochs, d["metrics"].get("train_acc", []), marker="o", label=f"Train nhead={nhead}")
        plt.plot(epochs, d["metrics"].get("val_acc", []), marker="s", linestyle="--", label=f"Val nhead={nhead}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Baseline: SPR_BENCH Accuracy Curves (n-head tuning)")
    plt.legend()
    plt.tight_layout()
    out_file = os.path.join("figures", "Baseline_Accuracy_Curves.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print("Saved", out_file)
except Exception as e:
    print("Error plotting Baseline Accuracy Curves:", e)
    plt.close()

# Plot 2: Baseline Loss Curves
try:
    plt.figure(figsize=(8,6))
    for nhead, d in results_baseline.items():
        epochs = range(1, len(d["losses"].get("train_loss", [])) + 1)
        plt.plot(epochs, d["losses"].get("train_loss", []), marker="o", label=f"Train nhead={nhead}")
        plt.plot(epochs, d["losses"].get("val_loss", []), marker="s", linestyle="--", label=f"Val nhead={nhead}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Baseline: SPR_BENCH Loss Curves (n-head tuning)")
    plt.legend()
    plt.tight_layout()
    out_file = os.path.join("figures", "Baseline_Loss_Curves.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print("Saved", out_file)
except Exception as e:
    print("Error plotting Baseline Loss Curves:", e)
    plt.close()

# Plot 3: Baseline Test Accuracy Bar Chart
try:
    plt.figure(figsize=(8,6))
    nheads = list(results_baseline.keys())
    test_accs = [results_baseline[n].get("test_acc", 0) for n in nheads]
    plt.bar(nheads, test_accs, color="skyblue")
    plt.xlabel("n-head")
    plt.ylabel("Test Accuracy")
    plt.title("Baseline: SPR_BENCH Test Accuracy by n-head")
    for i, acc in enumerate(test_accs):
        plt.text(i, acc + 0.01, f"{acc:.2f}", ha="center")
    plt.tight_layout()
    out_file = os.path.join("figures", "Baseline_Test_Accuracy.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print("Saved", out_file)
except Exception as e:
    print("Error plotting Baseline Test Accuracy:", e)
    plt.close()

# Plot 4: Baseline Confusion Matrix (using best nhead)
try:
    if best_nhead is not None:
        d = results_baseline[best_nhead]
        preds = np.array(d.get("predictions", []))
        gts = np.array(d.get("ground_truth", []))
        if preds.size and gts.size:
            num_classes = len(np.unique(gts))
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for p, g in zip(preds, gts):
                cm[g, p] += 1
            plt.figure(figsize=(6,5))
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(f"Baseline: SPR_BENCH Confusion Matrix (best nhead={best_nhead})")
            for i in range(num_classes):
                for j in range(num_classes):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
            plt.tight_layout()
            out_file = os.path.join("figures", "Baseline_Confusion_Matrix.png")
            plt.savefig(out_file, dpi=300)
            plt.close()
            print("Saved", out_file)
    else:
        print("No best nhead found for confusion matrix.")
except Exception as e:
    print("Error plotting Baseline Confusion Matrix:", e)
    plt.close()

#####################################################
# Section 2: Ablation - RemovePositionalEmbeddings   #
#####################################################
# File path from summary
rm_pos_emb_npy = "experiment_results/experiment_8ff134f7504443ea954ca8ff3dcc3aca_proc_3173696/experiment_data.npy"
try:
    data_rmpos = np.load(rm_pos_emb_npy, allow_pickle=True).item()
    results_rmpos = data_rmpos.get("RemovePositionalEmbeddings", {}).get("SPR_BENCH", {}).get("results", {})
    if not results_rmpos:
        raise ValueError("RemovePositionalEmbeddings results not found")
except Exception as e:
    print("Error loading RemovePositionalEmbeddings experiment data:", e)
    results_rmpos = {}

# Plot 5: Aggregated Test Accuracy vs. nhead for RemovePositionalEmbeddings
try:
    heads = []
    test_accs = []
    for head, res in sorted(results_rmpos.items(), key=lambda x: int(x[0])):
        acc = res.get("test_acc")
        if acc is not None:
            heads.append(int(head))
            test_accs.append(acc)
    if heads:
        plt.figure(figsize=(8,6))
        plt.plot(heads, test_accs, marker="o", label="Test Accuracy")
        plt.xlabel("Number of Attention Heads")
        plt.ylabel("Test Accuracy")
        plt.title("RemovePositionalEmbeddings: Test Accuracy vs. nhead")
        plt.legend()
        plt.tight_layout()
        out_file = os.path.join("figures", "RemovePositionalEmbeddings_TestAccuracy_vs_nheads.png")
        plt.savefig(out_file, dpi=300)
        plt.close()
        print("Saved", out_file)
    else:
        print("No valid heads data for RemovePositionalEmbeddings test accuracy plot.")
except Exception as e:
    print("Error plotting RemovePositionalEmbeddings test accuracy vs nhead:", e)
    plt.close()

# Plot 6: Sample Per-head Training Curves (Accuracy and Loss) for RemovePositionalEmbeddings
# For simplicity, combine first two head settings (if available) using subplots.
try:
    sorted_heads = sorted(results_rmpos.keys(), key=lambda x: int(x))
    selected = sorted_heads[:2]  # choose first 2 head configs
    fig, axes = plt.subplots(1, 2, figsize=(14,6))
    for idx, head in enumerate(selected):
        res = results_rmpos[head]
        epochs = range(1, len(res.get("metrics", {}).get("train_acc", [])) + 1)
        axes[idx].plot(epochs, res["metrics"].get("train_acc", []), marker="o", label="Train Acc")
        axes[idx].plot(epochs, res["metrics"].get("val_acc", []), marker="s", linestyle="--", label="Val Acc")
        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel("Accuracy")
        axes[idx].set_title(f"nhead={head} Accuracy")
        axes[idx].legend()
    plt.suptitle("RemovePositionalEmbeddings: Sample Training Accuracy Curves")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_file = os.path.join("figures", "RemovePositionalEmbeddings_TrainingCurves.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print("Saved", out_file)
except Exception as e:
    print("Error plotting RemovePositionalEmbeddings training curves:", e)
    plt.close()

#####################################################
# Section 3: Ablation - MultiSyntheticDatasets        #
#####################################################
# File path from summary
multi_syn_npy = "experiment_results/experiment_4211a431bb0b4a979d8603c36575fa48_proc_3173697/experiment_data.npy"
try:
    data_multi = np.load(multi_syn_npy, allow_pickle=True).item()
    multisyn = data_multi.get("MultiSyntheticDatasets", {})
    if not multisyn:
        raise ValueError("MultiSyntheticDatasets results not found")
except Exception as e:
    print("Error loading MultiSyntheticDatasets experiment data:", e)
    multisyn = {}

# For each synthetic task, create one figure with 3 subplots
for task_name, task_data in multisyn.items():
    try:
        nheads = sorted(task_data.keys(), key=lambda x: int(x))
        # Subplot layout: 1 row, 3 columns: Accuracy curves, Loss curves, Test accuracy bar
        fig, axes = plt.subplots(1, 3, figsize=(18,5))
        # Accuracy curves
        for nh in nheads:
            log = task_data[nh]
            epochs = range(1, len(log.get("metrics", {}).get("train", [])) + 1)
            axes[0].plot(epochs, log["metrics"].get("train", []), marker="o", label=f"nhead={nh} train")
            axes[0].plot(epochs, log["metrics"].get("val", []), marker="x", linestyle="--", label=f"nhead={nh} val")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title(f"{task_name}: Accuracy Curves")
        axes[0].legend()
        # Loss curves
        for nh in nheads:
            log = task_data[nh]
            epochs = range(1, len(log.get("losses", {}).get("train", [])) + 1)
            axes[1].plot(epochs, log["losses"].get("train", []), marker="o", label=f"nhead={nh} train")
            axes[1].plot(epochs, log["losses"].get("val", []), marker="x", linestyle="--", label=f"nhead={nh} val")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].set_title(f"{task_name}: Loss Curves")
        axes[1].legend()
        # Test accuracy bar chart
        test_accs = [task_data[nh].get("test_acc", 0) for nh in nheads]
        axes[2].bar([str(nh) for nh in nheads], test_accs, color="skyblue")
        axes[2].set_xlabel("nhead")
        axes[2].set_ylabel("Test Accuracy")
        axes[2].set_title(f"{task_name}: Test Accuracy")
        for idx, acc in enumerate(test_accs):
            axes[2].text(idx, acc + 0.01, f"{acc:.2f}", ha="center")
        plt.suptitle(f"MultiSyntheticDatasets - {task_name} Results")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_file = os.path.join("figures", f"MultiSynthetic_{task_name}_Results.png")
        plt.savefig(out_file, dpi=300)
        plt.close()
        print("Saved", out_file)
    except Exception as e:
        print(f"Error plotting MultiSynthetic task {task_name} results:", e)
        plt.close()

#####################################################
# Section 4: Ablation - NoTransformerEncoder          #
#####################################################
# File path from summary
no_trans_enc_npy = "experiment_results/experiment_e46f3c6a839048e584bf3541709fdb79_proc_3173699/experiment_data.npy"
try:
    data_notrans = np.load(no_trans_enc_npy, allow_pickle=True).item()
except Exception as e:
    print("Error loading NoTransformerEncoder experiment data:", e)
    data_notrans = {}

# For each dataset/model pair in NoTransformerEncoder, produce one combined figure (accuracy & loss)
try:
    # We assume one key at the top-level, iterate over models and datasets:
    for model_name, ds_data in data_notrans.items():
        # ds_data should be a dict with key as dataset name (e.g., "SPR_BENCH")
        for ds_name, record in ds_data.items():
            train_acc = record.get("metrics", {}).get("train", [])
            val_acc = record.get("metrics", {}).get("val", [])
            train_loss = record.get("losses", {}).get("train", [])
            val_loss = record.get("losses", {}).get("val", [])
            epochs = range(1, len(train_acc) + 1)
            fig, axes = plt.subplots(1, 2, figsize=(12,5))
            axes[0].plot(epochs, train_acc, marker="o", label="Train")
            axes[0].plot(epochs, val_acc, marker="s", linestyle="--", label="Validation")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Accuracy")
            axes[0].set_title(f"{model_name} on {ds_name}: Accuracy")
            axes[0].legend()
            axes[1].plot(epochs, train_loss, marker="o", label="Train")
            axes[1].plot(epochs, val_loss, marker="s", linestyle="--", label="Validation")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Loss")
            axes[1].set_title(f"{model_name} on {ds_name}: Loss")
            axes[1].legend()
            plt.suptitle("NoTransformerEncoder: Training Curves")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            out_file = os.path.join("figures", "NoTransformerEncoder_Curves.png")
            plt.savefig(out_file, dpi=300)
            plt.close()
            print("Saved", out_file)
except Exception as e:
    print("Error plotting NoTransformerEncoder curves:", e)
    plt.close()

#####################################################
# Section 5: Ablation - SingleTransformerLayer       #
#####################################################
# File path from summary
single_layer_npy = "experiment_results/experiment_2a41055a408c4fd09a30dc44e5fc15e7_proc_3173698/experiment_data.npy"
try:
    data_single = np.load(single_layer_npy, allow_pickle=True).item()
    results_single = data_single.get("SingleTransformerLayer", {}).get("SPR_BENCH", {}).get("results", {})
    if not results_single:
        raise ValueError("SingleTransformerLayer results not found")
except Exception as e:
    print("Error loading SingleTransformerLayer experiment data:", e)
    results_single = {}

# Plot 7: SingleTransformerLayer Accuracy Curves
try:
    plt.figure(figsize=(8,6))
    for nh, res in sorted(results_single.items(), key=lambda x: int(x[0])):
        epochs = range(1, len(res["metrics"].get("train", [])) + 1)
        plt.plot(epochs, res["metrics"].get("train", []), marker="o", label=f"{nh}-head Train")
        plt.plot(epochs, res["metrics"].get("val", []), marker="s", linestyle="--", label=f"{nh}-head Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SingleTransformerLayer: Accuracy Curves (SPR_BENCH)")
    plt.legend()
    plt.tight_layout()
    out_file = os.path.join("figures", "SingleTransformerLayer_Accuracy.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print("Saved", out_file)
except Exception as e:
    print("Error plotting SingleTransformerLayer Accuracy Curves:", e)
    plt.close()

# Plot 8: SingleTransformerLayer Loss Curves
try:
    plt.figure(figsize=(8,6))
    for nh, res in sorted(results_single.items(), key=lambda x: int(x[0])):
        epochs = range(1, len(res["losses"].get("train", [])) + 1)
        plt.plot(epochs, res["losses"].get("train", []), marker="o", label=f"{nh}-head Train")
        plt.plot(epochs, res["losses"].get("val", []), marker="s", linestyle="--", label=f"{nh}-head Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SingleTransformerLayer: Loss Curves (SPR_BENCH)")
    plt.legend()
    plt.tight_layout()
    out_file = os.path.join("figures", "SingleTransformerLayer_Loss.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print("Saved", out_file)
except Exception as e:
    print("Error plotting SingleTransformerLayer Loss Curves:", e)
    plt.close()

# Plot 9: SingleTransformerLayer Test Accuracy Bar Chart
try:
    nheads = []
    test_accs = []
    for nh, res in sorted(results_single.items(), key=lambda x: int(x[0])):
        nheads.append(nh)
        test_accs.append(res.get("test_acc", 0))
    plt.figure(figsize=(8,6))
    plt.bar(nheads, test_accs, color="skyblue")
    plt.xlabel("n-head")
    plt.ylabel("Test Accuracy")
    plt.title("SingleTransformerLayer: Test Accuracy (SPR_BENCH)")
    for idx, acc in enumerate(test_accs):
        plt.text(idx, acc + 0.01, f"{acc:.2f}", ha="center")
    plt.ylim(0,1)
    plt.tight_layout()
    out_file = os.path.join("figures", "SingleTransformerLayer_TestAccuracy.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print("Saved", out_file)
except Exception as e:
    print("Error plotting SingleTransformerLayer Test Accuracy Bar Chart:", e)
    plt.close()

print("Final aggregated plotting complete. All figures saved in 'figures/'")
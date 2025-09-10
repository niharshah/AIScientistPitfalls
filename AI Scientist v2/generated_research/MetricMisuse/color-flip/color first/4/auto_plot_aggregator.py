#!/usr/bin/env python3
"""
Final Aggregator Script for Scientific Figures in "figures/"

This script loads experimental .npy data as saved from various experiments
(including baseline, research and several ablation studies) and produces a
comprehensive set of final, publication‐ready figures. Each figure is created
inside a try/except block so that failure in one does not abort the entire
aggregation process.

All figures are saved in the "figures/" folder with a dpi of 300 and enhanced
font sizes for readability.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Increase font size for publication-quality figures
plt.rcParams.update({"font.size": 14})
os.makedirs("figures", exist_ok=True)

def save_and_close(fig, filename):
    fig.savefig(os.path.join("figures", filename), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------- FIGURE 1: Baseline Loss and Accuracy Curves -----------------------
try:
    # Load baseline experiment data (learning rate sweep)
    baseline_file = "experiment_results/experiment_afd01f5cd3e54ce2be1593003c6e591f_proc_1635406/experiment_data.npy"
    exp_data = np.load(baseline_file, allow_pickle=True).item()
    # Select keys beginning with "lr_"
    lr_keys = sorted([k for k in exp_data if k.startswith("lr_")],
                     key=lambda x: float(x.split("_")[1]))
    epochs = None
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # Left plot: Train and Validation Loss curves by learning rate
    for lr in lr_keys:
        rec = exp_data[lr]["SPR_BENCH"]
        train_loss = rec["losses"]["train"]
        val_loss = rec["losses"]["val"]
        if epochs is None:
            epochs = range(1, len(train_loss) + 1)
        lr_val = lr.split("_")[1]
        axs[0].plot(epochs, train_loss, label=f"Train (lr={lr_val})")
        axs[0].plot(epochs, val_loss, "--", label=f"Val (lr={lr_val})")
    axs[0].set_title("Baseline Loss Curves")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # Right plot: Validation Accuracy curves by learning rate
    for lr in lr_keys:
        rec = exp_data[lr]["SPR_BENCH"]
        # Some experiments use lower-case keys, others uppercase.
        acc = [d.get("acc", d.get("ACC", 0)) for d in rec["metrics"]["val"]]
        axs[1].plot(epochs, acc, marker="o", label=f"lr={lr.split('_')[1]}")
    axs[1].set_title("Baseline Validation Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()

    save_and_close(fig, "Baseline_Loss_Accuracy.png")
except Exception as e:
    print("Error in Baseline Loss/Accuracy plot:", e)


# ----------------------- FIGURE 2: Baseline Weighted Accuracies -----------------------
try:
    # Reload baseline data (same file)
    exp_data = np.load(baseline_file, allow_pickle=True).item()
    lr_keys = sorted([k for k in exp_data if k.startswith("lr_")],
                     key=lambda x: float(x.split("_")[1]))
    epochs = None
    fig, ax = plt.subplots(figsize=(8, 6))
    for lr in lr_keys:
        rec = exp_data[lr]["SPR_BENCH"]
        if epochs is None:
            epochs = range(1, len(rec["losses"]["train"]) + 1)
        # Use weighted metrics: Color, Shape and PC weighted accuracy
        cwa = [d.get("cwa", d.get("CWA", 0)) for d in rec["metrics"]["val"]]
        swa = [d.get("swa", d.get("SWA", 0)) for d in rec["metrics"]["val"]]
        pcwa = [d.get("pcwa", d.get("PCWA", 0)) for d in rec["metrics"]["val"]]
        ax.plot(epochs, cwa, label=f"CWA (lr={lr.split('_')[1]})")
        ax.plot(epochs, swa, "--", label=f"SWA (lr={lr.split('_')[1]})")
        ax.plot(epochs, pcwa, ":", label=f"PCWA (lr={lr.split('_')[1]})")
    ax.set_title("Baseline Weighted Accuracies")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.legend()
    save_and_close(fig, "Baseline_Weighted_Accuracies.png")
except Exception as e:
    print("Error in Baseline Weighted Accuracies plot:", e)


# ----------------------- FIGURE 3: MeanPool No RNN Ablation -----------------------
try:
    meanpool_file = "experiment_results/experiment_da3c4a4c9c164cd6859778325f259012_proc_1695393/experiment_data.npy"
    exp_data = np.load(meanpool_file, allow_pickle=True).item()
    rec = exp_data["MeanPool_NoRNN"]["SPR_BENCH"]
    epochs = range(1, len(rec["losses"]["train"]) + 1)
    # Plot Loss Curves in one figure
    fig_loss, ax_loss = plt.subplots(figsize=(8, 6))
    ax_loss.plot(epochs, rec["losses"]["train"], label="Train Loss")
    ax_loss.plot(epochs, rec["losses"]["val"], label="Validation Loss", linestyle="--")
    ax_loss.set_title("MeanPool No RNN Loss Curves")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    save_and_close(fig_loss, "MeanPool_No_RNN_Loss_Curves.png")
    
    # Plot Metrics in a 2x2 grid (ACC, CWA, SWA, CompWA)
    metrics = rec["metrics"]["val"]
    metric_info = [("acc", "Accuracy"),
                   ("CWA", "Color Weighted Accuracy"),
                   ("SWA", "Shape Weighted Accuracy"),
                   ("CompWA", "Complexity Weighted Accuracy")]
    fig_metrics, axs = plt.subplots(2, 2, figsize=(10, 8))
    for i, (key, label_text) in enumerate(metric_info):
        vals = [m.get(key, 0) for m in metrics]
        ax = axs[i // 2, i % 2]
        ax.plot(epochs, vals, marker="o")
        ax.set_title(label_text)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label_text)
    plt.tight_layout()
    save_and_close(fig_metrics, "MeanPool_No_RNN_Metrics.png")
except Exception as e:
    print("Error in MeanPool No RNN plots:", e)


# ----------------------- FIGURE 4: No Latent Glyph Clustering Ablation -----------------------
try:
    nlgc_file = "experiment_results/experiment_c960521e027b4512bf3f0ed8083c3288_proc_1695394/experiment_data.npy"
    exp_data = np.load(nlgc_file, allow_pickle=True).item()
    rec = exp_data["no_latent_glyph_clustering"]["SPR_BENCH"]
    epochs = range(1, len(rec["losses"]["train"]) + 1)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    # Top-left: Loss Curves
    axs[0, 0].plot(epochs, rec["losses"]["train"], label="Train")
    axs[0, 0].plot(epochs, rec["losses"]["val"], label="Val", linestyle="--")
    axs[0, 0].set_title("No Latent Clustering Loss")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend()
    # Top-right: Validation Accuracy
    acc = [m.get("acc", 0) for m in rec["metrics"]["val"]]
    axs[0, 1].plot(epochs, acc, marker="o")
    axs[0, 1].set_title("Validation Accuracy")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Accuracy")
    # Bottom-left: Weighted Metrics (CWA, SWA, CompWA)
    cwa = [m.get("CWA", 0) for m in rec["metrics"]["val"]]
    swa = [m.get("SWA", 0) for m in rec["metrics"]["val"]]
    comp = [m.get("CompWA", 0) for m in rec["metrics"]["val"]]
    axs[1, 0].plot(epochs, cwa, label="CWA")
    axs[1, 0].plot(epochs, swa, "--", label="SWA")
    axs[1, 0].plot(epochs, comp, ":", label="CompWA")
    axs[1, 0].set_title("Weighted Metrics")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("Score")
    axs[1, 0].legend()
    # Bottom-right: Confusion Matrix
    preds = rec.get("predictions", [])
    gts = rec.get("ground_truth", [])
    if preds and gts:
        preds = np.array(preds)
        gts = np.array(gts)
        n_cls = int(max(preds.max(), gts.max())) + 1
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        im = axs[1, 1].imshow(cm, cmap="Blues")
        axs[1, 1].set_title("Confusion Matrix")
        axs[1, 1].set_xlabel("Predicted")
        axs[1, 1].set_ylabel("True")
        fig.colorbar(im, ax=axs[1, 1])
    else:
        axs[1, 1].text(0.5, 0.5, "No Data", ha="center")
    plt.tight_layout()
    save_and_close(fig, "No_Latent_Glyph_Clustering.png")
except Exception as e:
    print("Error in No Latent Glyph Clustering plot:", e)


# ----------------------- FIGURE 5: Frozen Cluster Embeddings Ablation -----------------------
try:
    frozen_file = "experiment_results/experiment_e4f6330735c84e469b805711c8e742f0_proc_1695395/experiment_data.npy"
    exp_data = np.load(frozen_file, allow_pickle=True).item()
    if "frozen_cluster" in exp_data and "SPR_BENCH" in exp_data["frozen_cluster"]:
        rec = exp_data["frozen_cluster"]["SPR_BENCH"]
    else:
        raise ValueError("Frozen cluster data not found")
    epochs = range(1, len(rec["losses"]["train"]) + 1)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    # Left: Loss Curves
    axs[0].plot(epochs, rec["losses"]["train"], label="Train")
    axs[0].plot(epochs, rec["losses"]["val"], label="Val", linestyle="--")
    axs[0].set_title("Frozen Cluster Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    # Middle: Validation Metrics
    acc = [m.get("acc", 0) for m in rec["metrics"]["val"]]
    cwa = [m.get("CWA", 0) for m in rec["metrics"]["val"]]
    swa = [m.get("SWA", 0) for m in rec["metrics"]["val"]]
    comp = [m.get("CompWA", 0) for m in rec["metrics"]["val"]]
    axs[1].plot(epochs, acc, label="ACC")
    axs[1].plot(epochs, cwa, "--", label="CWA")
    axs[1].plot(epochs, swa, ":", label="SWA")
    axs[1].plot(epochs, comp, "-.", label="CompWA")
    axs[1].set_title("Frozen Cluster Metrics")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Score")
    axs[1].legend()
    # Right: Confusion Matrix
    preds = rec.get("predictions", [])
    gts = rec.get("ground_truth", [])
    if preds and gts:
        preds = np.array(preds)
        gts = np.array(gts)
        n_cls = int(max(preds.max(), gts.max())) + 1
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        im = axs[2].imshow(cm, cmap="Blues")
        axs[2].set_title("Frozen Cluster Confusion Matrix")
        axs[2].set_xlabel("Predicted")
        axs[2].set_ylabel("True")
        fig.colorbar(im, ax=axs[2])
    else:
        axs[2].text(0.5, 0.5, "No Data", ha="center")
    plt.tight_layout()
    save_and_close(fig, "Frozen_Cluster_Embeddings.png")
except Exception as e:
    print("Error in Frozen Cluster Embeddings plot:", e)


# ----------------------- FIGURE 6: UniGRU No Bidirection Ablation -----------------------
try:
    unid_file = "experiment_results/experiment_9734f0f677e64b63825122b6893547eb_proc_1695396/experiment_data.npy"
    exp_data = np.load(unid_file, allow_pickle=True).item()
    rec = exp_data["UniGRU_No_Bidirection"]["SPR_BENCH"]
    epochs = range(1, len(rec["losses"]["train"]) + 1)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    # Top-left: Loss Curves
    axs[0, 0].plot(epochs, rec["losses"]["train"], label="Train")
    axs[0, 0].plot(epochs, rec["losses"]["val"], label="Val", linestyle="--")
    axs[0, 0].set_title("UniGRU No Bidirection Loss")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend()
    # Top-right: Validation Accuracy
    acc = [m.get("acc", 0) for m in rec["metrics"]["val"]]
    axs[0, 1].plot(epochs, acc, marker="o")
    axs[0, 1].set_title("Validation Accuracy")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Accuracy")
    # Bottom-left: Weighted Metrics
    cwa = [m.get("CWA", 0) for m in rec["metrics"]["val"]]
    swa = [m.get("SWA", 0) for m in rec["metrics"]["val"]]
    comp = [m.get("CompWA", 0) for m in rec["metrics"]["val"]]
    axs[1, 0].plot(epochs, cwa, label="CWA")
    axs[1, 0].plot(epochs, swa, "--", label="SWA")
    axs[1, 0].plot(epochs, comp, ":", label="CompWA")
    axs[1, 0].set_title("Weighted Metrics")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("Score")
    axs[1, 0].legend()
    # Bottom-right: Confusion Matrix
    preds = rec.get("predictions", [])
    gts = rec.get("ground_truth", [])
    if preds and gts:
        preds = np.array(preds)
        gts = np.array(gts)
        n_cls = int(max(preds.max(), gts.max())) + 1
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        im = axs[1, 1].imshow(cm, cmap="Blues")
        axs[1, 1].set_title("Confusion Matrix")
        axs[1, 1].set_xlabel("Predicted")
        axs[1, 1].set_ylabel("True")
        fig.colorbar(im, ax=axs[1, 1])
    else:
        axs[1, 1].text(0.5, 0.5, "No Data", ha="center")
    plt.tight_layout()
    save_and_close(fig, "UniGRU_No_Bidirection.png")
except Exception as e:
    print("Error in UniGRU No Bidirection plot:", e)


# ----------------------- FIGURE 7: Random Glyph Clustering Ablation -----------------------
try:
    rand_file = "experiment_results/experiment_281d25c680e3444788db265be24a3653_proc_1695393/experiment_data.npy"
    exp_data = np.load(rand_file, allow_pickle=True).item()
    # The file contains several ablation variants.
    ablation_keys = list(exp_data.keys())
    # (a) Loss curves across ablations in one figure
    fig, ax = plt.subplots(figsize=(8, 6))
    for key in ablation_keys:
        rec = exp_data[key]["SPR_BENCH"]
        epochs = range(1, len(rec["losses"]["train"]) + 1)
        ax.plot(epochs, rec["losses"]["train"], "--", label=f"{key} Train")
        ax.plot(epochs, rec["losses"]["val"], label=f"{key} Val")
    ax.set_title("Random Glyph Clustering Loss Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    save_and_close(fig, "Random_Glyph_Clustering_Loss.png")
    
    # (b) Validation Accuracy curves across ablations
    fig, ax = plt.subplots(figsize=(8, 6))
    for key in ablation_keys:
        rec = exp_data[key]["SPR_BENCH"]
        epochs = range(1, len(rec["losses"]["train"]) + 1)
        acc = [m.get("acc", 0) for m in rec["metrics"]["val"]]
        ax.plot(epochs, acc, marker="o", label=key)
    ax.set_title("Random Glyph Clustering Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    save_and_close(fig, "Random_Glyph_Clustering_Accuracy.png")
    
    # (c) Bar chart of final weighted metrics (CWA, SWA, CompWA) for each ablation
    metrics_list = ["CWA", "SWA", "CompWA"]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(ablation_keys))
    width = 0.25
    for i, met in enumerate(metrics_list):
        vals = []
        for key in ablation_keys:
            rec = exp_data[key]["SPR_BENCH"]
            final = rec["metrics"]["val"][-1]
            vals.append(final.get(met, 0))
        ax.bar(x + i * width, vals, width, label=met)
    ax.set_xticks(x + width)
    ax.set_xticklabels(ablation_keys, rotation=45)
    ax.set_ylabel("Score")
    ax.set_title("Random Glyph Clustering Final Weighted Metrics")
    ax.legend()
    save_and_close(fig, "Random_Glyph_Clustering_Final_Metrics.png")
except Exception as e:
    print("Error in Random Glyph Clustering plots:", e)


# ----------------------- FIGURE 8: Token Order Shuffled Input Ablation -----------------------
try:
    shuffle_file = "experiment_results/experiment_b918f6b859094d89b4e059232084a231_proc_1695395/experiment_data.npy"
    exp_data = np.load(shuffle_file, allow_pickle=True).item()
    rec = exp_data["token_order_shuffled"]["SPR_BENCH"]
    epochs = range(1, len(rec["losses"]["train"]) + 1)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    # Loss Curves
    axs[0].plot(epochs, rec["losses"]["train"], label="Train")
    axs[0].plot(epochs, rec["losses"]["val"], "--", label="Val")
    axs[0].set_title("Shuffled Input Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    # Validation Metrics: Accuracy, CWA, SWA, and CompWA
    acc = [m.get("acc", 0) for m in rec["metrics"]["val"]]
    cwa = [m.get("CWA", 0) for m in rec["metrics"]["val"]]
    swa = [m.get("SWA", 0) for m in rec["metrics"]["val"]]
    comp = [m.get("CompWA", 0) for m in rec["metrics"]["val"]]
    axs[1].plot(epochs, acc, label="ACC")
    axs[1].plot(epochs, cwa, "--", label="CWA")
    axs[1].plot(epochs, swa, ":", label="SWA")
    axs[1].plot(epochs, comp, "-.", label="CompWA")
    axs[1].set_title("Shuffled Input Metrics")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Score")
    axs[1].legend()
    # Confusion Matrix
    preds = rec.get("predictions", [])
    gts = rec.get("ground_truth", [])
    if preds and gts:
        preds = np.array(preds)
        gts = np.array(gts)
        n_cls = int(max(preds.max(), gts.max())) + 1
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        im = axs[2].imshow(cm, cmap="Blues")
        axs[2].set_title("Shuffled Input Confusion Matrix")
        axs[2].set_xlabel("Predicted")
        axs[2].set_ylabel("True")
        fig.colorbar(im, ax=axs[2])
    else:
        axs[2].text(0.5, 0.5, "No Data", ha="center")
    plt.tight_layout()
    save_and_close(fig, "Token_Order_Shuffled_Input.png")
except Exception as e:
    print("Error in Token Order Shuffled Input plot:", e)


# ----------------------- FIGURE 9: Reinit RNN After Clustering Ablation -----------------------
try:
    reinit_file = "experiment_results/experiment_4e4be4f9935a4fd7a2ca125be7e1938d_proc_1695396/experiment_data.npy"
    exp_data = np.load(reinit_file, allow_pickle=True).item()
    # Take first two experiments for comparison
    exp_keys = list(exp_data.keys())[:2]
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    # Left: Loss curves from each experiment
    for key in exp_keys:
        rec = exp_data[key]["SPR_BENCH"]
        epochs = range(1, len(rec["losses"]["train"]) + 1)
        axs[0].plot(epochs, rec["losses"]["train"], "--", label=f"{key} Train")
        axs[0].plot(epochs, rec["losses"]["val"], label=f"{key} Val")
    axs[0].set_title("Reinit RNN Loss Curves")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    # Middle: Validation Accuracy curves
    for key in exp_keys:
        rec = exp_data[key]["SPR_BENCH"]
        epochs = range(1, len(rec["losses"]["train"]) + 1)
        acc = [m.get("acc", 0) for m in rec["metrics"]["val"]]
        axs[1].plot(epochs, acc, marker="o", label=key)
    axs[1].set_title("Reinit RNN Validation Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    # Right: Bar chart of final accuracies for comparison
    final_acc = []
    for key in exp_keys:
        rec = exp_data[key]["SPR_BENCH"]
        final = rec["metrics"]["val"][-1].get("acc", 0)
        final_acc.append(final)
    axs[2].bar(exp_keys, final_acc)
    axs[2].set_title("Final Accuracy Comparison")
    axs[2].set_xlabel("Experiment")
    axs[2].set_ylabel("Accuracy")
    axs[2].set_ylim(0, 1)
    plt.tight_layout()
    save_and_close(fig, "Reinit_RNN_After_Clustering.png")
except Exception as e:
    print("Error in Reinit RNN After Clustering plot:", e)


print("All final figures have been saved in the 'figures/' directory.")
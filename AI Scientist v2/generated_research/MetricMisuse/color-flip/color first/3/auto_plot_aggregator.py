#!/usr/bin/env python3
"""
Final Aggregator Script for Comprehensive Research Plots

This script loads experimental results from several .npy files (from Baseline,
Research and selected Ablation experiments) and produces publication‐quality
plots in the "figures/" directory. Each figure is produced in its own try/except
block so that an error in one plot does not prevent the others from running.
All plots use enlarged fonts, no top/right spines, and are saved with dpi=300.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Set global plotting parameters for readability.
plt.rcParams.update({'font.size': 12})
os.makedirs("figures", exist_ok=True)

# Helper: Remove top and right spines from a given axis.
def remove_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

###############################################################################
#                              BASELINE PLOTS                               #
###############################################################################
try:
    # Load Baseline experiment data (hyperparameter tuning results)
    baseline_file = "experiment_results/experiment_64f198e0273b438caff4e7125383fc9c_proc_1605335/experiment_data.npy"
    baseline_data = np.load(baseline_file, allow_pickle=True).item()
    runs = baseline_data["epochs_tuning"]["SPR_BENCH"]["runs"]

    # 1. Baseline: Train vs. Validation Loss Curves (combined for all runs)
    plt.figure(figsize=(8, 6), dpi=300)
    for name, run in runs.items():
        # Helper to unpack tuple lists.
        def unpack(lst, idx):
            return [t[idx] for t in lst]
        tr_epochs = unpack(run["losses"]["train"], 0)
        tr_loss   = unpack(run["losses"]["train"], 1)
        val_epochs = unpack(run["losses"]["val"], 0)
        val_loss   = unpack(run["losses"]["val"], 1)
        plt.plot(tr_epochs, tr_loss, "--", label=f"{name} Train")
        plt.plot(val_epochs, val_loss, "-", label=f"{name} Val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Baseline: Train vs. Validation Loss")
    plt.legend()
    ax = plt.gca()
    remove_spines(ax)
    plt.tight_layout()
    fname = os.path.join("figures", "Baseline_Loss_Curves.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print("Error in Baseline Loss Curves:", e)

try:
    # 2. Baseline: Validation HCSA Curves
    plt.figure(figsize=(8, 6), dpi=300)
    for name, run in runs.items():
        epochs = [t[0] for t in run["metrics"]["val"]]
        hcs    = [t[3] for t in run["metrics"]["val"]]
        plt.plot(epochs, hcs, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("HCSA")
    plt.title("Baseline: Validation HCSA")
    plt.legend()
    ax = plt.gca()
    remove_spines(ax)
    plt.tight_layout()
    fname = os.path.join("figures", "Baseline_Validation_HCSA.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print("Error in Baseline Validation HCSA:", e)

try:
    # 3. Baseline: Best HCSA per Run (Bar Chart)
    best_vals, labels = [], []
    for name, run in runs.items():
        hcs_list = [t[3] for t in run["metrics"]["val"]]
        if hcs_list:
            best_vals.append(max(hcs_list))
            labels.append(name)
    plt.figure(figsize=(8, 6), dpi=300)
    plt.bar(range(len(best_vals)), best_vals, tick_label=labels)
    plt.ylabel("Best Validation HCSA")
    plt.title("Baseline: Best HCSA per Run")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fname = os.path.join("figures", "Baseline_Best_HCSA_Bar.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print("Error in Baseline Best HCSA Bar Chart:", e)

###############################################################################
#                              RESEARCH PLOTS                                 #
###############################################################################
try:
    # Load Research experiment data
    research_file = "experiment_results/experiment_b3a7aca055a2450a97402dc98ed6bf18_proc_1610517/experiment_data.npy"
    research_data = np.load(research_file, allow_pickle=True).item()
    rd = research_data["SPR_BENCH"]

    # 4. Research: Train vs. Validation Loss Curves
    plt.figure(figsize=(8, 6), dpi=300)
    tr = np.array(rd["losses"]["train"])
    val = np.array(rd["losses"]["val"])
    if tr.size:
        plt.plot(tr[:, 0], tr[:, 1], "--", label="Train")
    if val.size:
        plt.plot(val[:, 0], val[:, 1], "-", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Research: Train vs. Validation Loss")
    plt.legend()
    ax = plt.gca()
    remove_spines(ax)
    plt.tight_layout()
    fname = os.path.join("figures", "Research_Loss_Curves.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print("Error in Research Loss Curves:", e)

try:
    # 5. Research: 2x2 Grid of Validation Metrics (CWA, SWA, HCSA, SNWA)
    metrics_val = np.array(rd["metrics"]["val"])
    if metrics_val.size and metrics_val.shape[1] >= 5:
        epochs = metrics_val[:, 0]
        cwa = metrics_val[:, 1]
        swa = metrics_val[:, 2]
        hcs = metrics_val[:, 3]
        snwa = metrics_val[:, 4]
        fig, axs = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
        axs = axs.flatten()
        axs[0].plot(epochs, cwa, "-o")
        axs[0].set_title("CWA")
        axs[1].plot(epochs, swa, "-o")
        axs[1].set_title("SWA")
        axs[2].plot(epochs, hcs, "-o")
        axs[2].set_title("HCSA")
        axs[3].plot(epochs, snwa, "-o")
        axs[3].set_title("SNWA")
        for ax in axs:
            ax.set_xlabel("Epoch")
            remove_spines(ax)
        fig.suptitle("Research: Validation Metrics")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fname = os.path.join("figures", "Research_Validation_Metrics.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    else:
        print("No metric data available in Research experiment for grid plot.")
except Exception as e:
    print("Error in Research Metric Curves:", e)

try:
    # 6. Research: Dev vs. Test Accuracy Bar Chart
    # Compute simple accuracy from stored predictions vs ground truth.
    for split in ["dev", "test"]:
        preds = np.array(rd["predictions"].get(split, []))
        gts   = np.array(rd["ground_truth"].get(split, []))
        rd.setdefault("acc", {})[split] = (preds == gts).mean() if preds.size > 0 else np.nan
    acc_dev, acc_test = rd["acc"]["dev"], rd["acc"]["test"]
    plt.figure(figsize=(6, 6), dpi=300)
    plt.bar(["Dev", "Test"], [acc_dev, acc_test], color=["steelblue", "orange"])
    plt.ylabel("Accuracy")
    plt.title("Research: Dev vs. Test Accuracy")
    plt.tight_layout()
    fname = os.path.join("figures", "Research_Dev_vs_Test_Accuracy.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print("Error in Research Accuracy Bar Chart:", e)

###############################################################################
#                      ABLATION: No-Glyph-Clustering                         #
###############################################################################
try:
    # Load No-Glyph-Clustering experiment data
    ngc_file = "experiment_results/experiment_70f75bb111704a5b8f1a47333c47a030_proc_1615733/experiment_data.npy"
    ngc_data = np.load(ngc_file, allow_pickle=True).item()
    # Expect structure: outer dict with exp names; use the SPR_BENCH dataset.
    for exp_name, datasets in ngc_data.items():
        for dset_name, d in datasets.items():
            if dset_name != "SPR_BENCH":
                continue
            # 7. No-Glyph-Clustering: Loss Curves
            loss_train = np.array(d["losses"]["train"]) if d["losses"]["train"] else np.empty((0, 2))
            loss_val   = np.array(d["losses"]["val"])
            plt.figure(figsize=(8, 6), dpi=300)
            if loss_train.size:
                plt.plot(loss_train[:, 0], loss_train[:, 1], label="Train")
            if loss_val.size:
                plt.plot(loss_val[:, 0], loss_val[:, 1], label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("No-Glyph-Clustering: Loss Curves")
            plt.legend()
            ax = plt.gca()
            remove_spines(ax)
            plt.tight_layout()
            fname = os.path.join("figures", "NoGlyphClustering_Loss_Curves.png")
            plt.savefig(fname)
            plt.close()
            print(f"Saved {fname}")

            # 8. No-Glyph-Clustering: Validation Metrics Grid (CWA, SWA, HCSA, SNWA)
            metrics_val = np.array(d["metrics"]["val"])
            if metrics_val.size and metrics_val.shape[1] >= 5:
                epochs = metrics_val[:, 0]
                cwa = metrics_val[:, 1]
                swa = metrics_val[:, 2]
                hcs = metrics_val[:, 3]
                snwa = metrics_val[:, 4]
                fig, axs = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
                axs[0].plot(epochs, cwa, "-o")
                axs[0].set_title("CWA")
                axs[1].plot(epochs, swa, "-o")
                axs[1].set_title("SWA")
                axs[2].plot(epochs, hcs, "-o")
                axs[2].set_title("HCSA")
                axs[3].plot(epochs, snwa, "-o")
                axs[3].set_title("SNWA")
                for ax in axs.flatten():
                    ax.set_xlabel("Epoch")
                    remove_spines(ax)
                fig.suptitle("No-Glyph-Clustering: Validation Metrics")
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                fname = os.path.join("figures", "NoGlyphClustering_Validation_Metrics.png")
                plt.savefig(fname)
                plt.close()
                print(f"Saved {fname}")
except Exception as e:
    print("Error in No-Glyph-Clustering plots:", e)

###############################################################################
#                      ABLATION: No-Bidirectional-GRU                          #
###############################################################################
try:
    nbg_file = "experiment_results/experiment_aedc4406de7a47af85a9f739966c4222_proc_1615735/experiment_data.npy"
    nbg_data = np.load(nbg_file, allow_pickle=True).item()
    model_key = "No-Bidirectional-GRU"
    dataset_key = "SPR_BENCH"
    ed = nbg_data.get(model_key, {}).get(dataset_key, {})

    # Helper: Downsample list to at most 5 points.
    def downsample(lst, max_pts=5):
        if len(lst) <= max_pts:
            return lst
        step = max(1, len(lst) // max_pts)
        return lst[::step]

    # 9. No-Bidirectional-GRU: Loss Curves
    train_loss = downsample(ed["losses"]["train"])
    val_loss   = downsample(ed["losses"]["val"])
    tr_epochs, tr_vals = zip(*train_loss)
    va_epochs, va_vals = zip(*val_loss)
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(tr_epochs, tr_vals, label="Train Loss")
    plt.plot(va_epochs, va_vals, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("No-Bidirectional-GRU: Loss Curves")
    plt.legend()
    ax = plt.gca()
    remove_spines(ax)
    plt.tight_layout()
    fname = os.path.join("figures", "NoBidirectionalGRU_Loss_Curves.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")

    # 10. No-Bidirectional-GRU: Validation Metrics (CWA, SWA, HCSA, SNWA)
    metrics = downsample(ed["metrics"]["val"])
    epochs, cwa, swa, hcs, snwa = zip(*metrics)
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(epochs, cwa, label="CWA")
    plt.plot(epochs, swa, label="SWA")
    plt.plot(epochs, hcs, label="HCSA")
    plt.plot(epochs, snwa, label="SNWA")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("No-Bidirectional-GRU: Validation Metrics")
    plt.legend()
    ax = plt.gca()
    remove_spines(ax)
    plt.tight_layout()
    fname = os.path.join("figures", "NoBidirectionalGRU_Validation_Metrics.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")

    # Confusion Matrix helper function.
    def plot_confusion(preds, gts, split, filename):
        num_cls = int(max(max(preds), max(gts))) + 1
        cm = np.zeros((num_cls, num_cls), dtype=int)
        for g, p in zip(preds, gts):
            cm[g, p] += 1
        plt.figure(figsize=(5, 4), dpi=300)
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(f"No-Bidirectional-GRU: Confusion Matrix ({split})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        for i in range(num_cls):
            for j in range(num_cls):
                plt.text(j, i, cm[i, j], ha="center", va="center",
                         fontsize=8, color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.tight_layout()
        plt.savefig(os.path.join("figures", filename))
        plt.close()

    plot_confusion(np.array(ed["predictions"].get("dev", [])),
                   np.array(ed["ground_truth"].get("dev", [])),
                   "Dev", "NoBidirectionalGRU_Confusion_Dev.png")
    plot_confusion(np.array(ed["predictions"].get("test", [])),
                   np.array(ed["ground_truth"].get("test", [])),
                   "Test", "NoBidirectionalGRU_Confusion_Test.png")
    print("Saved No-Bidirectional-GRU confusion matrices.")
except Exception as e:
    print("Error in No-Bidirectional-GRU plots:", e)

###############################################################################
#                    ABLATION: Mean-Pooling-Encoder                          #
###############################################################################
try:
    mpe_file = "experiment_results/experiment_3b478b9922e749bb98a7276122bb9792_proc_1615734/experiment_data.npy"
    mpe_data = np.load(mpe_file, allow_pickle=True).item()
    model_key, dataset_key = "MeanPoolEncoder", "SPR_BENCH"
    d = mpe_data.get(model_key, {}).get(dataset_key, {})

    # 11. Mean-Pooling-Encoder: Loss Curves
    loss_train = d.get("losses", {}).get("train", [])
    loss_val   = d.get("losses", {}).get("val", [])
    def to_xy(arr, idx=1):
        if not arr:
            return np.array([]), np.array([])
        arr = np.array(arr)
        return arr[:, 0], arr[:, idx]
    ep_tr, loss_tr = to_xy(loss_train)
    ep_val, loss_v = to_xy(loss_val)
    if ep_tr.size and ep_val.size:
        plt.figure(figsize=(8, 6), dpi=300)
        plt.plot(ep_tr, loss_tr, label="Train")
        plt.plot(ep_val, loss_v, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Mean-Pooling-Encoder: Train vs. Validation Loss")
        plt.legend()
        ax = plt.gca()
        remove_spines(ax)
        plt.tight_layout()
        fname = os.path.join("figures", "MeanPoolingEncoder_Loss_Curves.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    # Mean-Pooling-Encoder: Validation Metrics (HCSA & SNWA)
    metrics_val = d.get("metrics", {}).get("val", [])
    if metrics_val:
        metrics_val = np.array(metrics_val)
        epochs = metrics_val[:, 0]
        hcs = metrics_val[:, 3]
        snwa = metrics_val[:, 4]
        plt.figure(figsize=(8, 6), dpi=300)
        plt.plot(epochs, hcs, label="HCSA")
        plt.plot(epochs, snwa, label="SNWA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Mean-Pooling-Encoder: Validation Metrics")
        plt.legend()
        ax = plt.gca()
        remove_spines(ax)
        plt.tight_layout()
        fname = os.path.join("figures", "MeanPoolingEncoder_Validation_Metrics.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
except Exception as e:
    print("Error in Mean-Pooling-Encoder plots:", e)

###############################################################################
#                    ABLATION: No-Sequence-Packing                           #
###############################################################################
try:
    nsp_file = "experiment_results/experiment_7d840f36ece440d38ab83a9cbfcb16cf_proc_1615736/experiment_data.npy"
    nsp_data = np.load(nsp_file, allow_pickle=True).item()
    # For No-Sequence-Packing, we assume the outer dict has one key (algorithm) and its dataset.
    algo = list(nsp_data.keys())[0]
    dset = list(nsp_data[algo].keys())[0]  # expect "SPR_BENCH"
    record = nsp_data[algo][dset]

    def tup2arr(tups):
        if not tups:
            return np.array([]), np.array([])
        ep, vals = zip(*tups)
        return np.array(ep), np.array(vals)

    # 12. No-Sequence-Packing: Loss Curves
    tr_ep, tr_loss = tup2arr(record["losses"]["train"])
    va_ep, va_loss = tup2arr(record["losses"]["val"])
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(tr_ep, tr_loss, label="Train")
    plt.plot(va_ep, va_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("No-Sequence-Packing: Loss Curves")
    plt.legend()
    ax = plt.gca()
    remove_spines(ax)
    plt.tight_layout()
    fname = os.path.join("figures", "NoSequencePacking_Loss_Curves.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")

    # No-Sequence-Packing: Validation Metrics Plot
    try:
        vals = np.array(record["metrics"]["val"])
        epochs = vals[:, 0]
        labels_list = ["CWA", "SWA", "HCSA", "SNWA"]
        plt.figure(figsize=(8, 6), dpi=300)
        for i, lab in enumerate(labels_list, start=1):
            plt.plot(epochs, vals[:, i], label=lab)
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("No-Sequence-Packing: Validation Metrics")
        plt.legend()
        ax = plt.gca()
        remove_spines(ax)
        plt.tight_layout()
        fname = os.path.join("figures", "NoSequencePacking_Validation_Metrics.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print("Error in No-Sequence-Packing metric plot:", e)

    # Confusion matrices for dev and test splits.
    def plot_cm(split, filename):
        preds = np.array(record["predictions"][split])
        gts = np.array(record["ground_truth"][split])
        if preds.size == 0 or gts.size == 0:
            return
        n_cls = int(max(preds.max(), gts.max()) + 1)
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for p, g in zip(preds, gts):
            cm[g, p] += 1
        plt.figure(figsize=(5, 4), dpi=300)
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"No-Sequence-Packing: Confusion Matrix ({split})")
        plt.tight_layout()
        plt.savefig(os.path.join("figures", filename))
        plt.close()

    for split in ["dev", "test"]:
        fname = f"NoSequencePacking_Confusion_{split}.png"
        plot_cm(split, fname)
        print(f"Saved {fname} in figures")
except Exception as e:
    print("Error in No-Sequence-Packing plots:", e)

print("Final aggregation of plots complete.")
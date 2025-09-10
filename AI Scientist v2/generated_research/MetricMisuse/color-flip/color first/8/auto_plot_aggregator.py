#!/usr/bin/env python3
"""
Final Aggregator Script for SPR Experiments
Generates publication‐quality plots for the final research paper.
All final figures are stored in the "figures/" directory.
Each plot is wrapped in a try‐except block to avoid one failure halting execution.
Plots are produced using only the numerical data stored in .npy files from the experiment summaries.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Increase font size for publication-quality figures
plt.rcParams.update({'font.size': 14})

# Create the final output directory for figures
os.makedirs("figures", exist_ok=True)

def load_experiment_data(filepath):
    """Load npy experiment data from given filepath."""
    try:
        data = np.load(filepath, allow_pickle=True).item()
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

###########################################
# 1. BASELINE EXPERIMENT (Synthetic SPR)
###########################################
baseline_path = "experiment_results/experiment_8b00f511069d44cf94230142a48622de_proc_1726491/experiment_data.npy"
baseline_data = load_experiment_data(baseline_path)
if baseline_data is not None:
    # Plot 1: Training and Validation Loss Curves
    try:
        plt.figure()
        for exp_key, exp_dict in baseline_data["num_epochs"].items():
            # Training loss curves (dashed)
            train = exp_dict["losses"]["train"]
            if train:
                epochs, losses = zip(*train)
                plt.plot(epochs, losses, "--", label=f"Train ({exp_key})")
            # Validation loss curves (solid)
            val = exp_dict["losses"]["val"]
            if val:
                epochs, losses = zip(*val)
                plt.plot(epochs, losses, "-", label=f"Validation ({exp_key})")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Baseline: Training vs Validation Loss (Synthetic SPR)")
        plt.legend()
        plt.savefig(os.path.join("figures", "baseline_loss_curves.png"), dpi=300)
        plt.close()
    except Exception as e:
        print("Error in Baseline loss plot:", e)
    
    # Plot 2: Validation Colour-Shape Harmonic Mean (CSHM)
    try:
        plt.figure()
        for exp_key, exp_dict in baseline_data["num_epochs"].items():
            val_metrics = exp_dict["metrics"]["val"]  # list of (epoch, cwa, swa, cshm)
            if val_metrics:
                epochs = [t[0] for t in val_metrics]
                cshm = [t[3] for t in val_metrics]
                plt.plot(epochs, cshm, label=f"CSHM ({exp_key})")
        plt.xlabel("Epoch")
        plt.ylabel("CSHM")
        plt.title("Baseline: Validation CSHM (Synthetic SPR)")
        plt.legend()
        plt.savefig(os.path.join("figures", "baseline_validation_CSHM.png"), dpi=300)
        plt.close()
    except Exception as e:
        print("Error in Baseline CSHM plot:", e)

###########################################
# 2. RESEARCH EXPERIMENT
###########################################
research_path = "experiment_results/experiment_8b3dc021a58644a09d3734b8da7290ae_proc_1733408/experiment_data.npy"
research_data = load_experiment_data(research_path)
if research_data is not None:
    # The key is expected to be 'SPR'
    ds_key = next(iter(research_data.keys()))
    # Plot 1: Loss Curves (Train vs Validation)
    try:
        plt.figure()
        tr = research_data[ds_key]["losses"]["train"]
        vl = research_data[ds_key]["losses"]["val"]
        if tr:
            e, l = zip(*tr)
            plt.plot(e, l, "--", label="Train")
        if vl:
            e, l = zip(*vl)
            plt.plot(e, l, "-", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{ds_key}: Training and Validation Loss Curves")
        plt.legend()
        plt.savefig(os.path.join("figures", f"{ds_key}_loss_curves.png"), dpi=300)
        plt.close()
    except Exception as e:
        print("Error in Research loss curves:", e)
    
    # Plot 2: Validation Metrics (CWA, SWA, HM, OCGA)
    try:
        plt.figure()
        metrics = research_data[ds_key]["metrics"]["val"]  # (epoch, cwa, swa, hm, ocga)
        if metrics:
            ep = [t[0] for t in metrics]
            cwa = [t[1] for t in metrics]
            swa = [t[2] for t in metrics]
            hm = [t[3] for t in metrics]
            ocga = [t[4] for t in metrics]
            plt.plot(ep, cwa, label="CWA")
            plt.plot(ep, swa, label="SWA")
            plt.plot(ep, hm, label="HM")
            plt.plot(ep, ocga, label="OCGA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(f"{ds_key}: Validation Metrics")
        plt.legend()
        plt.savefig(os.path.join("figures", f"{ds_key}_validation_metrics.png"), dpi=300)
        plt.close()
    except Exception as e:
        print("Error in Research validation metrics:", e)
    
    # Plot 3: HM with Best Epoch Marker
    try:
        plt.figure()
        metrics = research_data[ds_key]["metrics"]["val"]
        if metrics:
            ep = np.array([t[0] for t in metrics])
            hm_arr = np.array([t[3] for t in metrics])
            plt.plot(ep, hm_arr, label="HM")
            best_idx = hm_arr.argmax()
            plt.scatter(ep[best_idx], hm_arr[best_idx], color="red", zorder=5, label=f"Best @ {int(ep[best_idx])}")
        plt.xlabel("Epoch")
        plt.ylabel("Harmonic Mean (HM)")
        plt.title(f"{ds_key}: Best Epoch via HM")
        plt.legend()
        plt.savefig(os.path.join("figures", f"{ds_key}_validation_HM.png"), dpi=300)
        plt.close()
    except Exception as e:
        print("Error in Research HM plot:", e)
    
    # Plot 4: Confusion Matrix (Test Set)
    try:
        plt.figure()
        y_true = np.array(research_data[ds_key]["ground_truth"])
        y_pred = np.array(research_data[ds_key]["predictions"])
        if y_true.size == 0 or y_true.size != y_pred.size:
            raise ValueError("Ground truth and predictions are missing or mismatched.")
        n_cls = int(max(y_true.max(), y_pred.max())) + 1
        cm = np.zeros((n_cls, n_cls), int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{ds_key}: Test Confusion Matrix")
        for i in range(n_cls):
            for j in range(n_cls):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=12)
        plt.savefig(os.path.join("figures", f"{ds_key}_confusion_matrix.png"), dpi=300)
        plt.close()
    except Exception as e:
        print("Error in Research confusion matrix:", e)

###########################################
# 3. ABLATION EXPERIMENTS
###########################################

# (a) Remove Shape-Color Variety Features (No-SHC)
nshc_path = "experiment_results/experiment_ee41de1d31854e20a7184174fcd5cfce_proc_1749405/experiment_data.npy"
nshc_data = load_experiment_data(nshc_path)
if nshc_data and "No-SHC" in nshc_data:
    spr_nshc = nshc_data["No-SHC"].get("SPR", {})
    # Plot: Loss Curves
    try:
        tr = np.array(spr_nshc.get("losses", {}).get("train", []))
        va = np.array(spr_nshc.get("losses", {}).get("val", []))
        if tr.size and va.size:
            plt.figure()
            plt.plot(tr[:, 0], tr[:, 1], label="Train")
            plt.plot(va[:, 0], va[:, 1], label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("No-SHC: SPR Loss Curves")
            plt.legend()
            plt.savefig(os.path.join("figures", "NoSHC_SPR_loss_curves.png"), dpi=300)
            plt.close()
    except Exception as e:
        print("Error in No-SHC loss curves:", e)
    # Plot: Validation Metrics
    try:
        mv = np.array(spr_nshc.get("metrics", {}).get("val", []))
        if mv.size:
            epochs, cwa, swa, hm, ocga = mv.T
            plt.figure()
            plt.plot(epochs, cwa, label="CWA")
            plt.plot(epochs, swa, label="SWA")
            plt.plot(epochs, hm, label="HMean")
            plt.plot(epochs, ocga, label="OCGA")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title("No-SHC: SPR Validation Metrics")
            plt.legend()
            plt.savefig(os.path.join("figures", "NoSHC_SPR_metric_curves.png"), dpi=300)
            plt.close()
    except Exception as e:
        print("Error in No-SHC validation metrics:", e)

# (b) No-KMeans-RawGlyphIDs
nokmeans_path = "experiment_results/experiment_21021ca8d7f44b5d83bcab53dc0355b6_proc_1749406/experiment_data.npy"
nokmeans_data = load_experiment_data(nokmeans_path)
if nokmeans_data and "NoKMeansRawGlyphIDs" in nokmeans_data:
    spr_nok = nokmeans_data["NoKMeansRawGlyphIDs"].get("SPR", {})
    def to_xy(lst):
        arr = np.array(lst)
        if arr.ndim > 1:
            return arr[:, 0], arr[:, 1:]
        else:
            return None, None
    # Plot: Loss Curves
    try:
        epochs_tr, losses_tr = to_xy(spr_nok.get("losses", {}).get("train", []))
        epochs_val, losses_val = to_xy(spr_nok.get("losses", {}).get("val", []))
        if epochs_tr is not None and epochs_val is not None:
            plt.figure()
            plt.plot(epochs_tr, losses_tr, label="Train")
            plt.plot(epochs_val, losses_val, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title("No-KMeansRawGlyphIDs: Loss Curves")
            plt.legend()
            plt.savefig(os.path.join("figures", "NoKMeans_loss_curves.png"), dpi=300)
            plt.close()
    except Exception as e:
        print("Error in No-KMeans loss curves:", e)
    # Plot: Validation Metrics (CWA, SWA, HM)
    try:
        epochs_m, metrics_arr = to_xy(spr_nok.get("metrics", {}).get("val", []))
        if epochs_m is not None:
            metrics_arr = np.array(metrics_arr)
            cwa = metrics_arr[:, 0]
            swa = metrics_arr[:, 1]
            hm_vals = metrics_arr[:, 2]
            plt.figure()
            plt.plot(epochs_m, cwa, label="CWA")
            plt.plot(epochs_m, swa, label="SWA")
            plt.plot(epochs_m, hm_vals, label="HM")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title("No-KMeansRawGlyphIDs: Validation Metrics")
            plt.legend()
            plt.savefig(os.path.join("figures", "NoKMeans_validation_metrics.png"), dpi=300)
            plt.close()
    except Exception as e:
        print("Error in No-KMeans validation metrics:", e)
    # Plot: OCGA Curve
    try:
        epochs_m, metrics_arr = to_xy(spr_nok.get("metrics", {}).get("val", []))
        if epochs_m is not None:
            metrics_arr = np.array(metrics_arr)
            if metrics_arr.shape[1] > 3:
                ocga_vals = metrics_arr[:, 3]
                plt.figure()
                plt.plot(epochs_m, ocga_vals, label="OCGA", color="purple")
                plt.xlabel("Epoch")
                plt.ylabel("OCGA")
                plt.title("No-KMeansRawGlyphIDs: OCGA over Epochs")
                plt.legend()
                plt.savefig(os.path.join("figures", "NoKMeans_OCGA.png"), dpi=300)
                plt.close()
    except Exception as e:
        print("Error in No-KMeans OCGA curve:", e)
    # Plot: Confusion Matrix
    try:
        preds = np.array(spr_nok.get("predictions", []))
        gts = np.array(spr_nok.get("ground_truth", []))
        if preds.size and gts.size:
            num_cls = int(max(preds.max(), gts.max())) + 1
            cm = np.zeros((num_cls, num_cls), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            acc = np.mean(preds == gts)
            plt.title(f"No-KMeansRawGlyphIDs: Confusion Matrix (Acc={acc:.3f})")
            plt.savefig(os.path.join("figures", "NoKMeans_confusion_matrix.png"), dpi=300)
            plt.close()
    except Exception as e:
        print("Error in No-KMeans confusion matrix:", e)

# (c) UniGRU (No-Bidirectional Encoder)
uni_path = "experiment_results/experiment_616e3332414b4ad189c06832d9137a47_proc_1749408/experiment_data.npy"
uni_data = load_experiment_data(uni_path)
if uni_data and "UniGRU_no_bidi" in uni_data:
    run_uni = uni_data["UniGRU_no_bidi"].get("SPR", {})
    # Plot: Loss Curves
    try:
        loss_tr = run_uni.get("losses", {}).get("train", [])
        loss_val = run_uni.get("losses", {}).get("val", [])
        if loss_tr and loss_val:
            ep_tr, v_tr = zip(*loss_tr)
            ep_val, v_val = zip(*loss_val)
            plt.figure()
            plt.plot(ep_tr, v_tr, label="Train")
            plt.plot(ep_val, v_val, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title("UniGRU (No-Bidirectional): Loss Curves")
            plt.legend()
            plt.savefig(os.path.join("figures", "UniGRU_no_bidi_loss.png"), dpi=300)
            plt.close()
    except Exception as e:
        print("Error in UniGRU loss curves:", e)
    # Plot: Validation Metrics
    try:
        met_val = run_uni.get("metrics", {}).get("val", [])
        if met_val:
            ep, cwa, swa, hm_val, ocga = zip(*met_val)
            plt.figure()
            plt.plot(ep, cwa, label="CWA")
            plt.plot(ep, swa, label="SWA")
            plt.plot(ep, hm_val, label="HM")
            plt.plot(ep, ocga, label="OCGA")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title("UniGRU (No-Bidirectional): Validation Metrics")
            plt.legend()
            plt.savefig(os.path.join("figures", "UniGRU_no_bidi_metrics.png"), dpi=300)
            plt.close()
    except Exception as e:
        print("Error in UniGRU validation metrics:", e)
    # Plot: Confusion Matrix
    try:
        preds = np.array(run_uni.get("predictions", []))
        gts = np.array(run_uni.get("ground_truth", []))
        if preds.size and gts.size:
            n_cls = int(max(preds.max(), gts.max())) + 1
            cm = np.zeros((n_cls, n_cls), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title("UniGRU (No-Bidirectional): Confusion Matrix")
            plt.savefig(os.path.join("figures", "UniGRU_no_bidi_confusion.png"), dpi=300)
            plt.close()
    except Exception as e:
        print("Error in UniGRU confusion matrix:", e)
    # Plot: Class Distribution
    try:
        preds = np.array(run_uni.get("predictions", []))
        gts = np.array(run_uni.get("ground_truth", []))
        if preds.size and gts.size:
            uniq = np.arange(max(max(preds), max(gts)) + 1)
            pred_cnt = [np.sum(preds == u) for u in uniq]
            gt_cnt = [np.sum(gts == u) for u in uniq]
            x = np.arange(len(uniq))
            width = 0.35
            plt.figure()
            plt.bar(x - width/2, gt_cnt, width, label="Ground Truth")
            plt.bar(x + width/2, pred_cnt, width, label="Predicted")
            plt.xlabel("Class")
            plt.ylabel("Count")
            plt.title("UniGRU (No-Bidirectional): Class Distribution")
            plt.legend()
            plt.savefig(os.path.join("figures", "UniGRU_no_bidi_class_distribution.png"), dpi=300)
            plt.close()
    except Exception as e:
        print("Error in UniGRU class distribution plot:", e)

# (d) Frozen-Random-Embedding
frozen_path = "experiment_results/experiment_85d1a7147a1642cf9f5c50214282ac0f_proc_1749407/experiment_data.npy"
frozen_data = load_experiment_data(frozen_path)
if frozen_data and "FrozenRandomEmbedding" in frozen_data:
    run_frozen = frozen_data["FrozenRandomEmbedding"].get("SPR", {})
    # Plot: Loss Curves
    try:
        train_loss = np.array(run_frozen.get("losses", {}).get("train", []))
        val_loss = np.array(run_frozen.get("losses", {}).get("val", []))
        if train_loss.size and val_loss.size:
            plt.figure()
            plt.plot(train_loss[:, 0], train_loss[:, 1], label="Train")
            plt.plot(val_loss[:, 0], val_loss[:, 1], label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title("FrozenRandomEmbedding: Loss Curves")
            plt.legend()
            plt.savefig(os.path.join("figures", "FrozenRandomEmbedding_loss.png"), dpi=300)
            plt.close()
    except Exception as e:
        print("Error in FrozenRandomEmbedding loss curves:", e)
    # Plot: Validation Metrics
    try:
        val_metrics = np.array(run_frozen.get("metrics", {}).get("val", []))
        if val_metrics.size:
            epochs = val_metrics[:, 0]
            metric_labels = ["CWA", "SWA", "HM", "OCGA"]
            plt.figure()
            for i, lbl in enumerate(metric_labels, start=1):
                plt.plot(epochs, val_metrics[:, i], label=lbl)
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title("FrozenRandomEmbedding: Validation Metrics")
            plt.legend()
            plt.savefig(os.path.join("figures", "FrozenRandomEmbedding_metrics.png"), dpi=300)
            plt.close()
    except Exception as e:
        print("Error in FrozenRandomEmbedding metrics plot:", e)

# (e) MeanPool-Only (No-GRU)
meanpool_path = "experiment_results/experiment_0ed1656aa984417791f3d1463218ad1f_proc_1749405/experiment_data.npy"
meanpool_data = load_experiment_data(meanpool_path)
if meanpool_data and "MeanPoolOnly" in meanpool_data:
    spr_mean = meanpool_data["MeanPoolOnly"].get("SPR", {})
    # Plot: Loss Curves
    try:
        loss_tr = np.array(spr_mean.get("losses", {}).get("train", []))
        loss_val = np.array(spr_mean.get("losses", {}).get("val", []))
        if loss_tr.size and loss_val.size:
            plt.figure()
            plt.plot(loss_tr[:, 0], loss_tr[:, 1], label="Train")
            plt.plot(loss_val[:, 0], loss_val[:, 1], label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("MeanPoolOnly: Loss Curves")
            plt.legend()
            plt.savefig(os.path.join("figures", "MeanPoolOnly_loss_curves.png"), dpi=300)
            plt.close()
    except Exception as e:
        print("Error in MeanPoolOnly loss curves:", e)
    # Plot: Validation Metrics
    try:
        metrics_val = np.array(spr_mean.get("metrics", {}).get("val", []))
        if metrics_val.size:
            ep, cwa, swa, hm, ocga = metrics_val.T
            plt.figure()
            plt.plot(ep, cwa, label="CWA")
            plt.plot(ep, swa, label="SWA")
            plt.plot(ep, hm, label="HM")
            plt.plot(ep, ocga, label="OCGA")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title("MeanPoolOnly: Validation Metrics")
            plt.legend()
            plt.savefig(os.path.join("figures", "MeanPoolOnly_metrics.png"), dpi=300)
            plt.close()
    except Exception as e:
        print("Error in MeanPoolOnly metrics plot:", e)
    # Plot: Confusion Matrix
    try:
        preds = np.array(spr_mean.get("predictions", []))
        gts = np.array(spr_mean.get("ground_truth", []))
        if preds.size and gts.size:
            classes = np.unique(np.concatenate([preds, gts]))
            cm = np.zeros((len(classes), len(classes)), dtype=int)
            for t, p in zip(gts, preds):
                idx_t = np.where(classes == t)[0][0]
                idx_p = np.where(classes == p)[0][0]
                cm[idx_t, idx_p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xticks(range(len(classes)), classes)
            plt.yticks(range(len(classes)), classes)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title("MeanPoolOnly: Confusion Matrix")
            plt.savefig(os.path.join("figures", "MeanPoolOnly_confusion_matrix.png"), dpi=300)
            plt.close()
    except Exception as e:
        print("Error in MeanPoolOnly confusion matrix:", e)

# (f) Train-Only KMeans (No Cross-Split Leakage)
trainonly_path = "experiment_results/experiment_35836adbe0564235b47b5e6ad28970ff_proc_1749406/experiment_data.npy"
trainonly_data = load_experiment_data(trainonly_path)
if trainonly_data and "train_only_kmeans" in trainonly_data:
    spr_train = trainonly_data["train_only_kmeans"].get("SPR", {})
    # Plot: Loss Curves
    try:
        plt.figure()
        for split in ["train", "val"]:
            if spr_train.get("losses", {}).get(split):
                epochs, losses = zip(*spr_train["losses"][split])
                plt.plot(epochs, losses, label=f"{split.capitalize()} Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Train-Only KMeans: Loss Curves")
        plt.legend()
        plt.savefig(os.path.join("figures", "TrainOnlyKMeans_loss_curves.png"), dpi=300)
        plt.close()
    except Exception as e:
        print("Error in Train-Only KMeans loss curves:", e)
    # Plot: Validation Metrics
    try:
        if spr_train.get("metrics", {}).get("val"):
            epochs, cwa, swa, hm, ocga = zip(*spr_train["metrics"]["val"])
            plt.figure()
            plt.plot(epochs, cwa, label="CWA")
            plt.plot(epochs, swa, label="SWA")
            plt.plot(epochs, hm, label="HM")
            plt.plot(epochs, ocga, label="OCGA")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title("Train-Only KMeans: Validation Metrics")
            plt.legend()
            plt.savefig(os.path.join("figures", "TrainOnlyKMeans_validation_metrics.png"), dpi=300)
            plt.close()
    except Exception as e:
        print("Error in Train-Only KMeans validation metrics:", e)
    # Plot: Confusion Matrix
    try:
        preds = np.array(spr_train.get("predictions", []))
        gts = np.array(spr_train.get("ground_truth", []))
        if preds.size and gts.size:
            n_cls = int(max(preds.max(), gts.max())) + 1
            cm = np.zeros((n_cls, n_cls), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Train-Only KMeans: Confusion Matrix")
            plt.savefig(os.path.join("figures", "TrainOnlyKMeans_confusion_matrix.png"), dpi=300)
            plt.close()
    except Exception as e:
        print("Error in Train-Only KMeans confusion matrix:", e)

# (g) Random-Cluster-Assignment (No-KMeans Semantic Clustering)
randcluster_path = "experiment_results/experiment_799d1033f5874a45a66ca59eecfac90e_proc_1749408/experiment_data.npy"
randcluster_data = load_experiment_data(randcluster_path)
if randcluster_data and "RandomCluster" in randcluster_data:
    run_rand = randcluster_data["RandomCluster"].get("SPR", {})
    # Plot: Loss Curves
    try:
        tr_epochs, tr_loss = zip(*run_rand["losses"]["train"])
        va_epochs, va_loss = zip(*run_rand["losses"]["val"])
        plt.figure()
        plt.plot(tr_epochs, tr_loss, label="Train")
        plt.plot(va_epochs, va_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("RandomCluster: SPR Loss Curves")
        plt.legend()
        plt.savefig(os.path.join("figures", "RandomCluster_SPR_loss_curves.png"), dpi=300)
        plt.close()
    except Exception as e:
        print("Error in RandomCluster loss curves:", e)
    # Plot: Validation Metrics
    try:
        met = np.array(run_rand["metrics"]["val"])  # rows: (epoch, CWA, SWA, HM, OCGA)
        epochs, cwa, swa, hm_rand, ocga = met.T
        plt.figure()
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, hm_rand, label="HM")
        plt.plot(epochs, ocga, label="OCGA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("RandomCluster: SPR Validation Metrics")
        plt.legend()
        plt.savefig(os.path.join("figures", "RandomCluster_SPR_metric_curves.png"), dpi=300)
        plt.close()
    except Exception as e:
        print("Error in RandomCluster validation metrics:", e)
    # Plot: Confusion Matrix
    try:
        preds = np.array(run_rand.get("predictions", []))
        gts = np.array(run_rand.get("ground_truth", []))
        classes = np.unique(np.concatenate([gts, preds]))
        cm = np.zeros((classes.size, classes.size), dtype=int)
        for t, p in zip(gts, preds):
            idx_t = np.where(classes == t)[0][0]
            idx_p = np.where(classes == p)[0][0]
            cm[idx_t, idx_p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(len(classes)), classes)
        plt.yticks(range(len(classes)), classes)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("RandomCluster: SPR Confusion Matrix")
        plt.savefig(os.path.join("figures", "RandomCluster_SPR_confusion_matrix.png"), dpi=300)
        plt.close()
    except Exception as e:
        print("Error in RandomCluster confusion matrix:", e)

print("Final figures saved in the 'figures' directory.")
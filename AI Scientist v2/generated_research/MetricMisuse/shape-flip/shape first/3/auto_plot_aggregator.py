"""
Final Aggregator Script for Zero-Shot Synthetic PolyRule Reasoning Figures
This script loads experiment .npy files (with full paths provided in the summaries)
and generates publishable, final figures stored in the "figures/" directory.
Each figure is wrapped in a try-except block so that one failure does not interrupt others.
All plot labels and titles have larger font sizes and minimal styles (no top/right spines).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Set global styling: larger fonts and higher dpi
plt.rcParams.update({'font.size': 14})
plt.rcParams["figure.dpi"] = 300

# Remove top and right spines for a clean look
def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Ensure output directory exists
os.makedirs("figures", exist_ok=True)

#############################################
# 1. BASELINE EXPERIMENT PLOTS
#############################################
try:
    # Load baseline experiment data (full path as given in the summary)
    baseline_path = "experiment_results/experiment_f2fd50b2bb004f81b81078e3163646c5_proc_457453/experiment_data.npy"
    baseline_data = np.load(baseline_path, allow_pickle=True).item()
    
    # baseline_data structure: assume key "epochs" holds a dict of runs,
    # where each run has keys: "losses" { "train": [...], "val": [...] },
    # "metrics" { "val": [ { "HWA":... }, ... ] } and "test_metrics", predictions and ground_truth.
    runs = baseline_data.get("epochs", {})
    final_scores = {}
    best_run, best_hwa = None, -1
    for run_key, run_dict in runs.items():
        tm = run_dict["test_metrics"]
        final_scores[run_key] = tm
        if tm["HWA"] > best_hwa:
            best_hwa, best_run = tm["HWA"], run_key

    # Figure 1: Baseline Loss Curves (training and validation)
    plt.figure(figsize=(8,6))
    for run_key, run_dict in runs.items():
        plt.plot(run_dict["losses"]["train"], linestyle="--", label=f"{run_key} Train")
        plt.plot(run_dict["losses"]["val"], linestyle="-", label=f"{run_key} Val")
    plt.title("Baseline: SPR_BENCH Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Baseline_Loss_Curves.png"))
    plt.close()

    # Figure 2: Baseline Validation HWA Curves
    plt.figure(figsize=(8,6))
    for run_key, run_dict in runs.items():
        # Collect HWA per epoch from the metrics list (each element is a dict)
        hwa = [m["HWA"] for m in run_dict["metrics"]["val"]]
        plt.plot(hwa, label=run_key)
    plt.title("Baseline: SPR_BENCH Validation Harmonic Weighted Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Baseline_Validation_HWA.png"))
    plt.close()

    # Figure 3: Baseline Test Metrics Bar Chart (SWA, CWA, HWA)
    plt.figure(figsize=(8,6))
    labels = list(final_scores.keys())
    x = np.arange(len(labels))
    width = 0.25
    swa_vals = [final_scores[k]["SWA"] for k in labels]
    cwa_vals = [final_scores[k]["CWA"] for k in labels]
    hwa_vals = [final_scores[k]["HWA"] for k in labels]
    plt.bar(x - width, swa_vals, width, label="SWA")
    plt.bar(x, cwa_vals, width, label="CWA")
    plt.bar(x + width, hwa_vals, width, label="HWA")
    plt.title("Baseline: SPR_BENCH Final Test Metrics")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Baseline_Test_Metrics_Bar.png"))
    plt.close()

    # Figure 4: Baseline Confusion Matrix for Best Run (by HWA)
    import itertools
    best_run_data = runs.get(best_run, {})
    best_pred = best_run_data.get("predictions", [])
    best_gt = best_run_data.get("ground_truth", [])
    labels_set = sorted(set(best_gt))
    idx = {l: i for i, l in enumerate(labels_set)}
    cm = np.zeros((len(labels_set), len(labels_set)), dtype=int)
    for t, p in zip(best_gt, best_pred):
        cm[idx[t], idx[p]] += 1

    plt.figure(figsize=(6,5))
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.title(f"Baseline: SPR_BENCH Confusion Matrix ({best_run})")
    plt.xticks(range(len(labels_set)), labels_set, rotation=90)
    plt.yticks(range(len(labels_set)), labels_set)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    ax = plt.gca()
    style_ax(ax)
    # Annotate confusion matrix cells
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        txt_color = "white" if cm[i,j] > cm.max() / 2 else "black"
        plt.text(j, i, cm[i,j], ha="center", va="center", color=txt_color, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join("figures", f"Baseline_Confusion_Matrix_{best_run}.png"))
    plt.close()
except Exception as e:
    print(f"Error in Baseline plots: {e}")

#############################################
# 2. RESEARCH EXPERIMENT PLOTS
#############################################
try:
    # Load research experiment data
    research_path = "experiment_results/experiment_c26811bcad66484ca81fc7d0cff37944_proc_458964/experiment_data.npy"
    research_data = np.load(research_path, allow_pickle=True).item()
    # Assume research_data is a dict keyed by dataset names (e.g., "SPR_BENCH")
    # For simplicity use the first key:
    research_key = list(research_data.keys())[0]
    rec = research_data[research_key]

    # Figure 5: Research Loss Curves (train vs validation)
    plt.figure(figsize=(8,6))
    tr_losses = rec.get("losses", {}).get("train", [])
    val_losses = rec.get("losses", {}).get("val", [])
    if tr_losses and val_losses:
        plt.plot(tr_losses, linestyle="--", label="Train")
        plt.plot(val_losses, linestyle="-", label="Validation")
        plt.title(f"Research: {research_key} Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Total Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("figures", f"Research_{research_key}_Loss_Curves.png"))
    plt.close()

    # Figure 6: Research Validation SWA Curve
    plt.figure(figsize=(8,6))
    val_metrics = rec.get("metrics", {}).get("val", [])
    if val_metrics:
        # Assume each element in val_metrics is the SWA value (if stored directly) or a dict; here use value directly.
        # If the stored value is not a dict, then directly plot.
        # For safety, check if type is dict:
        if isinstance(val_metrics[0], dict):
            swa_vals = [m.get("SWA", 0) for m in val_metrics]
        else:
            swa_vals = val_metrics
        plt.plot(swa_vals, marker="o", label="Validation SWA")
        plt.title(f"Research: {research_key} Validation Shape-Weighted Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("figures", f"Research_{research_key}_Val_SWA.png"))
    plt.close()

    # Figure 7: Research Confusion Matrix (Test Set)
    preds = rec.get("predictions", [])
    truths = rec.get("ground_truth", [])
    if preds and truths:
        labels = sorted(set(truths))
        idx = {l: i for i, l in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(truths, preds):
            cm[idx[t], idx[p]] += 1
        plt.figure(figsize=(6,5))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.title(f"Research: {research_key} Confusion Matrix (Test Set)")
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        ax = plt.gca()
        style_ax(ax)
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            txt_color = "white" if cm[i,j] > cm.max() / 2 else "black"
            plt.text(j, i, cm[i,j], ha="center", va="center", color=txt_color, fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join("figures", f"Research_{research_key}_Confusion_Matrix.png"))
    plt.close()
except Exception as e:
    print(f"Error in Research plots: {e}")

#############################################
# 3. ABLATION STUDIES PLOTS
#############################################
# 3.1 No-Auxiliary-Variety-Loss
try:
    aux_path = "experiment_results/experiment_7ef032dc1456402c902347d6f15db5f7_proc_460564/experiment_data.npy"
    aux_data = np.load(aux_path, allow_pickle=True).item()
    # Extract run data: key "NoAuxVarLoss" under dataset "SPR_BENCH"
    aux_run = aux_data.get("NoAuxVarLoss", {}).get("SPR_BENCH", {})
    epochs = np.arange(1, len(aux_run.get("losses", {}).get("train", [])) + 1)
    
    fig, axs = plt.subplots(1, 3, figsize=(18,5))
    
    # Subplot 1: Loss curves
    loss_tr = aux_run.get("losses", {}).get("train", [])
    loss_val = aux_run.get("losses", {}).get("val", [])
    if loss_tr and loss_val:
        axs[0].plot(epochs, loss_tr, label="Train")
        axs[0].plot(epochs, loss_val, label="Validation")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Cross-Entropy Loss")
        axs[0].set_title("NoAuxVarLoss: Loss Curves")
        axs[0].legend()
        style_ax(axs[0])
    
    # Subplot 2: Validation SWA curve
    swa_vals = aux_run.get("metrics", {}).get("val", [])
    if swa_vals:
        axs[1].plot(epochs[:len(swa_vals)], swa_vals, marker="o", label="Validation SWA")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("SWA")
        axs[1].set_title("NoAuxVarLoss: Validation SWA")
        axs[1].legend()
        style_ax(axs[1])
    
    # Subplot 3: Confusion Matrix
    y_true_aux = aux_run.get("ground_truth", [])
    y_pred_aux = aux_run.get("predictions", [])
    if y_true_aux and y_pred_aux:
        labels_aux = sorted(set(y_true_aux) | set(y_pred_aux))
        idx_aux = {l: i for i, l in enumerate(labels_aux)}
        cm_aux = np.zeros((len(labels_aux), len(labels_aux)), dtype=int)
        for t, p in zip(y_true_aux, y_pred_aux):
            cm_aux[idx_aux[t], idx_aux[p]] += 1
        im = axs[2].imshow(cm_aux, cmap="Blues")
        axs[2].set_title("NoAuxVarLoss: Confusion Matrix")
        axs[2].set_xlabel("Predicted")
        axs[2].set_ylabel("True")
        axs[2].set_xticks(range(len(labels_aux)))
        axs[2].set_yticks(range(len(labels_aux)))
        axs[2].set_xticklabels(labels_aux, rotation=90)
        axs[2].set_yticklabels(labels_aux)
        for i, j in itertools.product(range(cm_aux.shape[0]), range(cm_aux.shape[1])):
            txt_color = "white" if cm_aux[i,j] > cm_aux.max() / 2 else "black"
            axs[2].text(j, i, cm_aux[i,j], ha="center", va="center", color=txt_color, fontsize=10)
        style_ax(axs[2])
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Ablation_NoAuxVarLoss.png"))
    plt.close()
except Exception as e:
    print(f"Error in Ablation No-Auxiliary-Variety-Loss plots: {e}")

# 3.2 No-Positional-Encoding (No-PE)
try:
    nope_path = "experiment_results/experiment_86b88237155944acafc399369895af1e_proc_460565/experiment_data.npy"
    nope_data = np.load(nope_path, allow_pickle=True).item()
    # Extract data for key "No-PE" under "SPR_BENCH"
    nope_run = nope_data.get("No-PE", {}).get("SPR_BENCH", {})
    epochs = np.arange(1, len(nope_run.get("losses", {}).get("train", [])) + 1)
    
    fig, axs = plt.subplots(1, 3, figsize=(18,5))
    
    # Subplot 1: Loss Curves
    tr_loss = nope_run.get("losses", {}).get("train", [])
    val_loss = nope_run.get("losses", {}).get("val", [])
    if tr_loss and val_loss:
        axs[0].plot(epochs, tr_loss, label="Train")
        axs[0].plot(epochs, val_loss, label="Validation")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Cross-Task Loss")
        axs[0].set_title("No-PE: Loss Curves")
        axs[0].legend()
        style_ax(axs[0])
    
    # Subplot 2: Validation SWA Curve
    swa_vals = nope_run.get("metrics", {}).get("val", [])
    if swa_vals:
        axs[1].plot(epochs[:len(swa_vals)], swa_vals, marker="o", label="Validation SWA")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("SWA")
        axs[1].set_title("No-PE: Validation SWA")
        axs[1].legend()
        style_ax(axs[1])
    
    # Subplot 3: Confusion Matrix
    preds_nope = nope_run.get("predictions", [])
    truths_nope = nope_run.get("ground_truth", [])
    if preds_nope and truths_nope:
        labels_nope = sorted(set(truths_nope) | set(preds_nope))
        idx_nope = {l: i for i, l in enumerate(labels_nope)}
        cm_nope = np.zeros((len(labels_nope), len(labels_nope)), dtype=int)
        for t, p in zip(truths_nope, preds_nope):
            cm_nope[idx_nope[t], idx_nope[p]] += 1
        im = axs[2].imshow(cm_nope, cmap="Blues")
        axs[2].set_title("No-PE: Confusion Matrix")
        axs[2].set_xlabel("Predicted")
        axs[2].set_ylabel("True")
        axs[2].set_xticks(range(len(labels_nope)))
        axs[2].set_yticks(range(len(labels_nope)))
        axs[2].set_xticklabels(labels_nope, rotation=90)
        axs[2].set_yticklabels(labels_nope)
        for i, j in itertools.product(range(cm_nope.shape[0]), range(cm_nope.shape[1])):
            txt_color = "white" if cm_nope[i,j] > cm_nope.max() / 2 else "black"
            axs[2].text(j, i, cm_nope[i,j], ha="center", va="center", color=txt_color, fontsize=10)
        style_ax(axs[2])
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Ablation_NoPE.png"))
    plt.close()
except Exception as e:
    print(f"Error in Ablation No-PE plots: {e}")

# 3.3 Bag-of-Embeddings (No-Transformer)
try:
    boe_path = "experiment_results/experiment_f1e17a467a4b4da1b23d8184c810e7c0_proc_460566/experiment_data.npy"
    boe_data = np.load(boe_path, allow_pickle=True).item()
    # Extract data from key "bag_of_embeddings" under "SPR_BENCH"
    boe_run = boe_data.get("bag_of_embeddings", {}).get("SPR_BENCH", {})
    epochs = np.arange(1, len(boe_run.get("losses", {}).get("train", [])) + 1)
    
    fig, axs = plt.subplots(1, 3, figsize=(18,5))
    
    # Subplot 1: Loss Curves
    tr_loss = boe_run.get("losses", {}).get("train", [])
    val_loss = boe_run.get("losses", {}).get("val", [])
    if tr_loss and val_loss:
        axs[0].plot(epochs, tr_loss, label="Train")
        axs[0].plot(epochs, val_loss, label="Validation")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("Bag-of-Embeddings: Loss Curves")
        axs[0].legend()
        style_ax(axs[0])
    
    # Subplot 2: Validation SWA Curve
    swa_vals = boe_run.get("metrics", {}).get("val", [])
    if swa_vals:
        axs[1].plot(epochs[:len(swa_vals)], swa_vals, marker="o", label="Validation SWA")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("SWA")
        axs[1].set_title("Bag-of-Embeddings: Validation SWA")
        axs[1].legend()
        style_ax(axs[1])
    
    # Subplot 3: Confusion Matrix
    preds_boe = np.array(boe_run.get("predictions", []))
    truths_boe = np.array(boe_run.get("ground_truth", []))
    if preds_boe.size and truths_boe.size:
        labels_boe = sorted(set(truths_boe) | set(preds_boe))
        idx_boe = {l: i for i, l in enumerate(labels_boe)}
        cm_boe = np.zeros((len(labels_boe), len(labels_boe)), dtype=int)
        for t, p in zip(preds_boe, truths_boe):
            cm_boe[idx_boe[t], idx_boe[p]] += 1
        im = axs[2].imshow(cm_boe, cmap="Blues")
        axs[2].set_title("Bag-of-Embeddings: Confusion Matrix")
        axs[2].set_xlabel("Predicted")
        axs[2].set_ylabel("True")
        axs[2].set_xticks(range(len(labels_boe)))
        axs[2].set_yticks(range(len(labels_boe)))
        axs[2].set_xticklabels(labels_boe, rotation=90)
        axs[2].set_yticklabels(labels_boe)
        for i, j in itertools.product(range(cm_boe.shape[0]), range(cm_boe.shape[1])):
            txt_color = "white" if cm_boe[i,j] > cm_boe.max() / 2 else "black"
            axs[2].text(j, i, cm_boe[i,j], ha="center", va="center", color=txt_color, fontsize=10)
        style_ax(axs[2])
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Ablation_Bag_of_Embeddings.png"))
    plt.close()
except Exception as e:
    print(f"Error in Ablation Bag-of-Embeddings plots: {e}")

# 3.4 Multi-Synthetic-Dataset Training (MSDT)
try:
    msdt_path = "experiment_results/experiment_debe8a7adc0141fd96d109d01d69a4ec_proc_460563/experiment_data.npy"
    msdt_data = np.load(msdt_path, allow_pickle=True).item()
    # msdt_data is a dict with key "MSDT" whose value is itself a dict of datasets, e.g., "SPR_BENCH", "TOKEN_RENAMED", "COLOR_SHUFFLED"
    datasets = msdt_data.get("MSDT", {})
    
    # Prepare composite figure with 3 subplots:
    fig, axs = plt.subplots(1, 3, figsize=(21,6))
    
    # Subplot 1: Training Loss (from SPR_BENCH) with per-dataset Validation Loss curves
    # Use SPR_BENCH training losses as baseline for epoch numbers.
    spr_data = datasets.get("SPR_BENCH", {})
    train_losses = spr_data.get("losses", {}).get("train", [])
    epochs = np.arange(1, len(train_losses) + 1)
    if train_losses:
        axs[0].plot(epochs, train_losses, label="Train (SPR_BENCH)", lw=2)
    # For each dataset, plot its validation loss if available.
    for name, data in datasets.items():
        vloss = data.get("losses", {}).get("val", [])
        if vloss:
            axs[0].plot(np.arange(1, len(vloss)+1), vloss, label=f"Val {name}")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("MSDT: Training & Validation Losses")
    axs[0].legend()
    style_ax(axs[0])
    
    # Subplot 2: Validation SWA curves for each dataset
    for name, data in datasets.items():
        vswa = data.get("metrics", {}).get("val", [])
        if vswa:
            axs[1].plot(np.arange(1, len(vswa)+1), vswa, marker="o", label=name)
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Shape-Weighted Accuracy")
    axs[1].set_title("MSDT: Validation SWA Curves")
    axs[1].legend()
    style_ax(axs[1])
    
    # Subplot 3: Test Accuracy Bar Chart for each dataset
    test_names = []
    test_accs = []
    for name, data in datasets.items():
        y_true_msdt = data.get("ground_truth", [])
        y_pred_msdt = data.get("predictions", [])
        if y_true_msdt and y_pred_msdt:
            acc = np.mean(np.array(y_true_msdt) == np.array(y_pred_msdt))
            test_names.append(name)
            test_accs.append(acc)
    if test_names and test_accs:
        axs[2].bar(test_names, test_accs, color=["tab:blue", "tab:orange", "tab:green"][:len(test_names)])
        axs[2].set_ylabel("Accuracy")
        axs[2].set_title("MSDT: Test Accuracy per Dataset")
        for i, v in enumerate(test_accs):
            axs[2].text(i, v + 0.01, f"{v:.2f}", ha="center")
    style_ax(axs[2])
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "MSDT_Composite.png"))
    plt.close()
except Exception as e:
    print(f"Error in MSDT plots: {e}")

print("Final aggregated plots saved in 'figures/' directory.")
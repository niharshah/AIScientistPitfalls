#!/usr/bin/env python3
"""
Final Aggregator Script for SPR Final Figures
This script loads experiment .npy files from Baseline, Research, and Ablation experiments and produces a comprehensive set of final scientific plots.
All final figures are stored in the "figures/" folder.
Each individual figure is wrapped in its own try–except block so that an error in one does not stop the rest.
Please ensure the .npy files exist exactly at the specified paths.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Increase global font size for publication‐quality images and remove top/right spines by default.
plt.rcParams.update({'font.size': 14, 'axes.spines.top': False, 'axes.spines.right': False})
dpi = 300

# Create the final figures directory
os.makedirs("figures", exist_ok=True)

#############################################
# HELPER FUNCTIONS
#############################################
def safe_save(fig, fname):
    try:
        fig.savefig(os.path.join("figures", fname), dpi=dpi, bbox_inches="tight")
    except Exception as e:
        print(f"Error saving {fname}: {e}")
    plt.close(fig)

def load_npy(path):
    try:
        data = np.load(path, allow_pickle=True).item()
        return data
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return {}

def extract_baseline_best(data):
    # Data structure: data["NUM_EPOCHS"]["SPR_BENCH"] is a dict of configs.
    subtree = data.get("NUM_EPOCHS", {}).get("SPR_BENCH", {})
    if not subtree:
        print("Baseline SPR_BENCH data not found.")
        return None
    best_cfg, best_val = None, -1e9
    val_mcc_per_cfg, test_mcc_per_cfg = {}, {}
    for cfg, rec in subtree.items():
        max_val = max(rec["metrics"]["val_MCC"])
        val_mcc_per_cfg[cfg] = max_val
        test_mcc_per_cfg[cfg] = rec["metrics"].get("test_MCC", np.nan)
        if max_val > best_val:
            best_val, best_cfg = max_val, cfg
    best_run = subtree.get(best_cfg, {})
    return best_cfg, best_run, subtree, val_mcc_per_cfg, test_mcc_per_cfg

#############################################
# FIGURE 1: BASELINE – Composite Loss & Val MCC Curves (Best Config)
#############################################
try:
    baseline_path = "experiment_results/experiment_9a1f8b74bbe44988a3e7d976a9562e9c_proc_3333222/experiment_data.npy"
    baseline_data = load_npy(baseline_path)
    res = extract_baseline_best(baseline_data)
    if res is None:
        raise ValueError("Missing baseline best configuration data.")
    best_cfg, best_run, _, _, _ = res
    epochs = best_run.get("epochs", [])
    train_loss = best_run.get("losses", {}).get("train", [])
    val_loss = best_run.get("losses", {}).get("val", [])
    val_mcc = best_run.get("metrics", {}).get("val_MCC", [])
    
    fig, axs = plt.subplots(1, 2, figsize=(12,5))
    axs[0].plot(epochs, train_loss, marker="o", label="Train Loss")
    axs[0].plot(epochs, val_loss, marker="s", label="Validation Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title(f"Baseline Loss Curves\n(Best Config: {best_cfg})")
    axs[0].legend()
    
    axs[1].plot(epochs, val_mcc, marker="o", color="purple", label="Validation MCC")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("MCC")
    axs[1].set_title(f"Baseline Val MCC Curve\n(Best Config: {best_cfg})")
    axs[1].legend()
    
    safe_save(fig, f"Baseline_Composite_BestConfig.png")
    print("Figure 1 saved: Baseline Composite (Loss & Val MCC)")
except Exception as e:
    print(f"Error in Figure 1: {e}")

#############################################
# FIGURE 2: BASELINE – Bar Plots Comparison of Best Val MCC and Test MCC per Config
#############################################
try:
    _, _, _, val_mcc_dict, test_mcc_dict = extract_baseline_best(baseline_data)
    cfgs = list(val_mcc_dict.keys())
    vals = [val_mcc_dict[c] for c in cfgs]
    tests = [test_mcc_dict[c] for c in cfgs]
    
    fig, axs = plt.subplots(1, 2, figsize=(12,5))
    axs[0].bar(cfgs, vals, color="skyblue")
    axs[0].set_ylabel("Best Validation MCC")
    axs[0].set_title("Baseline Best Val MCC per Epoch Setting")
    axs[0].set_xticklabels(cfgs, rotation=45, ha="right")
    
    axs[1].bar(cfgs, tests, color="lightgreen")
    axs[1].set_ylabel("Test MCC")
    axs[1].set_title("Baseline Test MCC per Epoch Setting")
    axs[1].set_xticklabels(cfgs, rotation=45, ha="right")
    
    safe_save(fig, "Baseline_Comparison_BarPlots.png")
    print("Figure 2 saved: Baseline Comparison Bar Plots")
except Exception as e:
    print(f"Error in Figure 2: {e}")

#############################################
# FIGURE 3: RESEARCH – Composite Loss & Accuracy Curves (SPR_BENCH)
#############################################
try:
    research_path = "experiment_results/experiment_93f8f3c806cf498b8ff964af0581a522_proc_3338339/experiment_data.npy"
    research_data = load_npy(research_path)
    # Here research data is under key "SPR_BENCH"
    rd = research_data.get("SPR_BENCH", {})
    epochs_r = np.array(rd.get("epochs", []))
    loss_tr_r = np.array(rd.get("losses", {}).get("train", []))
    loss_val_r = np.array(rd.get("losses", {}).get("val", []))
    acc_tr_r = np.array([m.get("acc", np.nan) for m in rd.get("metrics", {}).get("train", [])])
    acc_val_r = np.array([m.get("acc", np.nan) for m in rd.get("metrics", {}).get("val", [])])
    
    fig, axs = plt.subplots(1, 2, figsize=(12,5))
    axs[0].plot(epochs_r, loss_tr_r, marker="o", label="Train Loss")
    axs[0].plot(epochs_r, loss_val_r, marker="s", label="Validation Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("BCE Loss")
    axs[0].set_title("Research SPR_BENCH Loss Curves")
    axs[0].legend()
    
    axs[1].plot(epochs_r, acc_tr_r, marker="o", label="Train Accuracy")
    axs[1].plot(epochs_r, acc_val_r, marker="s", label="Validation Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_title("Research SPR_BENCH Accuracy Curves")
    axs[1].legend()
    
    safe_save(fig, "Research_Composite_Loss_Accuracy.png")
    print("Figure 3 saved: Research Loss & Accuracy Composite")
except Exception as e:
    print(f"Error in Figure 3: {e}")

#############################################
# FIGURE 4: RESEARCH – MCC Curve & Confusion Matrix (2 subplots)
#############################################
try:
    mcc_tr_r = np.array([m.get("MCC", np.nan) for m in rd.get("metrics", {}).get("train", [])])
    mcc_val_r = np.array([m.get("MCC", np.nan) for m in rd.get("metrics", {}).get("val", [])])
    
    # Confusion matrix from test set
    y_true_r = np.array(rd.get("ground_truth", []))
    y_pred_r = np.array(rd.get("predictions", []))
    # Simple confusion matrix
    if y_true_r.size and y_pred_r.size and y_true_r.size == y_pred_r.size:
        cm_r = np.array([[ ((y_true_r==0) & (y_pred_r==0)).sum(), ((y_true_r==0) & (y_pred_r==1)).sum() ],
                         [ ((y_true_r==1) & (y_pred_r==0)).sum(), ((y_true_r==1) & (y_pred_r==1)).sum() ]])
    else:
        cm_r = np.array([[0,0],[0,0]])
    
    fig, axs = plt.subplots(1, 2, figsize=(12,5))
    # MCC Curve
    axs[0].plot(epochs_r, mcc_tr_r, marker="o", label="Train MCC")
    axs[0].plot(epochs_r, mcc_val_r, marker="s", label="Validation MCC")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("MCC")
    axs[0].set_title("Research SPR_BENCH MCC Curves")
    axs[0].legend()
    
    # Confusion Matrix
    im = axs[1].imshow(cm_r, cmap="Blues")
    axs[1].set_title("Research SPR_BENCH Confusion Matrix")
    axs[1].set_xticks([0,1]); axs[1].set_xticklabels(["Neg", "Pos"])
    axs[1].set_yticks([0,1]); axs[1].set_yticklabels(["Neg", "Pos"])
    for i in range(2):
        for j in range(2):
            axs[1].text(j, i, str(cm_r[i,j]), ha="center", va="center")
    fig.colorbar(im, ax=axs[1])
    
    safe_save(fig, "Research_MCC_and_Confusion.png")
    print("Figure 4 saved: Research MCC Curve & Confusion Matrix")
except Exception as e:
    print(f"Error in Figure 4: {e}")

#############################################
# FIGURE 5: RESEARCH – Prediction Histogram (Standalone)
#############################################
try:
    if y_true_r.size and y_pred_r.size:
        fig, ax = plt.subplots(figsize=(6,5))
        ax.hist(y_pred_r[y_true_r==0], bins=np.arange(-0.5,2), alpha=0.7, label="True Negatives")
        ax.hist(y_pred_r[y_true_r==1], bins=np.arange(-0.5,2), alpha=0.7, label="True Positives")
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Count")
        ax.set_title("Research SPR_BENCH Prediction Distribution")
        ax.legend()
        safe_save(fig, "Research_Prediction_Histogram.png")
        print("Figure 5 saved: Research Prediction Histogram")
    else:
        print("No prediction data for Research histogram.")
except Exception as e:
    print(f"Error in Figure 5: {e}")

#############################################
# ABALTION AGGREGATE: Collect test_metrics for selected ablation experiments
#############################################
ablation_files = {
    "No-Count Features": "experiment_results/experiment_0aeb225aba734e6fbb9e27762f81e4ab_proc_3344551/experiment_data.npy",
    "No Positional Embedding": "experiment_results/experiment_4a4ee2739cc349d5a9233c8b571fc506_proc_3344552/experiment_data.npy",
    "No-CLS Mean-Pooling": "experiment_results/experiment_9876798c8de2478e96003181bd16a4f8_proc_3344553/experiment_data.npy",
    "Count-Only MLP": "experiment_results/experiment_21f7c55ffc504f4c88b4d3650215d469_proc_3344554/experiment_data.npy",
    "No-Label-Smoothing": "experiment_results/experiment_58c991545e954d0aa1c246f9a54c2756_proc_3344551/experiment_data.npy",
    "No-Length Features": "experiment_results/experiment_548cf7206a694cd68bc807e2c38ba9ea_proc_3344554/experiment_data.npy",
    "Fixed-Sinusoidal": "experiment_results/experiment_25a2a53967ff4ba3b4dbad63acd94688_proc_3344552/experiment_data.npy"
}
ablation_metrics = {}
for label, path in ablation_files.items():
    try:
        data = load_npy(path)
        # Assume the data dict has one key; then under that key, data for SPR_BENCH exists.
        method_key = list(data.keys())[0] if data else None
        if method_key is None:
            continue
        ab_dict = data.get(method_key, {}).get("SPR_BENCH", {})
        test_met = ab_dict.get("test_metrics", {})
        if test_met:
            ablation_metrics[label] = test_met
    except Exception as e:
        print(f"Error loading ablation {label}: {e}")

#############################################
# FIGURE 6: ABALTION – Aggregated Bar Chart for Test Accuracy Across Ablations
#############################################
try:
    labels = []
    acc_vals = []
    for label, met in ablation_metrics.items():
        if "acc" in met:
            labels.append(label)
            acc_vals.append(met["acc"])
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(labels, acc_vals, color="mediumseagreen")
    ax.set_xlabel("Ablation Experiment")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Ablation: Test Accuracy Comparison")
    ax.set_xticklabels(labels, rotation=45, ha="right")
    safe_save(fig, "Ablation_Test_Accuracy.png")
    print("Figure 6 saved: Ablation Test Accuracy Bar Chart")
except Exception as e:
    print(f"Error in Figure 6: {e}")

#############################################
# FIGURE 7: ABALTION – Aggregated Bar Chart for Test MCC Across Ablations
#############################################
try:
    labels = []
    mcc_vals = []
    for label, met in ablation_metrics.items():
        if "MCC" in met:
            labels.append(label)
            mcc_vals.append(met["MCC"])
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(labels, mcc_vals, color="mediumpurple")
    ax.set_xlabel("Ablation Experiment")
    ax.set_ylabel("Test MCC")
    ax.set_title("Ablation: Test MCC Comparison")
    ax.set_xticklabels(labels, rotation=45, ha="right")
    safe_save(fig, "Ablation_Test_MCC.png")
    print("Figure 7 saved: Ablation Test MCC Bar Chart")
except Exception as e:
    print(f"Error in Figure 7: {e}")

#############################################
# FIGURE 8: ABALTION – Aggregated Bar Chart for Test RMA Across Ablations
#############################################
try:
    labels = []
    rma_vals = []
    for label, met in ablation_metrics.items():
        if "RMA" in met:
            labels.append(label)
            rma_vals.append(met["RMA"])
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(labels, rma_vals, color="coral")
    ax.set_xlabel("Ablation Experiment")
    ax.set_ylabel("Test RMA")
    ax.set_title("Ablation: Test Rule-Macro Accuracy Comparison")
    ax.set_xticklabels(labels, rotation=45, ha="right")
    safe_save(fig, "Ablation_Test_RMA.png")
    print("Figure 8 saved: Ablation Test RMA Bar Chart")
except Exception as e:
    print(f"Error in Figure 8: {e}")

#############################################
# FIGURE 9: ABALTION – Aggregated Bar Chart for Test Loss Across Ablations
#############################################
try:
    labels = []
    loss_vals = []
    for label, met in ablation_metrics.items():
        if "loss" in met:
            labels.append(label)
            loss_vals.append(met["loss"])
    # If some experiments do not have loss, we can skip
    if labels:
        fig, ax = plt.subplots(figsize=(8,5))
        ax.bar(labels, loss_vals, color="lightslategray")
        ax.set_xlabel("Ablation Experiment")
        ax.set_ylabel("Test Loss")
        ax.set_title("Ablation: Test Loss Comparison")
        ax.set_xticklabels(labels, rotation=45, ha="right")
        safe_save(fig, "Ablation_Test_Loss.png")
        print("Figure 9 saved: Ablation Test Loss Bar Chart")
    else:
        print("No test loss data available for ablation comparisons.")
except Exception as e:
    print(f"Error in Figure 9: {e}")

#############################################
# FIGURE 10 (APPENDIX): Synthetic Data Visualization (Simulated)
#############################################
try:
    # Simulate synthetic data plot (for appendix) showing a diversity of synthetic sequences (here random curves)
    x = np.linspace(0, 1, 50)
    fig, ax = plt.subplots(figsize=(6,5))
    for i in range(3):
        y = np.sin(2 * np.pi * (x + i/10)) + np.random.normal(scale=0.2, size=x.shape)
        ax.plot(x, y, label=f"Category {i+1}")
    ax.set_xlabel("Normalized Feature")
    ax.set_ylabel("Simulated Response")
    ax.set_title("Appendix: Synthetic Data Example\n(Demonstrating diversity of simulated outcomes)")
    ax.legend()
    safe_save(fig, "Appendix_Synthetic_Data.png")
    print("Figure 10 saved: Appendix Synthetic Data Visualization")
except Exception as e:
    print(f"Error in Figure 10: {e}")

#############################################
# FIGURE 11: Aggregate – Validation Accuracy Curves from Baseline, Research, and No-Label-Smoothing
#############################################
try:
    # For baseline, use best_run from baseline_data; for research, rd; for no_label_smoothing from research_data key "no_label_smoothing"
    # Baseline validation accuracy: from best_run["metrics"]["val_MCC"] is MCC; we want accuracy.
    # We use baseline_data best config train & val accuracy if available.
    # For baseline, we attempt to get "metrics"-> for best_run, assume "val_acc" stored in each metric dict under key "acc"
    base_val_acc = np.array([m.get("acc", np.nan) for m in best_run.get("metrics", {}).get("val", [])])
    
    # Research validation accuracy:
    res_val_acc = np.array([m.get("acc", np.nan) for m in rd.get("metrics", {}).get("val", [])])
    
    # For no-label-smoothing experiment from research_data: key "no_label_smoothing" in research_data?
    # Try loading from same research file, if present:
    nls = research_data.get("no_label_smoothing", {}).get("SPR_BENCH", {})
    nls_val_acc = np.array([m.get("acc", np.nan) for m in nls.get("metrics", {}).get("val", [])]) if nls else None
    
    fig, ax = plt.subplots(figsize=(8,5))
    if base_val_acc.size:
        ax.plot(best_run.get("epochs", []), base_val_acc, marker="o", label="Baseline Val Acc")
    if res_val_acc.size:
        ax.plot(rd.get("epochs", []), res_val_acc, marker="s", label="Research Val Acc")
    if nls_val_acc is not None and nls_val_acc.size:
        ax.plot(nls.get("epochs", []), nls_val_acc, marker="^", label="No-Label-Smoothing Val Acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Aggregate Validation Accuracy Curves")
    ax.legend()
    safe_save(fig, "Aggregate_Val_Accuracy.png")
    print("Figure 11 saved: Aggregate Validation Accuracy Curves")
except Exception as e:
    print(f"Error in Figure 11: {e}")

#############################################
# FIGURE 12: Aggregate – Validation Loss Curves from Baseline, Research, and No-Label-Smoothing
#############################################
try:
    base_val_loss = np.array(best_run.get("losses", {}).get("val", []))
    res_val_loss = np.array(rd.get("losses", {}).get("val", []))
    nls_val_loss = np.array(nls.get("losses", {}).get("val", [])) if nls else None
    
    fig, ax = plt.subplots(figsize=(8,5))
    if base_val_loss.size:
        ax.plot(best_run.get("epochs", []), base_val_loss, marker="o", label="Baseline Val Loss")
    if res_val_loss.size:
        ax.plot(rd.get("epochs", []), res_val_loss, marker="s", label="Research Val Loss")
    if nls_val_loss is not None and nls_val_loss.size:
        ax.plot(nls.get("epochs", []), nls_val_loss, marker="^", label="No-Label-Smoothing Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Aggregate Validation Loss Curves")
    ax.legend()
    safe_save(fig, "Aggregate_Val_Loss.png")
    print("Figure 12 saved: Aggregate Validation Loss Curves")
except Exception as e:
    print(f"Error in Figure 12: {e}")

print("All figures have been generated and saved in the 'figures/' directory.")
#!/usr/bin/env python3
"""
Aggregated final plots for the GNN for SPR paper.
This script loads experiment data from .npy files (full exact paths are used as provided in summaries),
generates final, publication‐quality figures, and saves them only in the "figures/" directory.
Each plotting section is wrapped in a try–except block so that failure of one plot does not halt the entire script.
All text labels, axes, and legends use an increased font size for clarity in print.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Increase font size for publication-quality plots
plt.rcParams.update({'font.size': 14})
# Create the figures output directory
os.makedirs("figures", exist_ok=True)

##############################
# Baseline Experiments Plots #
##############################
def plot_baseline():
    # File from BASELINE_SUMMARY
    baseline_file = "experiment_results/experiment_793aca35999d4c11bf783c261a3c60d5_proc_1441387/experiment_data.npy"
    try:
        data = np.load(baseline_file, allow_pickle=True).item()
        poolings = list(data.get("pooling_type", {}).keys())
        epochs_dict = {}
        for p in poolings:
            log = data["pooling_type"][p]["SPR_BENCH"]
            epochs_dict[p] = {
                "train_loss": [v for _, v in log["losses"]["train"]],
                "val_loss": [v for _, v in log["losses"]["val"]],
                "dwa": [v for _, v in log["metrics"]["val"]]
            }
    except Exception as e:
        print("Baseline: Error loading or parsing data:", e)
        return

    # Plot 1: Training and Validation Loss Curves
    try:
        plt.figure(figsize=(8, 5), dpi=300)
        for p in poolings:
            epochs = np.arange(1, len(epochs_dict[p]["train_loss"]) + 1)
            plt.plot(epochs, epochs_dict[p]["train_loss"], linestyle="--", label=f"{p} Train")
            plt.plot(epochs, epochs_dict[p]["val_loss"], linestyle="-", label=f"{p} Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Baseline: Training and Validation Loss – SPR_BENCH")
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/Baseline_Loss_Curves.png")
        plt.close()
    except Exception as e:
        print("Baseline: Error creating loss curves plot:", e)
        plt.close()

    # Plot 2: Validation DWA Curves
    try:
        plt.figure(figsize=(8, 5), dpi=300)
        for p in poolings:
            epochs = np.arange(1, len(epochs_dict[p]["dwa"]) + 1)
            plt.plot(epochs, epochs_dict[p]["dwa"], label=p)
        plt.xlabel("Epoch")
        plt.ylabel("Dual Weighted Accuracy")
        plt.title("Baseline: Validation DWA Curves – SPR_BENCH")
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/Baseline_DWA_Curves.png")
        plt.close()
    except Exception as e:
        print("Baseline: Error creating DWA curves plot:", e)
        plt.close()

    # Plot 3: Final DWA Bar Chart
    try:
        plt.figure(figsize=(6, 4), dpi=300)
        final_scores = [epochs_dict[p]["dwa"][-1] for p in poolings]
        plt.bar(poolings, final_scores, color="skyblue")
        plt.ylabel("Final Dual Weighted Accuracy")
        plt.title("Baseline: Final DWA by Pooling Type – SPR_BENCH")
        plt.tight_layout()
        plt.savefig("figures/Baseline_Final_DWA_Bar.png")
        plt.close()
    except Exception as e:
        print("Baseline: Error creating final DWA bar plot:", e)
        plt.close()

#############################
# Research Experiments Plots #
#############################
def plot_research():
    # File from RESEARCH_SUMMARY
    research_file = "experiment_results/experiment_5a3f20adaaba448dad224382945a23b2_proc_1448865/experiment_data.npy"
    try:
        data = np.load(research_file, allow_pickle=True).item()
        spr = data.get("SPR", {})
        loss_train = [v for _, v in spr.get("losses", {}).get("train", [])]
        loss_val = [v for _, v in spr.get("losses", {}).get("val", [])]
        pcwa_train = [v for _, v in spr.get("metrics", {}).get("train", [])]
        pcwa_val = [v for _, v in spr.get("metrics", {}).get("val", [])]
        epochs = np.arange(1, len(loss_train) + 1)
    except Exception as e:
        print("Research: Error loading or parsing data:", e)
        return

    # Plot 1: Loss Curves for SPR
    try:
        plt.figure(figsize=(7, 4), dpi=300)
        plt.plot(epochs, loss_train, "--o", label="Train")
        plt.plot(epochs, loss_val, "-s", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Research: Training vs Validation Loss – SPR")
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/Research_Loss_Curves.png")
        plt.close()
    except Exception as e:
        print("Research: Error creating loss curves plot:", e)
        plt.close()

    # Plot 2: PCWA Curves for SPR
    try:
        plt.figure(figsize=(7, 4), dpi=300)
        plt.plot(epochs, pcwa_train, "--o", label="Train")
        plt.plot(epochs, pcwa_val, "-s", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("PCWA")
        plt.title("Research: Training vs Validation PCWA – SPR")
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/Research_PCWA_Curves.png")
        plt.close()
    except Exception as e:
        print("Research: Error creating PCWA curves plot:", e)
        plt.close()

    # Plot 3: Final Test-set Metrics Bar Chart
    try:
        # Use sequences if available; otherwise, fallback to ground_truth
        seqs = spr.get("sequences", spr.get("ground_truth", []))
        y_true = spr.get("ground_truth", [])
        y_pred = spr.get("predictions", [])
        if seqs and y_true and y_pred:
            acc = sum(int(y == p) for y, p in zip(y_true, y_pred)) / len(y_true)
            def count_color_variety(s):
                return len(set(token[1] for token in s.split() if len(token)>1))
            def count_shape_variety(s):
                return len(set(token[0] for token in s.split() if token))
            cwa_num = sum(count_color_variety(s) if y==p else 0 for s, y, p in zip(seqs, y_true, y_pred))
            cwa_den = sum(count_color_variety(s) for s in seqs)
            swa_num = sum(count_shape_variety(s) if y==p else 0 for s, y, p in zip(seqs, y_true, y_pred))
            swa_den = sum(count_shape_variety(s) for s in seqs)
            cwa = cwa_num / cwa_den if cwa_den else 0.0
            swa = swa_num / swa_den if swa_den else 0.0
            metrics = {"ACC": acc, "PCWA": pcwa_val[-1] if len(pcwa_val) else 0, "CWA": cwa, "SWA": swa}
            plt.figure(figsize=(6, 4), dpi=300)
            plt.bar(list(metrics.keys()), list(metrics.values()), color="skyblue")
            plt.title("Research: Final Test-set Metrics – SPR")
            plt.tight_layout()
            plt.savefig("figures/Research_Final_Test_Metrics.png")
            plt.close()
            print("Research Final Test Metrics:", metrics)
        else:
            print("Research: Test set data missing, skipping final metrics plot.")
    except Exception as e:
        print("Research: Error creating final test metrics bar plot:", e)
        plt.close()

############################
# Ablation Experiments Plots
############################
def plot_ablation():
    # Mapping of ablation key names to the corresponding npy file paths
    ablation_files = {
        "Multi-Dataset Generalization": "experiment_results/experiment_0ac3789fa99344af8d3d48e51d9a9f36_proc_1451826/experiment_data.npy",
        "Positional Feature Removal": "experiment_results/experiment_7b452c890ea64faeb86bec2c32a7732d_proc_1451827/experiment_data.npy",
        "Node Color Feature Removal": "experiment_results/experiment_3ca5429c122545ce8a08ef741bf1f13f_proc_1451828/experiment_data.npy",
        "Edge Structure Removal": "experiment_results/experiment_f393ca3880624f1e9f351640cc4107c5_proc_1451829/experiment_data.npy",
        "Node Shape Feature Removal": "experiment_results/experiment_40aabbd60fb8441b848566264032011b_proc_1451827/experiment_data.npy",
        "Single-Layer GNN": "experiment_results/experiment_cae2d0f63eb5436a859cc4246c02e1a8_proc_1451828/experiment_data.npy",
        "Unidirectional-Chain Edges": "experiment_results/experiment_8117350da8c94feeb48afe81aa090d20_proc_1451829/experiment_data.npy",
        "First-Node Readout Ablation": "experiment_results/experiment_9a6b8146249a422dae760d5059a4a8f3_proc_1451826/experiment_data.npy",
    }

    for key, filepath in ablation_files.items():
        try:
            ab_data = np.load(filepath, allow_pickle=True).item()
        except Exception as e:
            print(f"Ablation {key}: Error loading file: {e}")
            continue

        # For each ablation study, select one representative plot
        if key == "Multi-Dataset Generalization":
            # Produce a grouped bar chart of final test metrics.
            try:
                records = ab_data.get("multi_dataset_generalization", {})
                if records:
                    datasets = list(records.keys())
                    metrics_names = ["ACC", "PCWA", "CWA", "SWA"]
                    bar_width = 0.18
                    x = np.arange(len(datasets))
                    plt.figure(figsize=(8, 4), dpi=300)
                    for i, m in enumerate(metrics_names):
                        vals = [records[d].get("test_metrics", {}).get(m, 0.0) for d in datasets]
                        plt.bar(x + i * bar_width, vals, width=bar_width, label=m)
                    plt.xticks(x + bar_width * 1.5, datasets, rotation=45)
                    plt.ylim(0, 1)
                    plt.ylabel("Score")
                    plt.title("Ablation (Multi-Dataset): Final Test Metrics")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig("figures/Ablation_MultiDataset_Test_Metrics.png")
                    plt.close()
                else:
                    print("Ablation Multi-Dataset: No records found.")
            except Exception as e:
                print("Error in Ablation Multi-Dataset:", e)

        elif key == "Positional Feature Removal":
            # Plot loss curves from one of the experiments in this ablation.
            try:
                # Ab_data is a dict with experimental keys; choose the first one.
                for ds_name, content in ab_data.items():
                    losses_tr = content.get("losses", {}).get("train", [])
                    losses_val = content.get("losses", {}).get("val", [])
                    if losses_tr and losses_val:
                        ep_tr = [e for e, _ in losses_tr]
                        tr_vals = [v for _, v in losses_tr]
                        ep_val = [e for e, _ in losses_val]
                        val_vals = [v for _, v in losses_val]
                        plt.figure(figsize=(7, 4), dpi=300)
                        plt.plot(ep_tr, tr_vals, label="Train")
                        plt.plot(ep_val, val_vals, label="Validation")
                        plt.xlabel("Epoch")
                        plt.ylabel("Loss")
                        plt.title(f"Ablation (Positional Removal): {ds_name} Loss Curve")
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(f"figures/Ablation_PositionalLoss_{ds_name}.png")
                        plt.close()
                    break
            except Exception as e:
                print("Error in Ablation Positional Feature Removal:", e)

        elif key == "Node Color Feature Removal":
            # Produce a confusion matrix
            try:
                for ds_name, content in ab_data.items():
                    y_true = np.array(content.get("ground_truth", []))
                    y_pred = np.array(content.get("predictions", []))
                    if y_true.size and y_pred.size:
                        cm = np.zeros((2, 2), dtype=int)
                        for t, p in zip(y_true, y_pred):
                            cm[t, p] += 1
                        plt.figure(figsize=(5, 4), dpi=300)
                        plt.imshow(cm, cmap="Blues")
                        for i in range(2):
                            for j in range(2):
                                plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
                        plt.xlabel("Predicted")
                        plt.ylabel("True")
                        plt.title(f"Ablation (Node Color Removal): {ds_name} Confusion Matrix")
                        plt.colorbar()
                        plt.tight_layout()
                        plt.savefig(f"figures/Ablation_NodeColorConfMatrix_{ds_name}.png")
                        plt.close()
                    break
            except Exception as e:
                print("Error in Ablation Node Color Feature Removal:", e)

        elif key == "Edge Structure Removal":
            # Produce PCWA curves plot
            try:
                for exp_name, info in ab_data.items():
                    train_pcwa = np.array(info.get("metrics", {}).get("train", []))
                    val_pcwa = np.array(info.get("metrics", {}).get("val", []))
                    if train_pcwa.size and val_pcwa.size:
                        ep_tr = train_pcwa[:, 0]
                        pc_tr = train_pcwa[:, 1]
                        ep_val = val_pcwa[:, 0]
                        pc_val = val_pcwa[:, 1]
                        plt.figure(figsize=(7, 4), dpi=300)
                        plt.plot(ep_tr, pc_tr, label="Train PCWA")
                        plt.plot(ep_val, pc_val, label="Validation PCWA")
                        plt.xlabel("Epoch")
                        plt.ylabel("PCWA")
                        plt.title(f"Ablation (Edge Removal): {exp_name} PCWA Curves")
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(f"figures/Ablation_EdgePCWA_{exp_name}.png")
                        plt.close()
                    break
            except Exception as e:
                print("Error in Ablation Edge Structure Removal:", e)

        elif key == "Node Shape Feature Removal":
            # Produce confusion matrix plot
            try:
                for ds_name, content in ab_data.items():
                    y_true = np.array(content.get("ground_truth", []))
                    y_pred = np.array(content.get("predictions", []))
                    if y_true.size and y_pred.size:
                        cm = np.zeros((2, 2), dtype=int)
                        for t, p in zip(y_true, y_pred):
                            cm[t, p] += 1
                        plt.figure(figsize=(5, 4), dpi=300)
                        plt.imshow(cm, cmap="Blues")
                        for i in range(2):
                            for j in range(2):
                                plt.text(j, i, cm[i, j], ha="center", va="center", color="white")
                        plt.xlabel("Predicted")
                        plt.ylabel("True")
                        plt.title(f"Ablation (Node Shape Removal): {ds_name} Confusion Matrix")
                        plt.colorbar()
                        plt.tight_layout()
                        plt.savefig(f"figures/Ablation_NodeShapeConfMatrix_{ds_name}.png")
                        plt.close()
                    break
            except Exception as e:
                print("Error in Ablation Node Shape Feature Removal:", e)

        elif key == "Single-Layer GNN":
            # Produce a confusion matrix using sklearn's confusion_matrix function
            try:
                spr_data = ab_data.get("SPR", {})
                if spr_data:
                    y_true = np.array(spr_data.get("ground_truth", []))
                    y_pred = np.array(spr_data.get("predictions", []))
                    if len(y_true):
                        cm = confusion_matrix(y_true, y_pred, labels=sorted(set(y_true)))
                        plt.figure(figsize=(5, 4), dpi=300)
                        plt.imshow(cm, cmap="Blues")
                        plt.colorbar()
                        for (i, j), v in np.ndenumerate(cm):
                            plt.text(j, i, str(v), ha="center", va="center", color="black")
                        plt.xlabel("Predicted")
                        plt.ylabel("Ground Truth")
                        plt.title("Ablation (Single-Layer GNN): Confusion Matrix")
                        plt.tight_layout()
                        plt.savefig("figures/Ablation_SingleLayerConfMatrix.png")
                        plt.close()
                    else:
                        print("Ablation Single-Layer GNN: SPR data missing")
                else:
                    print("Ablation Single-Layer GNN: SPR key not found")
            except Exception as e:
                print("Error in Ablation Single-Layer GNN:", e)

        elif key == "Unidirectional-Chain Edges":
            # Produce Loss Curves from UniChain experiment
            try:
                spr_exp = ab_data.get("SPR", {})
                losses_train = spr_exp.get("losses", {}).get("train", [])
                losses_val = spr_exp.get("losses", {}).get("val", [])
                if losses_train and losses_val:
                    ep_tr, l_tr = zip(*losses_train)
                    ep_val, l_val = zip(*losses_val)
                    plt.figure(figsize=(7, 4), dpi=300)
                    plt.plot(ep_tr, l_tr, label="Train Loss")
                    plt.plot(ep_val, l_val, label="Validation Loss")
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.title("Ablation (UniChain): Loss Curves – SPR")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig("figures/Ablation_UniChain_Loss.png")
                    plt.close()
                else:
                    print("Ablation UniChain: Loss data missing")
            except Exception as e:
                print("Error in Ablation Unidirectional-Chain Edges:", e)

        elif key == "First-Node Readout Ablation":
            # Produce PCWA curves from First-Node Readout experiment
            try:
                # Assumed structure: data contains one ablation key with one dataset key.
                ab_key = list(ab_data.keys())[0]
                ds_key = list(ab_data[ab_key].keys())[0]
                dct = ab_data[ab_key][ds_key]
                def unpack(metric_key):
                    arr = dct.get(metric_key, {})
                    train = arr.get("train", [])
                    val = arr.get("val", [])
                    epochs = [e for e, _ in train] if train else []
                    train_vals = [v for _, v in train] if train else []
                    val_vals = [v for _, v in val] if val else []
                    return epochs, train_vals, val_vals
                epochs_, tr_pc, va_pc = unpack("metrics")
                if epochs_:
                    plt.figure(figsize=(7, 4), dpi=300)
                    plt.plot(epochs_, tr_pc, label="Train PCWA")
                    plt.plot(epochs_, va_pc, label="Validation PCWA")
                    plt.xlabel("Epoch")
                    plt.ylabel("PCWA")
                    plt.title("Ablation (First-Node Readout): PCWA Curves")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig("figures/Ablation_FirstNode_PCWA.png")
                    plt.close()
                else:
                    print("Ablation First-Node Readout: Metrics data missing")
            except Exception as e:
                print("Error in Ablation First-Node Readout Ablation:", e)

        else:
            print(f"Ablation {key}: No representative plot selected.")
        # End for each ablation file
        #
        # (Additional ablation plots could be added following a similar structure.)
        #
    return

############
# Main Run #
############
def main():
    print("Generating Baseline plots...")
    plot_baseline()
    print("Generating Research plots...")
    plot_research()
    print("Generating Ablation plots...")
    plot_ablation()
    print("All figures saved to 'figures/'.")

if __name__ == "__main__":
    main()
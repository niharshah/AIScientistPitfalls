"""
Final Aggregator Script for GNN for SPR Final Plots

This script aggregates experiment results from baseline, research, and ablation studies.
It loads existing .npy files (using full paths from the summaries) and produces a set
of final, publication‐quality figures saved in the "figures/" directory.

Plots produced (each wrapped in a try–except block so that individual failures do not stop the script):
-----------------------------------------------------------------------------------------------
Baseline (from exp_results_npy_files in BASELINE_SUMMARY):
  Figure 1: Loss Curves by Batch Size (aggregated subplots)
  Figure 2: Validation CompWA Curves by Batch Size (aggregated subplots)
  Figure 3: Test CompWA Summary (bar chart)
  Appendix A: Loss Curve for Synthetic Dataset (if present)

Research (from RESEARCH_SUMMARY):
  Figure 4: SPR_BENCH – Aggregated Loss and Validation Metrics (Loss, CWA, SWA, CplxWA) as 4 subplots
  Figure 5: Test CplxWA Summary by Dataset (bar chart)

Ablation (select key experiments from ABLATION_SUMMARY):
  Figure 6: Multi-Rule Synthetic Ablation – Validation Accuracy Curves (one line plot for each rule dataset)
  Figure 7: Multi-Rule Synthetic Ablation – Transfer Accuracy Heatmap
  Figure 8: Static One-Hot Node Feature Ablation – Loss and Metric Curves (aggregated: Loss, CWA, SWA, CplxWA)
  Figure 9: Single-Hop RGCN Ablation – Loss, Metrics, and Test Metrics Bar Chart

All plots use a larger font size, a dpi of 300, and professional styling.
No data are fabricated; all are loaded from the .npy files.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Set global font size for publication-quality plots.
plt.rcParams.update({'font.size': 14, 'axes.spines.top': False, 'axes.spines.right': False})

# Create directory for final figures.
os.makedirs("figures", exist_ok=True)

##############################
# Baseline Plots (Best Node)
##############################
def plot_baseline():
    # Load baseline experiment data from the given full path.
    baseline_file = "experiment_results/experiment_ca1bec6343304b748fede089172b795c_proc_1490563/experiment_data.npy"
    try:
        baseline_data = np.load(baseline_file, allow_pickle=True).item()
    except Exception as e:
        print("Error loading baseline data:", e)
        return

    # Expecting baseline_data to have a key "batch_size" with batch entries.
    bs_data = baseline_data.get("batch_size", {})
    if not bs_data:
        print("No batch_size data found in baseline.")
        return

    # Gather the batch keys sorted (e.g., bs16, bs32, bs64, bs128)
    batch_keys = sorted(bs_data.keys(), key=lambda k: int(k.split("_")[-1]))
    
    # Figure 1: Loss Curves by Batch Size (each subplot shows train and val loss)
    try:
        n_batches = len(batch_keys)
        cols = min(3, n_batches)
        rows = (n_batches + cols - 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*4), squeeze=False)
        for i, key in enumerate(batch_keys):
            subdict = bs_data[key]
            epochs = subdict.get("epochs", [])
            train_loss = subdict.get("losses", {}).get("train", [])
            val_loss = subdict.get("losses", {}).get("val", [])
            ax = axs[i // cols][i % cols]
            ax.plot(epochs, train_loss, label="Train", marker='o')
            ax.plot(epochs, val_loss, label="Validation", marker='o')
            bs_val = key.split("_")[-1]
            ax.set_title(f"Loss Curve (Batch Size {bs_val})")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join("figures", "baseline_loss_curves.png"), dpi=300)
        plt.close(fig)
    except Exception as e:
        print("Error in Baseline Figure 1 (Loss Curves):", e)
        plt.close()

    # Figure 2: Validation CompWA Curves by Batch Size
    try:
        n_batches = len(batch_keys)
        cols = min(3, n_batches)
        rows = (n_batches + cols - 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*4), squeeze=False)
        for i, key in enumerate(batch_keys):
            subdict = bs_data[key]
            epochs = subdict.get("epochs", [])
            val_compwa = subdict.get("metrics", {}).get("val_compwa", [])
            ax = axs[i // cols][i % cols]
            ax.plot(epochs, val_compwa, marker='o', color="green")
            bs_val = key.split("_")[-1]
            ax.set_title(f"Validation CompWA (Batch Size {bs_val})")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("CompWA")
            ax.set_ylim(0, 1)
            ax.legend(["Validation CompWA"])
        plt.tight_layout()
        fig.savefig(os.path.join("figures", "baseline_compwa_curves.png"), dpi=300)
        plt.close(fig)
    except Exception as e:
        print("Error in Baseline Figure 2 (CompWA Curves):", e)
        plt.close()

    # Figure 3: Test CompWA Summary Bar Plot
    try:
        test_scores = {}
        for key in batch_keys:
            subdict = bs_data[key]
            bs_val = int(key.split("_")[-1])
            test_scores[bs_val] = subdict.get("metrics", {}).get("test_compwa", None)
        if test_scores:
            bs_vals = sorted(test_scores.keys())
            scores = [test_scores[bs] for bs in bs_vals]
            fig, ax = plt.subplots(figsize=(6,5))
            ax.bar([str(bs) for bs in bs_vals], scores, color="skyblue")
            ax.set_title("Test CompWA by Batch Size")
            ax.set_xlabel("Batch Size")
            ax.set_ylabel("Test CompWA")
            ax.set_ylim(0, 1)
            ax.legend(["Test CompWA"])
            fig.tight_layout()
            fig.savefig(os.path.join("figures", "baseline_test_compwa_summary.png"), dpi=300)
            plt.close(fig)
        else:
            print("No test CompWA scores to plot in baseline.")
    except Exception as e:
        print("Error in Baseline Figure 3 (Test CompWA Summary):", e)
        plt.close()

    # Appendix Figure A1: Synthetic Data Loss Curve (if available)
    # Some baseline entries might include synthetic dataset info (e.g., key containing "SPR_synth").
    try:
        for key in bs_data.keys():
            if "synth" in key.lower():
                subdict = bs_data[key]
                epochs = subdict.get("epochs", [])
                train_loss = subdict.get("losses", {}).get("train", [])
                val_loss = subdict.get("losses", {}).get("val", [])
                fig, ax = plt.subplots(figsize=(6,5))
                ax.plot(epochs, train_loss, label="Train", marker='o')
                ax.plot(epochs, val_loss, label="Validation", marker='o')
                ax.set_title("Synthetic Dataset Loss Curve")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.legend()
                fig.tight_layout()
                fig.savefig(os.path.join("figures", "appendix_synthetic_loss_curve.png"), dpi=300)
                plt.close(fig)
                break  # plot only one synthetic figure if present
    except Exception as e:
        print("Error in Appendix Synthetic Loss Curve:", e)
        plt.close()

##############################
# Research Plots
##############################
def plot_research():
    # Load research experiment data from the given full path.
    research_file = "experiment_results/experiment_e78be9bcc34648e591f8982238addc99_proc_1493251/experiment_data.npy"
    try:
        research_data = np.load(research_file, allow_pickle=True).item()
    except Exception as e:
        print("Error loading research data:", e)
        return

    # Assume research_data has keys corresponding to datasets (e.g., "SPR_BENCH")
    # For final paper assume one primary dataset "SPR_BENCH"
    ds_name = "SPR_BENCH"
    if ds_name not in research_data:
        print(f"{ds_name} not found in research data.")
        return
    ddata = research_data[ds_name]
    epochs = ddata.get("epochs", [])
    losses = ddata.get("losses", {})
    metrics = ddata.get("metrics", {}).get("val", {})

    # Figure 4: Aggregated Plots of Loss and Validation Metrics (4 subplots)
    try:
        fig, axs = plt.subplots(1, 4, figsize=(20,5))
        # Loss curve: using train and validation
        axs[0].plot(epochs, losses.get("train", []), label="Train", marker='o')
        axs[0].plot(epochs, losses.get("val", []), label="Validation", marker='o')
        axs[0].set_title(f"{ds_name} Loss Curve")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].legend()
        # Validation CWA
        axs[1].plot(epochs, metrics.get("CWA", []), color="purple", marker='o')
        axs[1].set_title(f"{ds_name} Val CWA")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("CWA")
        axs[1].set_ylim(0, 1)
        axs[1].legend(["CWA"])
        # Validation SWA
        axs[2].plot(epochs, metrics.get("SWA", []), color="orange", marker='o')
        axs[2].set_title(f"{ds_name} Val SWA")
        axs[2].set_xlabel("Epoch")
        axs[2].set_ylabel("SWA")
        axs[2].set_ylim(0, 1)
        axs[2].legend(["SWA"])
        # Validation CplxWA
        axs[3].plot(epochs, metrics.get("CplxWA", []), color="green", marker='o')
        axs[3].set_title(f"{ds_name} Val CplxWA")
        axs[3].set_xlabel("Epoch")
        axs[3].set_ylabel("CplxWA")
        axs[3].set_ylim(0, 1)
        axs[3].legend(["CplxWA"])
        fig.tight_layout()
        fig.savefig(os.path.join("figures", "research_sprbench_metrics.png"), dpi=300)
        plt.close(fig)
    except Exception as e:
        print("Error in Research Figure 4 (Aggregated Metrics):", e)
        plt.close()

    # Figure 5: Test CplxWA Summary Bar Plot by Dataset
    try:
        # Collect test metrics from each dataset entry in research_data.
        test_summary = {}
        for ds in research_data:
            tst = research_data[ds].get("metrics", {}).get("test", {})
            if "CplxWA" in tst:
                test_summary[ds] = tst["CplxWA"]
        if test_summary:
            labels = list(test_summary.keys())
            values = [test_summary[k] for k in labels]
            fig, ax = plt.subplots(figsize=(6,5))
            ax.bar(labels, values, color="salmon")
            ax.set_title("Test CplxWA by Dataset")
            ax.set_xlabel("Dataset")
            ax.set_ylabel("CplxWA")
            ax.set_ylim(0, 1)
            ax.legend(["Test CplxWA"])
            fig.tight_layout()
            fig.savefig(os.path.join("figures", "research_test_cplxwa_summary.png"), dpi=300)
            plt.close(fig)
        else:
            print("No test CplxWA data found in research.")
    except Exception as e:
        print("Error in Research Figure 5 (Test CplxWA Summary):", e)
        plt.close()

##############################
# Ablation Plots
##############################
def plot_ablation():
    # --- Figure 6 and 7 from "Multi-Rule Synthetic Dataset Ablation"
    multi_rule_file = "experiment_results/experiment_90c6080f5c6140c9ab6566fd1ad606ea_proc_1497709/experiment_data.npy"
    try:
        multi_rule = np.load(multi_rule_file, allow_pickle=True).item().get("Multi-Rule Synthetic Dataset Ablation", {})
    except Exception as e:
        print("Error loading multi-rule ablation data:", e)
        multi_rule = {}
    if multi_rule:
        # Define dataset list as in the summary.
        ds_list = ["variety", "freq", "mod", "union"]
        # Figure 6: Validation Accuracy over Epochs for each rule dataset
        try:
            fig, ax = plt.subplots(figsize=(7,5))
            for ds in ds_list:
                subdict = multi_rule.get(ds, {})
                val_acc = subdict.get("losses", {}).get("val_acc", [])
                if val_acc:
                    ax.plot(range(1, len(val_acc) + 1), val_acc, marker='o', label=ds.capitalize())
            ax.set_title("Multi-Rule Ablation: Validation Accuracy")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join("figures", "multirule_val_accuracy.png"), dpi=300)
            plt.close(fig)
        except Exception as e:
            print("Error in Ablation Figure 6 (Validation Accuracy):", e)
            plt.close()

        # Figure 7: Transfer Accuracy Heatmap
        try:
            # Create a heatmap where rows are trained on ds_list and columns are test on subset ["variety", "freq", "mod"]
            test_ds = ["variety", "freq", "mod"]
            heat = np.zeros((len(ds_list), len(test_ds)))
            for i, train_ds in enumerate(ds_list):
                tr_dict = multi_rule.get(train_ds, {}).get("transfer_acc", {})
                for j, test_name in enumerate(test_ds):
                    # Expect each entry to have a key "accuracy"
                    val_entry = tr_dict.get(test_name, {})
                    heat[i,j] = val_entry.get("accuracy", 0)
            fig, ax = plt.subplots(figsize=(6,5))
            im = ax.imshow(heat, cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks(range(len(test_ds)))
            ax.set_xticklabels(test_ds, rotation=45)
            ax.set_yticks(range(len(ds_list)))
            ax.set_yticklabels([x.capitalize() for x in ds_list])
            ax.set_title("Multi-Rule Ablation: Transfer Accuracy\n(Rows: Trained on, Columns: Tested on)")
            for i in range(heat.shape[0]):
                for j in range(heat.shape[1]):
                    color = "white" if heat[i,j] < 0.5 else "black"
                    ax.text(j, i, f"{heat[i,j]:.2f}", ha="center", va="center", color=color)
            fig.colorbar(im, ax=ax, label="Accuracy")
            fig.tight_layout()
            fig.savefig(os.path.join("figures", "multirule_transfer_heatmap.png"), dpi=300)
            plt.close(fig)
        except Exception as e:
            print("Error in Ablation Figure 7 (Transfer Heatmap):", e)
            plt.close()

    # --- Figure 8 from "Static One-Hot Node Feature Ablation"
    static_onehot_file = "experiment_results/experiment_38bc98fc58654764b1864b291605fa9e_proc_1497711/experiment_data.npy"
    try:
        static_data = np.load(static_onehot_file, allow_pickle=True).item()
    except Exception as e:
        print("Error loading static one-hot data:", e)
        static_data = {}
    # We expect static_data to have keys for different variants (e.g. dual_channel, shape_only, color_only)
    if static_data:
        variants = list(static_data.keys())
        # Figure 8A: For each variant, plot Loss Curves over epochs on SPR_BENCH dataset.
        try:
            fig, axs = plt.subplots(1, len(variants), figsize=(len(variants)*5, 4), squeeze=False)
            for i, var in enumerate(variants):
                try:
                    dset = static_data[var]["SPR_BENCH"]
                    epochs = dset.get("epochs", [])
                    train_loss = dset.get("losses", {}).get("train", [])
                    val_loss = dset.get("losses", {}).get("val", [])
                    ax = axs[0][i]
                    ax.plot(epochs, train_loss, label="Train", marker='o')
                    ax.plot(epochs, val_loss, label="Validation", marker='o')
                    ax.set_title(f"{var} Loss")
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Loss")
                    ax.legend()
                except Exception as inner_e:
                    print(f"Error plotting loss for {var}:", inner_e)
            fig.tight_layout()
            fig.savefig(os.path.join("figures", "static_onehot_loss_curves.png"), dpi=300)
            plt.close(fig)
        except Exception as e:
            print("Error in Static One-Hot Figure 8A (Loss Curves):", e)
            plt.close()

        # Figure 8B: Aggregated metric curves (CWA, SWA, CplxWA) for one variant (e.g. first variant)
        try:
            var = variants[0]
            dset = static_data[var]["SPR_BENCH"]
            epochs = dset.get("epochs", [])
            metrics_train = dset.get("metrics", {}).get("train", {})
            metrics_val = dset.get("metrics", {}).get("val", {})
            metric_names = ["CWA", "SWA", "CplxWA"]
            cols = len(metric_names)
            fig, axs = plt.subplots(1, cols, figsize=(cols*5,4))
            for i, m in enumerate(metric_names):
                tr = metrics_train.get(m, [])
                vl = metrics_val.get(m, [])
                axs[i].plot(epochs, tr, label="Train", marker='o')
                axs[i].plot(epochs, vl, label="Validation", marker='o')
                axs[i].set_title(f"{var} {m}")
                axs[i].set_xlabel("Epoch")
                axs[i].set_ylabel(m)
                axs[i].set_ylim(0, 1)
                axs[i].legend()
            fig.tight_layout()
            fig.savefig(os.path.join("figures", "static_onehot_metric_curves.png"), dpi=300)
            plt.close(fig)
        except Exception as e:
            print("Error in Static One-Hot Figure 8B (Metric Curves):", e)
            plt.close()
    
    # --- Figure 9 from "Single-Hop RGCN Ablation"
    single_hop_file = "experiment_results/experiment_7207e88c77d0442ab202f2b2010dfe0d_proc_1497709/experiment_data.npy"
    try:
        single_hop = np.load(single_hop_file, allow_pickle=True).item().get("single_hop_rgcn", {}).get("SPR_BENCH", {})
    except Exception as e:
        print("Error loading single-hop RGCN data:", e)
        single_hop = {}
    if single_hop:
        epochs = single_hop.get("epochs", [])
        # Figure 9A: Loss curve for Single-Hop RGCN
        try:
            fig, ax = plt.subplots(figsize=(6,5))
            ax.plot(epochs, single_hop.get("losses", {}).get("train", []), label="Train", marker='o')
            ax.plot(epochs, single_hop.get("losses", {}).get("val", []), label="Validation", marker='o')
            ax.set_title("Single-Hop RGCN Loss Curve")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join("figures", "singlehop_loss_curve.png"), dpi=300)
            plt.close(fig)
        except Exception as e:
            print("Error in Single-Hop Figure 9A (Loss Curve):", e)
            plt.close()
        
        # Figure 9B: Metric curves (CWA, SWA, CplxWA) for Single-Hop RGCN
        try:
            metrics_train = single_hop.get("metrics", {}).get("train", {})
            metrics_val = single_hop.get("metrics", {}).get("val", {})
            metric_names = ["CWA", "SWA", "CplxWA"]
            cols = len(metric_names)
            fig, axs = plt.subplots(1, cols, figsize=(cols*5,4))
            for i, m in enumerate(metric_names):
                tr = metrics_train.get(m, [])
                vl = metrics_val.get(m, [])
                axs[i].plot(epochs, tr, label="Train", marker='o')
                axs[i].plot(epochs, vl, label="Validation", marker='o')
                axs[i].set_title(f"{m} Curve")
                axs[i].set_xlabel("Epoch")
                axs[i].set_ylabel(m)
                axs[i].set_ylim(0, 1)
                axs[i].legend()
            fig.tight_layout()
            fig.savefig(os.path.join("figures", "singlehop_metric_curves.png"), dpi=300)
            plt.close(fig)
        except Exception as e:
            print("Error in Single-Hop Figure 9B (Metric Curves):", e)
            plt.close()
        
        # Figure 9C: Test Metrics Bar Chart for Single-Hop RGCN
        try:
            test_metrics = single_hop.get("metrics", {}).get("test", {})
            if test_metrics:
                labels = list(test_metrics.keys())
                values = [test_metrics[k] for k in labels]
                fig, ax = plt.subplots(figsize=(6,5))
                ax.bar(labels, values, color="orchid")
                ax.set_title("Single-Hop RGCN: Test Metrics")
                ax.set_xlabel("Metric")
                ax.set_ylabel("Value")
                ax.set_ylim(0, 1)
                fig.tight_layout()
                fig.savefig(os.path.join("figures", "singlehop_test_metrics.png"), dpi=300)
                plt.close(fig)
                # Also print test metrics to stdout.
                print("Single-Hop RGCN Test Metrics:")
                for k, v in test_metrics.items():
                    print(f"  {k}: {v:.3f}")
            else:
                print("No Single-Hop RGCN test metrics available.")
        except Exception as e:
            print("Error in Single-Hop Figure 9C (Test Metrics Bar Chart):", e)
            plt.close()

##############################
# Run All Plot Functions
##############################
if __name__ == "__main__":
    print("Generating Baseline Figures...")
    plot_baseline()
    print("Generating Research Figures...")
    plot_research()
    print("Generating Ablation Figures...")
    plot_ablation()
    print("All figures saved in 'figures/' directory.")
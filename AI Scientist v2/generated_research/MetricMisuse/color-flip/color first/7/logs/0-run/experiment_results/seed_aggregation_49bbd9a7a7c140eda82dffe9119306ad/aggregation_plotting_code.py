import matplotlib.pyplot as plt
import numpy as np
import os
from math import sqrt

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Load all experiment_data dicts that actually exist on disk
experiment_data_path_list = [
    "experiments/2025-08-31_14-11-51_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_19cca88cbc314673b4f2b9f4ff158634_proc_1723012/experiment_data.npy",
    "experiments/2025-08-31_14-11-51_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_722629c045064cc6ba1521f6302b7317_proc_1723011/experiment_data.npy",
    "experiments/2025-08-31_14-11-51_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_17244b84bcbf422982f041dc468b8afb_proc_1723010/experiment_data.npy",
]
all_experiment_data = []
for p in experiment_data_path_list:
    try:
        root = os.getenv("AI_SCIENTIST_ROOT", ".")
        ed = np.load(os.path.join(root, p), allow_pickle=True).item()
        all_experiment_data.append(ed)
    except Exception as e:
        print(f"Error loading {p}: {e}")

if not all_experiment_data:
    print("No experiment data could be loaded; exiting.")
    exit()

# ----------------------------------------------------------------------
# 2. Aggregate across runs
# Collect all dataset names
dataset_names = set()
for run in all_experiment_data:
    dataset_names.update(run.keys())

for ds_name in dataset_names:
    # Gather per-run arrays; some runs might miss the dataset
    train_losses_runs, val_losses_runs = [], []
    train_metrics_runs, val_metrics_runs = [], []
    epochs_ref = None

    for run in all_experiment_data:
        ds_dict = run.get(ds_name, {})
        epochs = ds_dict.get("epochs", [])
        if epochs and epochs_ref is None:
            epochs_ref = np.array(epochs)
        # Ensure same epoch length for stacking
        if epochs_ref is not None and len(epochs) == len(epochs_ref):
            train_losses_runs.append(ds_dict.get("losses", {}).get("train", []))
            val_losses_runs.append(ds_dict.get("losses", {}).get("val", []))
            train_metrics_runs.append(ds_dict.get("metrics", {}).get("train", []))
            val_metrics_runs.append(ds_dict.get("metrics", {}).get("val", []))

    n_runs = len(train_losses_runs)
    if n_runs == 0:
        continue  # nothing to plot for this dataset

    epochs = epochs_ref

    # ------------------------------------------------------------------
    # Helper to compute mean & SE given list-of-lists and key extractor
    def mean_se(metric_runs, key):
        """Return mean and standard error arrays for given key in metrics."""
        # metric_runs: list[ list[dict] ]
        vals = []
        for run in metric_runs:
            vals.append([m[key] for m in run])
        vals = np.array(vals, dtype=float)  # shape (n_runs, n_epochs)
        mean = vals.mean(axis=0)
        se = (
            vals.std(axis=0, ddof=1) / np.sqrt(vals.shape[0])
            if vals.shape[0] > 1
            else np.zeros_like(mean)
        )
        return mean, se

    # ----------- 1) Training loss mean ± SE --------------------------
    try:
        if all(train_losses_runs):
            train_losses_arr = np.array(
                train_losses_runs, dtype=float
            )  # (n_runs, n_epochs)
            train_mean = train_losses_arr.mean(axis=0)
            train_se = (
                train_losses_arr.std(axis=0, ddof=1) / sqrt(n_runs)
                if n_runs > 1
                else np.zeros_like(train_mean)
            )

            plt.figure()
            plt.plot(epochs, train_mean, color="C0", label="Mean Train Loss")
            plt.fill_between(
                epochs,
                train_mean - train_se,
                train_mean + train_se,
                color="C0",
                alpha=0.3,
                label="SE",
            )
            plt.title(f"{ds_name} Dataset – Training Loss (Mean ± SE)")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            fname = os.path.join(
                working_dir, f"{ds_name.lower()}_train_loss_mean_se.png"
            )
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated training-loss plot for {ds_name}: {e}")
        plt.close()

    # ----------- 2) Validation weighted-accuracy mean ± SE ----------
    try:
        if all(val_metrics_runs):
            cwa_mean, cwa_se = mean_se(val_metrics_runs, "cwa")
            swa_mean, swa_se = mean_se(val_metrics_runs, "swa")
            cpx_mean, cpx_se = mean_se(val_metrics_runs, "cpx")

            plt.figure()
            for mean, se, label, color in zip(
                [cwa_mean, swa_mean, cpx_mean],
                [cwa_se, swa_se, cpx_se],
                ["CWA", "SWA", "CpxWA"],
                ["C0", "C1", "C2"],
            ):
                plt.plot(epochs, mean, color=color, label=f"{label} Mean")
                plt.fill_between(
                    epochs,
                    mean - se,
                    mean + se,
                    color=color,
                    alpha=0.25,
                    label=f"{label} SE",
                )
            plt.title(f"{ds_name} Dataset – Validation Weighted Accuracy (Mean ± SE)")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            fname = os.path.join(
                working_dir, f"{ds_name.lower()}_val_weighted_acc_mean_se.png"
            )
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated validation-accuracy plot for {ds_name}: {e}")
        plt.close()

    # ----------- 3) Train vs Val CpxWA mean ± SE --------------------
    try:
        if all(train_metrics_runs) and all(val_metrics_runs):
            train_cpx_mean, train_cpx_se = mean_se(train_metrics_runs, "cpx")
            val_cpx_mean, val_cpx_se = mean_se(val_metrics_runs, "cpx")

            plt.figure()
            plt.plot(epochs, train_cpx_mean, color="C3", label="Train CpxWA Mean")
            plt.fill_between(
                epochs,
                train_cpx_mean - train_cpx_se,
                train_cpx_mean + train_cpx_se,
                color="C3",
                alpha=0.25,
                label="Train SE",
            )
            plt.plot(epochs, val_cpx_mean, color="C4", label="Val CpxWA Mean")
            plt.fill_between(
                epochs,
                val_cpx_mean - val_cpx_se,
                val_cpx_mean + val_cpx_se,
                color="C4",
                alpha=0.25,
                label="Val SE",
            )
            plt.title(f"{ds_name} Dataset – CpxWA Train vs Val (Mean ± SE)")
            plt.xlabel("Epoch")
            plt.ylabel("CpxWA")
            plt.legend()
            fname = os.path.join(
                working_dir, f"{ds_name.lower()}_cpxwa_train_val_mean_se.png"
            )
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated CpxWA plot for {ds_name}: {e}")
        plt.close()

    # ----------------------------------------------------------------
    # 3. Console summary of last epoch statistics
    try:
        last_idx = -1
        if all(train_losses_runs):
            print(
                f"{ds_name} – Final Train Loss: {train_mean[last_idx]:.4f} ± {train_se[last_idx]:.4f} (N={n_runs})"
            )
        if all(val_metrics_runs):
            print(
                f"{ds_name} – Final Val CpxWA: {cpx_mean[last_idx]:.4f} ± {cpx_se[last_idx]:.4f} (N={n_runs})"
            )
    except Exception as e:
        print(f"Error printing summary for {ds_name}: {e}")

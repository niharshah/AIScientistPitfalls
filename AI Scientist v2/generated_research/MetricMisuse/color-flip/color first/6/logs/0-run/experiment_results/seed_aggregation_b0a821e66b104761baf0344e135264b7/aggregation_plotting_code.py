import matplotlib.pyplot as plt
import numpy as np
import os

# Mandatory working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# 1) Load all experiment_data.npy files that were provided
# ------------------------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-31_03-29-17_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_6bb0b2be204e4b67b5402495313c9752_proc_1645238/experiment_data.npy",
    "experiments/2025-08-31_03-29-17_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_f0f85018621c4716a1c21a2c14bc520c_proc_1645241/experiment_data.npy",
    "experiments/2025-08-31_03-29-17_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_2920292c4ab74e04b905f4c7c39beb77_proc_1645239/experiment_data.npy",
]

all_experiment_data = []
try:
    for experiment_data_path in experiment_data_path_list:
        full_path = os.path.join(
            os.getenv("AI_SCIENTIST_ROOT", ""), experiment_data_path
        )
        exp_dict = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_dict)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# ------------------------------------------------------------------
# 2) Discover dataset names that exist across experiments
# ------------------------------------------------------------------
dataset_names = set()
for exp in all_experiment_data:
    dataset_names.update(exp.keys())

# ------------------------------------------------------------------
# 3) Aggregate and plot
# ------------------------------------------------------------------
summary_last_epoch = {}  # For printing at the end

for dataset in sorted(dataset_names):
    # Collect the list of experiments that actually contain this dataset
    relevant_exps = [exp for exp in all_experiment_data if dataset in exp]
    if len(relevant_exps) == 0:
        continue

    # Discover metric names that appear in every relevant experiment
    common_metric_names = None
    for exp in relevant_exps:
        metrics_here = set(exp[dataset].get("metrics", {}).keys())
        common_metric_names = (
            metrics_here
            if common_metric_names is None
            else common_metric_names & metrics_here
        )
    if not common_metric_names:
        continue

    # Limit to at most 5 metrics per dataset to obey guideline
    for metric_idx, metric_name in enumerate(sorted(common_metric_names)):
        if metric_idx >= 5:
            break

        # ------------------------------------------------------------------
        # Gather the metric curves across experiments
        # ------------------------------------------------------------------
        curves = []
        for exp in relevant_exps:
            curve = np.asarray(exp[dataset]["metrics"][metric_name]).astype(float)
            curves.append(curve)

        # Align lengths
        min_len = min(len(c) for c in curves)
        if min_len == 0:
            continue
        curves = np.stack(
            [c[:min_len] for c in curves], axis=0
        )  # Shape: (n_runs, min_len)

        # Mean and standard error
        mean_curve = curves.mean(axis=0)
        stderr_curve = curves.std(axis=0, ddof=1) / np.sqrt(curves.shape[0])

        # ------------------------------------------------------------------
        # Plot in its own try-except block
        # ------------------------------------------------------------------
        try:
            plt.figure()
            epochs = np.arange(min_len)
            plt.plot(epochs, mean_curve, label="Mean")
            plt.fill_between(
                epochs,
                mean_curve - stderr_curve,
                mean_curve + stderr_curve,
                color="blue",
                alpha=0.3,
                label="± SE",
            )
            plt.xlabel("Epoch")
            plt.ylabel(metric_name)
            plt.title(f"{dataset} – {metric_name} over Epochs (Mean ± SE)")
            plt.legend()
            file_name = f"{dataset}_{metric_name}_mean_stderr.png".replace(" ", "_")
            save_path = os.path.join(working_dir, file_name)
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"Error creating plot for {dataset}/{metric_name}: {e}")
            plt.close()

        # Store last epoch summary
        summary_last_epoch.setdefault(dataset, {})[metric_name] = mean_curve[-1]

# ------------------------------------------------------------------
# 4) Print final-epoch summary
# ------------------------------------------------------------------
for dset, metrics in summary_last_epoch.items():
    print(f"\nDataset: {dset}")
    for mname, val in metrics.items():
        print(f"  {mname} (final epoch mean): {val:.4f}")

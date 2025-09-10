import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------------------------------------------------
# Basic set-up
# ----------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------------------------------------------------
# 1. Load all experiment files
# ----------------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-31_14-12-02_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_10d72db1c0474992b6a6086500c8f17d_proc_1749406/experiment_data.npy",
    "experiments/2025-08-31_14-12-02_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_d84cde604ac34573886c5a5d6334202c_proc_1749405/experiment_data.npy",
    "experiments/2025-08-31_14-12-02_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_667337c1d19747d08e6725989e13b910_proc_1749408/experiment_data.npy",
]

all_experiment_data = []
for path in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), path)
        exp = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp)
    except Exception as e:
        print(f"Error loading {path}: {e}")

if not all_experiment_data:
    print("No experiment data loaded. Nothing to plot.")
    exit()

# ----------------------------------------------------------
# 2. Re-organise information by dataset
# ----------------------------------------------------------
dataset_bucket = {}
for exp in all_experiment_data:
    for run_key, run_val in exp.items():
        for ds_key, logs in run_val.items():
            bucket = dataset_bucket.setdefault(
                ds_key,
                {"train_loss": [], "val_loss": [], "val_metrics": []},
            )
            bucket["train_loss"].append(np.array(logs["losses"]["train"]))
            bucket["val_loss"].append(np.array(logs["losses"]["val"]))
            bucket["val_metrics"].append(np.array(logs["metrics"]["val"]))

# ----------------------------------------------------------
# 3. Create plots per dataset
# ----------------------------------------------------------
for ds_key, data in dataset_bucket.items():
    # ------------------------------------------------------
    # 3a. Aggregate LOSS curves
    # ------------------------------------------------------
    try:
        train_runs = data["train_loss"]
        val_runs = data["val_loss"]
        if not train_runs or not val_runs:
            raise ValueError("No loss data found")

        min_len = min(arr.shape[0] for arr in train_runs + val_runs)
        epochs = train_runs[0][:min_len, 0]

        # Stack values
        train_stack = np.stack([arr[:min_len, 1] for arr in train_runs])
        val_stack = np.stack([arr[:min_len, 1] for arr in val_runs])

        # Mean and SE
        train_mean = train_stack.mean(0)
        train_se = train_stack.std(0, ddof=1) / np.sqrt(train_stack.shape[0])

        val_mean = val_stack.mean(0)
        val_se = val_stack.std(0, ddof=1) / np.sqrt(val_stack.shape[0])

        plt.figure()
        plt.plot(epochs, train_mean, label="Train loss (mean)")
        plt.fill_between(
            epochs, train_mean - train_se, train_mean + train_se, alpha=0.2
        )

        plt.plot(epochs, val_mean, label="Val loss (mean)")
        plt.fill_between(epochs, val_mean - val_se, val_mean + val_se, alpha=0.2)

        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy")
        plt.title(f"{ds_key} – Aggregated Loss Curves\nMean ± SE across runs")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_key}_loss_mean_se.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {ds_key}: {e}")
        plt.close()

    # ------------------------------------------------------
    # 3b. Aggregate VALIDATION METRICS
    # ------------------------------------------------------
    try:
        metrics_runs = data["val_metrics"]
        if not metrics_runs:
            raise ValueError("No validation metric data found")

        min_len = min(arr.shape[0] for arr in metrics_runs)
        epochs = metrics_runs[0][:min_len, 0]

        metric_names = ["CWA", "SWA", "HM"]
        colors = ["tab:blue", "tab:orange", "tab:green"]

        plt.figure()
        for idx, (mname, color) in enumerate(zip(metric_names, colors), start=1):
            stack = np.stack([arr[:min_len, idx] for arr in metrics_runs])
            mean = stack.mean(0)
            se = stack.std(0, ddof=1) / np.sqrt(stack.shape[0])

            plt.plot(epochs, mean, label=f"{mname} (mean)", color=color)
            plt.fill_between(
                epochs,
                mean - se,
                mean + se,
                alpha=0.2,
                color=color,
            )

        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(f"{ds_key} – Aggregated Validation Metrics\nMean ± SE across runs")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_key}_val_metrics_mean_se.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated metrics plot for {ds_key}: {e}")
        plt.close()

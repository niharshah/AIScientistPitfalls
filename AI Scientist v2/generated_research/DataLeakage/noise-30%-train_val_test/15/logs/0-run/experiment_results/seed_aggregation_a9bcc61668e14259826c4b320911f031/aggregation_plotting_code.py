import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------
# Basic setup
# ---------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------
# Load all experiment_data dicts
# ---------------------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-17_23-44-27_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_86eeacd8dfc7482484902d40d518f91e_proc_3458578/experiment_data.npy",
    "experiments/2025-08-17_23-44-27_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_8b2d89fd83c54ef090340e2cdd92729c_proc_3458580/experiment_data.npy",
    "None/experiment_data.npy",
]
all_exp_data = []
for p in experiment_data_path_list:
    try:
        abs_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp_dict = np.load(abs_path, allow_pickle=True).item()
        all_exp_data.append(exp_dict)
    except Exception as e:
        print(f"Error loading {p}: {e}")

# ---------------------------------------------------------------
# Aggregate per dataset
# ---------------------------------------------------------------
for dataset_name in set(
    k for exp in all_exp_data for k in exp.keys()
):  # iterate over unique dataset names
    # gather per-run arrays; skip run if missing keys
    tr_loss_list, val_loss_list = [], []
    tr_f1_list, val_f1_list = [], []
    test_metric_list = []

    for exp in all_exp_data:
        if dataset_name not in exp:
            continue
        ds = exp[dataset_name]
        try:
            tr_loss_list.append(np.asarray(ds["losses"]["train"]))
            val_loss_list.append(np.asarray(ds["losses"]["val"]))
            tr_f1_list.append(np.asarray(ds["metrics"]["train"]))
            val_f1_list.append(np.asarray(ds["metrics"]["val"]))
            if "test_macroF1" in ds:
                test_metric_list.append(float(ds["test_macroF1"]))
        except Exception as e:
            print(f"Skipping run for {dataset_name} due to missing keys: {e}")

    # Need at least 2 runs to show error bars
    n_runs = len(tr_loss_list)
    if n_runs == 0:
        continue

    # -----------------------------------------------------------
    # Align to common epoch length (min length across runs)
    # -----------------------------------------------------------
    min_len = min(len(x) for x in tr_loss_list)
    tr_loss_arr = np.stack([x[:min_len] for x in tr_loss_list])
    val_loss_arr = np.stack([x[:min_len] for x in val_loss_list])
    tr_f1_arr = np.stack([x[:min_len] for x in tr_f1_list])
    val_f1_arr = np.stack([x[:min_len] for x in val_f1_list])
    epochs = np.arange(1, min_len + 1)

    # -----------------------------------------------------------
    # Compute mean and standard error
    # -----------------------------------------------------------
    def mean_se(arr):
        mean = arr.mean(axis=0)
        se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        return mean, se

    tr_loss_mean, tr_loss_se = mean_se(tr_loss_arr)
    val_loss_mean, val_loss_se = mean_se(val_loss_arr)
    tr_f1_mean, tr_f1_se = mean_se(tr_f1_arr)
    val_f1_mean, val_f1_se = mean_se(val_f1_arr)

    # -----------------------------------------------------------
    # Plot 1: aggregated loss
    # -----------------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, tr_loss_mean, label="Train Loss (mean)")
        plt.fill_between(
            epochs,
            tr_loss_mean - tr_loss_se,
            tr_loss_mean + tr_loss_se,
            alpha=0.3,
            label="Train Loss ± SE",
        )
        plt.plot(epochs, val_loss_mean, label="Val Loss (mean)")
        plt.fill_between(
            epochs,
            val_loss_mean - val_loss_se,
            val_loss_mean + val_loss_se,
            alpha=0.3,
            label="Val Loss ± SE",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dataset_name}: Aggregated Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        fname = f"{dataset_name}_agg_loss_curve.png".replace(os.sep, "_")
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss for {dataset_name}: {e}")
        plt.close()

    # -----------------------------------------------------------
    # Plot 2: aggregated macro-F1
    # -----------------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, tr_f1_mean, label="Train Macro-F1 (mean)")
        plt.fill_between(
            epochs,
            tr_f1_mean - tr_f1_se,
            tr_f1_mean + tr_f1_se,
            alpha=0.3,
            label="Train Macro-F1 ± SE",
        )
        plt.plot(epochs, val_f1_mean, label="Val Macro-F1 (mean)")
        plt.fill_between(
            epochs,
            val_f1_mean - val_f1_se,
            val_f1_mean + val_f1_se,
            alpha=0.3,
            label="Val Macro-F1 ± SE",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title(f"{dataset_name}: Aggregated Training vs Validation Macro-F1")
        plt.legend()
        plt.tight_layout()
        fname = f"{dataset_name}_agg_macroF1_curve.png".replace(os.sep, "_")
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated macro-F1 for {dataset_name}: {e}")
        plt.close()

    # -----------------------------------------------------------
    # Print aggregated test metric
    # -----------------------------------------------------------
    if test_metric_list:
        test_arr = np.asarray(test_metric_list)
        test_mean = test_arr.mean()
        test_se = test_arr.std(ddof=1) / np.sqrt(len(test_arr))
        print(
            f"{dataset_name} Test Macro-F1 (mean ± SE over {len(test_arr)} runs): "
            f"{test_mean:.4f} ± {test_se:.4f}"
        )

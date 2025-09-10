import matplotlib.pyplot as plt
import numpy as np
import os
from math import sqrt

# ---------------------------------------------------------
# Prepare working directory
# ---------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------
# Paths to all experiment_data.npy files (provided)
# ---------------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-17_23-44-27_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_d7213184e2734b10b3b559c2186faf68_proc_3469034/experiment_data.npy",
    "experiments/2025-08-17_23-44-27_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_357ab890370d42178e3bcfa1e9dda187_proc_3469032/experiment_data.npy",
    "experiments/2025-08-17_23-44-27_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_60f7274c4ba14a2fb3b2b0f5fdb3912b_proc_3469033/experiment_data.npy",
]

# ---------------------------------------------------------
# Load every experiment file
# ---------------------------------------------------------
all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp_data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# ---------------------------------------------------------
# Aggregate runs by dataset
# ---------------------------------------------------------
datasets_runs = {}  # {ds_name: [run_dict1, run_dict2, ...]}
for run in all_experiment_data:
    for ds_name, ds_dict in run.items():
        datasets_runs.setdefault(ds_name, []).append(ds_dict)

# Will be used for cross-dataset comparison
cross_final_val_mean = {}
cross_final_val_se = {}

# ---------------------------------------------------------
# Iterate over datasets and aggregate plots
# ---------------------------------------------------------
for ds_name, run_list in datasets_runs.items():
    # Collect arrays for each metric
    train_loss_curves, val_loss_curves = [], []
    train_f1_curves, val_f1_curves = [], []

    for r in run_list:
        # Retrieve metrics if present
        epochs = r.get("epochs", [])
        losses = r.get("losses", {})
        metrics = r.get("metrics", {})

        if epochs and losses.get("train") and losses.get("val"):
            min_len = min(len(epochs), len(losses["train"]), len(losses["val"]))
            train_loss_curves.append(np.array(losses["train"][:min_len]))
            val_loss_curves.append(np.array(losses["val"][:min_len]))

        if epochs and metrics.get("train") and metrics.get("val"):
            min_len = min(len(epochs), len(metrics["train"]), len(metrics["val"]))
            train_f1_curves.append(np.array(metrics["train"][:min_len]))
            val_f1_curves.append(np.array(metrics["val"][:min_len]))

    # -----------------------------------------------------
    # Aggregated LOSS curve
    # -----------------------------------------------------
    try:
        if train_loss_curves and val_loss_curves:
            # Align all curves to the shortest one
            min_len = min([c.shape[0] for c in train_loss_curves + val_loss_curves])
            train_stack = np.vstack([c[:min_len] for c in train_loss_curves])
            val_stack = np.vstack([c[:min_len] for c in val_loss_curves])
            epochs_axis = np.arange(1, min_len + 1)

            tr_mean = train_stack.mean(axis=0)
            tr_se = train_stack.std(axis=0) / sqrt(train_stack.shape[0])
            val_mean = val_stack.mean(axis=0)
            val_se = val_stack.std(axis=0) / sqrt(val_stack.shape[0])

            plt.figure()
            plt.plot(epochs_axis, tr_mean, label="Train Loss (mean)")
            plt.fill_between(
                epochs_axis,
                tr_mean - tr_se,
                tr_mean + tr_se,
                alpha=0.3,
                label="Train ±1SE",
            )
            plt.plot(epochs_axis, val_mean, label="Validation Loss (mean)")
            plt.fill_between(
                epochs_axis,
                val_mean - val_se,
                val_mean + val_se,
                alpha=0.3,
                label="Val ±1SE",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{ds_name}: Aggregated Training vs Validation Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, f"{ds_name}_aggregated_loss_curve.png")
            )
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {ds_name}: {e}")
        plt.close()

    # -----------------------------------------------------
    # Aggregated Macro-F1 curve
    # -----------------------------------------------------
    try:
        if train_f1_curves and val_f1_curves:
            min_len = min([c.shape[0] for c in train_f1_curves + val_f1_curves])
            train_stack = np.vstack([c[:min_len] for c in train_f1_curves])
            val_stack = np.vstack([c[:min_len] for c in val_f1_curves])
            epochs_axis = np.arange(1, min_len + 1)

            tr_mean = train_stack.mean(axis=0)
            tr_se = train_stack.std(axis=0) / sqrt(train_stack.shape[0])
            val_mean = val_stack.mean(axis=0)
            val_se = val_stack.std(axis=0) / sqrt(val_stack.shape[0])

            plt.figure()
            plt.plot(epochs_axis, tr_mean, label="Train Macro-F1 (mean)")
            plt.fill_between(
                epochs_axis,
                tr_mean - tr_se,
                tr_mean + tr_se,
                alpha=0.3,
                label="Train ±1SE",
            )
            plt.plot(epochs_axis, val_mean, label="Validation Macro-F1 (mean)")
            plt.fill_between(
                epochs_axis,
                val_mean - val_se,
                val_mean + val_se,
                alpha=0.3,
                label="Val ±1SE",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Macro-F1")
            plt.title(f"{ds_name}: Aggregated Training vs Validation Macro-F1")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, f"{ds_name}_aggregated_macroF1_curve.png")
            )
            plt.close()

            # store final val stats for cross-dataset bar chart
            cross_final_val_mean[ds_name] = val_stack[:, -1].mean()
            cross_final_val_se[ds_name] = val_stack[:, -1].std() / sqrt(
                val_stack.shape[0]
            )
    except Exception as e:
        print(f"Error creating aggregated f1 plot for {ds_name}: {e}")
        plt.close()

# ---------------------------------------------------------
# Cross-dataset comparison bar chart (final val Macro-F1)
# ---------------------------------------------------------
try:
    if cross_final_val_mean:
        plt.figure(figsize=(6, max(2, len(cross_final_val_mean) * 0.4)))
        names = list(cross_final_val_mean.keys())
        means = np.array([cross_final_val_mean[n] for n in names])
        errors = np.array([cross_final_val_se[n] for n in names])

        y_pos = np.arange(len(names))
        plt.barh(
            y_pos,
            means,
            xerr=errors,
            align="center",
            color="skyblue",
            ecolor="darkblue",
            capsize=4,
        )
        plt.yticks(y_pos, names)
        plt.xlabel("Final Validation Macro-F1 (mean ±1SE)")
        plt.title("Aggregated Final Validation Macro-F1 Across Datasets")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "cross_dataset_val_macroF1_aggregated.png")
        )
        plt.close()
except Exception as e:
    print(f"Error creating cross-dataset comparison plot: {e}")
    plt.close()

import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- experiment paths (given) ----------
experiment_data_path_list = [
    "experiments/2025-08-14_12-19-19_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_1b55575e425e48c88f63a6e9c87c5931_proc_2637981/experiment_data.npy",
    "experiments/2025-08-14_12-19-19_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_56303e46285f4a98b79180b762ff0b61_proc_2637983/experiment_data.npy",
    "experiments/2025-08-14_12-19-19_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_051bc24ed0c64cd2a1b71a398f4cf8d0_proc_2637982/experiment_data.npy",
]

# ---------- load data ----------
all_experiment_data = []
for path in experiment_data_path_list:
    try:
        exp_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), path), allow_pickle=True
        ).item()
        all_experiment_data.append(exp_data)
    except Exception as e:
        print(f"Error loading {path}: {e}")


# ---------- helper: aggregate arrays ----------
def stack_and_trim(list_of_arrays):
    """
    Stack 1-D arrays of possibly different length by trimming to min length.
    Returns stacked 2-D np.array shape (runs, epochs).
    """
    min_len = min(len(arr) for arr in list_of_arrays)
    trimmed = [arr[:min_len] for arr in list_of_arrays]
    return np.stack(trimmed, axis=0)


# ---------- gather union of dataset names ----------
dataset_names = set()
for exp in all_experiment_data:
    dataset_names.update(exp.keys())

# ---------- iterate per dataset ----------
for ds_name in dataset_names:
    # collect per-run data for this dataset
    per_run_data = []
    for exp in all_experiment_data:
        if ds_name in exp:
            per_run_data.append(exp[ds_name])

    # ---------- 1. Loss curves (mean ± stderr) ----------
    try:
        train_curves = []
        val_curves = []
        for run in per_run_data:
            if "losses" in run and run["losses"]:
                train_curves.append(np.asarray(run["losses"]["train"]))
                val_curves.append(np.asarray(run["losses"]["val"]))
        if train_curves and val_curves:
            train_stack = stack_and_trim(train_curves)
            val_stack = stack_and_trim(val_curves)
            epochs = np.arange(train_stack.shape[1])

            train_mean, train_stderr = train_stack.mean(0), train_stack.std(
                0
            ) / np.sqrt(train_stack.shape[0])
            val_mean, val_stderr = val_stack.mean(0), val_stack.std(0) / np.sqrt(
                val_stack.shape[0]
            )

            plt.figure()
            plt.plot(epochs, train_mean, label="train mean")
            plt.fill_between(
                epochs,
                train_mean - train_stderr,
                train_mean + train_stderr,
                alpha=0.3,
                label="train ± stderr",
            )
            plt.plot(epochs, val_mean, linestyle="--", label="val mean")
            plt.fill_between(
                epochs,
                val_mean - val_stderr,
                val_mean + val_stderr,
                alpha=0.3,
                label="val ± stderr",
            )
            plt.title(f"{ds_name} Loss Curves (Aggregated)\nMean ± Standard Error")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy")
            plt.legend(fontsize=7)
            fname = f"{ds_name}_agg_loss_curves.png"
            plt.savefig(os.path.join(working_dir, fname))
        else:
            raise ValueError("loss curves not found in any run")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss curves for {ds_name}: {e}")
        plt.close()

    # ---------- 2. Accuracy curves ----------
    try:
        train_curves = []
        val_curves = []
        for run in per_run_data:
            if "metrics" in run and run["metrics"]:
                train_curves.append(np.asarray(run["metrics"]["train"]))
                val_curves.append(np.asarray(run["metrics"]["val"]))
        if train_curves and val_curves:
            train_stack = stack_and_trim(train_curves)
            val_stack = stack_and_trim(val_curves)
            epochs = np.arange(train_stack.shape[1])

            train_mean, train_stderr = train_stack.mean(0), train_stack.std(
                0
            ) / np.sqrt(train_stack.shape[0])
            val_mean, val_stderr = val_stack.mean(0), val_stack.std(0) / np.sqrt(
                val_stack.shape[0]
            )

            plt.figure()
            plt.plot(epochs, train_mean, label="train mean")
            plt.fill_between(
                epochs,
                train_mean - train_stderr,
                train_mean + train_stderr,
                alpha=0.3,
                label="train ± stderr",
            )
            plt.plot(epochs, val_mean, linestyle="--", label="val mean")
            plt.fill_between(
                epochs,
                val_mean - val_stderr,
                val_mean + val_stderr,
                alpha=0.3,
                label="val ± stderr",
            )
            plt.title(f"{ds_name} Accuracy Curves (Aggregated)\nMean ± Standard Error")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend(fontsize=7)
            fname = f"{ds_name}_agg_accuracy_curves.png"
            plt.savefig(os.path.join(working_dir, fname))
        else:
            raise ValueError("accuracy curves not found")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated accuracy curves for {ds_name}: {e}")
        plt.close()

    # ---------- 3. Shape-Weighted Accuracy curves ----------
    try:
        train_curves = []
        val_curves = []
        for run in per_run_data:
            if "swa" in run and run["swa"]:
                train_curves.append(np.asarray(run["swa"]["train"]))
                val_curves.append(np.asarray(run["swa"]["val"]))
        if train_curves and val_curves:
            train_stack = stack_and_trim(train_curves)
            val_stack = stack_and_trim(val_curves)
            epochs = np.arange(train_stack.shape[1])

            train_mean, train_stderr = train_stack.mean(0), train_stack.std(
                0
            ) / np.sqrt(train_stack.shape[0])
            val_mean, val_stderr = val_stack.mean(0), val_stack.std(0) / np.sqrt(
                val_stack.shape[0]
            )

            plt.figure()
            plt.plot(epochs, train_mean, label="train mean")
            plt.fill_between(
                epochs,
                train_mean - train_stderr,
                train_mean + train_stderr,
                alpha=0.3,
                label="train ± stderr",
            )
            plt.plot(epochs, val_mean, linestyle="--", label="val mean")
            plt.fill_between(
                epochs,
                val_mean - val_stderr,
                val_mean + val_stderr,
                alpha=0.3,
                label="val ± stderr",
            )
            plt.title(f"{ds_name} SWA Curves (Aggregated)\nMean ± Standard Error")
            plt.xlabel("Epoch")
            plt.ylabel("Shape-Weighted Accuracy")
            plt.legend(fontsize=7)
            fname = f"{ds_name}_agg_swa_curves.png"
            plt.savefig(os.path.join(working_dir, fname))
        else:
            raise ValueError("swa curves not found")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated SWA curves for {ds_name}: {e}")
        plt.close()

    # ---------- 4. Final Test Metrics bar plot ----------
    try:
        metrics_list = {"loss": [], "acc": [], "swa": []}
        for run in per_run_data:
            if "test_metrics" in run:
                for k in metrics_list.keys():
                    if k in run["test_metrics"]:
                        metrics_list[k].append(run["test_metrics"][k])
        if any(len(v) for v in metrics_list.values()):
            bars = list(metrics_list.keys())
            means = [
                np.mean(metrics_list[k]) if metrics_list[k] else np.nan for k in bars
            ]
            stderr = [
                (
                    np.std(metrics_list[k]) / np.sqrt(len(metrics_list[k]))
                    if metrics_list[k]
                    else 0
                )
                for k in bars
            ]
            x = np.arange(len(bars))
            plt.figure()
            plt.bar(x, means, yerr=stderr, capsize=5, color="skyblue")
            plt.xticks(x, bars)
            plt.ylabel("Score")
            plt.title(f"{ds_name} Final Test Metrics\nMean ± Standard Error")
            fname = f"{ds_name}_agg_test_metrics.png"
            plt.savefig(os.path.join(working_dir, fname))
        else:
            raise ValueError("no test metrics found")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated test metrics for {ds_name}: {e}")
        plt.close()

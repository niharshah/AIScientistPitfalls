import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------------------------
import matplotlib

matplotlib.rcParams.update({"figure.max_open_warning": 0})

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# All experiment files to aggregate
experiment_data_path_list = [
    "experiments/2025-08-17_02-43-40_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_b729f3272edc4e1e99a903c7218a2c0f_proc_3198317/experiment_data.npy",
    "experiments/2025-08-17_02-43-40_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_39f76a7592064842840ca3a2981390e5_proc_3198319/experiment_data.npy",
    "experiments/2025-08-17_02-43-40_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_befe90ecadd74cdbbf6d6ab06c34bbbe_proc_3198318/experiment_data.npy",
]

all_runs = []
root = os.getenv("AI_SCIENTIST_ROOT", "")
for p in experiment_data_path_list:
    try:
        data = np.load(os.path.join(root, p), allow_pickle=True).item()
        all_runs.append(data)
    except Exception as e:
        print(f"Error loading experiment data {p}: {e}")


# ------------------------------------------------------------------
# Helper functions
def confusion_matrix_np(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def collect_series(run_list, ds_name, outer_key, inner_key):
    """Return stacked np.array shape (n_runs, min_len) for a series present in every run."""
    series_list = []
    for run in run_list:
        if ds_name not in run:  # dataset missing in that run
            return None
        container = run[ds_name].get(outer_key, {})
        if inner_key not in container:
            return None
        series_list.append(np.asarray(container[inner_key]))
    if not series_list:
        return None
    # truncate to shortest length
    min_len = min(len(s) for s in series_list)
    series_list = [s[:min_len] for s in series_list]
    return np.vstack(series_list)


def aggregate_and_plot(ds_name, run_list):
    # --------------- LOSS CURVES -----------------------------------
    for outer, left_key, right_key, ylabel, ftag in [
        ("losses", "train", "val", "Loss", "loss"),
        ("metrics", "train_acc", "val_acc", "Accuracy", "accuracy"),
    ]:
        try:
            left_mat = collect_series(run_list, ds_name, outer, left_key)
            right_mat = collect_series(run_list, ds_name, outer, right_key)
            if left_mat is None or right_mat is None:
                continue
            epochs = np.arange(1, left_mat.shape[1] + 1)
            n_runs = left_mat.shape[0]

            left_mean, left_sem = left_mat.mean(0), left_mat.std(0, ddof=1) / np.sqrt(
                n_runs
            )
            right_mean, right_sem = right_mat.mean(0), right_mat.std(
                0, ddof=1
            ) / np.sqrt(n_runs)

            plt.figure()
            plt.plot(epochs, left_mean, label=f"Train mean ({n_runs} runs)")
            plt.fill_between(
                epochs,
                left_mean - left_sem,
                left_mean + left_sem,
                alpha=0.3,
                label="Train ± SEM",
            )
            plt.plot(epochs, right_mean, label="Val mean")
            plt.fill_between(
                epochs,
                right_mean - right_sem,
                right_mean + right_sem,
                alpha=0.3,
                label="Val ± SEM",
            )
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.title(f"{ds_name} {ylabel} (mean ± SEM)\nLeft: Train, Right: Val")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name}_{ftag}_agg.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating aggregated {ftag} plot for {ds_name}: {e}")
            plt.close()

    # --------------- RULE FIDELITY -----------------------------------
    try:
        rf_mat = collect_series(run_list, ds_name, "metrics", "rule_fidelity")
        if rf_mat is not None:
            epochs = np.arange(1, rf_mat.shape[1] + 1)
            n_runs = rf_mat.shape[0]
            rf_mean = rf_mat.mean(0)
            rf_sem = rf_mat.std(0, ddof=1) / np.sqrt(n_runs)

            plt.figure()
            plt.plot(epochs, rf_mean, marker="o", label=f"Mean ({n_runs} runs)")
            plt.fill_between(
                epochs, rf_mean - rf_sem, rf_mean + rf_sem, alpha=0.3, label="± SEM"
            )
            plt.xlabel("Epoch")
            plt.ylabel("Fidelity")
            plt.title(f"{ds_name} Rule Fidelity (mean ± SEM)")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name}_rule_fidelity_agg.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated rule fidelity plot for {ds_name}: {e}")
        plt.close()

    # --------------- CONFUSION MATRIX -----------------------------------
    try:
        # collect predictions and gt
        cm_total = None
        for run in run_list:
            ds_dict = run.get(ds_name, {})
            preds = ds_dict.get("predictions", None)
            gt = ds_dict.get("ground_truth", None)
            if preds is None or gt is None:
                cm_total = None
                break
            num_classes = len(np.unique(gt))
            cm = confusion_matrix_np(gt, preds, num_classes)
            cm_total = cm if cm_total is None else cm_total + cm
        if cm_total is not None:
            cm_percent = cm_total / cm_total.sum() * 100.0
            plt.figure(figsize=(4, 4))
            plt.imshow(cm_percent, cmap="Blues")
            plt.colorbar(label="%")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(
                f"{ds_name} Confusion Matrix\nAggregated over {len(run_list)} runs"
            )
            for i in range(num_classes):
                for j in range(num_classes):
                    plt.text(
                        j,
                        i,
                        f"{cm_percent[i, j]:.1f}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=8,
                    )
            fname = os.path.join(working_dir, f"{ds_name}_confusion_matrix_agg.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated confusion matrix for {ds_name}: {e}")
        plt.close()

    # --------------- FINAL TEST ACCURACY -----------------------------------
    try:
        acc_list = []
        for run in run_list:
            ds_dict = run.get(ds_name, {})
            preds = ds_dict.get("predictions", None)
            gt = ds_dict.get("ground_truth", None)
            if preds is not None and gt is not None:
                acc_list.append((preds == gt).mean())
        if acc_list:
            acc_arr = np.asarray(acc_list)
            mean_acc = acc_arr.mean()
            sem_acc = acc_arr.std(ddof=1) / np.sqrt(len(acc_arr))
            print(
                f"{ds_name} test accuracy: {mean_acc:.3f} ± {sem_acc:.3f} (SEM) over {len(acc_arr)} runs"
            )
    except Exception as e:
        print(f"Error computing aggregated accuracy for {ds_name}: {e}")


# ------------------------------------------------------------------
# Group runs by dataset and plot
datasets = set()
for r in all_runs:
    datasets.update(r.keys())

for ds in datasets:
    runs_with_ds = [r for r in all_runs if ds in r]
    if runs_with_ds:
        aggregate_and_plot(ds, runs_with_ds)

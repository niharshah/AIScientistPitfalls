import matplotlib.pyplot as plt
import numpy as np
import os
from math import sqrt

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- experiment paths ----------
experiment_data_path_list = [
    "experiments/2025-08-14_23-40-39_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_313b21f24c984907af3035ec4db733a8_proc_2780780/experiment_data.npy",
    "experiments/2025-08-14_23-40-39_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_f05e7c1370ce4f12b8bf8093cf6d7120_proc_2780781/experiment_data.npy",
    # the “None/…” entry is ignored if it does not exist
]

# ---------- load all runs ----------
all_experiment_data = []
for p in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        if not os.path.isfile(full_path):
            continue
        exp_d = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_d)
    except Exception as e:
        print(f"Error loading experiment data from {p}: {e}")

if not all_experiment_data:
    print("No experiment files loaded – nothing to plot.")
    exit()


# ---------- helper ----------
def downsample(arr, max_pts=200):
    if len(arr) <= max_pts:
        return np.arange(len(arr)), arr
    idx = np.linspace(0, len(arr) - 1, max_pts, dtype=int)
    return idx, np.array(arr)[idx]


def aggregate_time_series(series_list):
    """Pad each 1-D array with np.nan to the maximum length, return mean & sem."""
    max_len = max(len(s) for s in series_list)
    data = np.full((len(series_list), max_len), np.nan)
    for r, s in enumerate(series_list):
        data[r, : len(s)] = s
    mean = np.nanmean(data, axis=0)
    sem = np.nanstd(data, axis=0) / np.sqrt(np.sum(~np.isnan(data), axis=0))
    return mean, sem


# ---------- collect all datasets ----------
dataset_names = set()
for run_data in all_experiment_data:
    dataset_names.update(run_data.keys())

best_swa_summary = {}  # dataset -> list of best swa values across runs

# ---------- per-dataset plots ----------
for ds_name in dataset_names:
    # collect metrics from each run that contains this dataset
    train_losses, val_losses, val_swas = [], [], []
    for run_data in all_experiment_data:
        ds_dict = run_data.get(ds_name)
        if ds_dict is None:
            continue
        metrics = ds_dict.get("metrics", {})
        if "train_loss" in metrics:
            train_losses.append(np.asarray(metrics["train_loss"], dtype=float))
        if "val_loss" in metrics:
            val_losses.append(np.asarray(metrics["val_loss"], dtype=float))
        if "val_swa" in metrics:
            val_swas.append(np.asarray(metrics["val_swa"], dtype=float))
            best_swa_summary.setdefault(ds_name, []).append(
                np.nanmax(metrics["val_swa"])
            )

    # ----- 1. aggregated loss curves -----
    try:
        if train_losses and val_losses:
            mean_train, sem_train = aggregate_time_series(train_losses)
            mean_val, sem_val = aggregate_time_series(val_losses)
            ep = np.arange(1, len(mean_train) + 1)
            idx, _ = downsample(mean_train)  # reuse for val so epochs line up
            plt.figure()
            plt.plot(ep[idx], mean_train[idx], label="Train Loss (mean)")
            plt.fill_between(
                ep[idx],
                (mean_train - sem_train)[idx],
                (mean_train + sem_train)[idx],
                alpha=0.3,
                label="Train SEM",
            )
            plt.plot(ep[idx], mean_val[idx], label="Val Loss (mean)")
            plt.fill_between(
                ep[idx],
                (mean_val - sem_val)[idx],
                (mean_val + sem_val)[idx],
                alpha=0.3,
                label="Val SEM",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(
                f"{ds_name}: Aggregated Training vs Validation Loss\n(Mean ± SEM over {len(train_losses)} runs)"
            )
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name}_agg_loss_curves.png")
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {ds_name}: {e}")
        plt.close()

    # ----- 2. aggregated validation SWA -----
    try:
        if val_swas:
            mean_swa, sem_swa = aggregate_time_series(val_swas)
            ep = np.arange(1, len(mean_swa) + 1)
            idx, _ = downsample(mean_swa)
            plt.figure()
            plt.plot(ep[idx], mean_swa[idx], label="Val SWA (mean)")
            plt.fill_between(
                ep[idx],
                (mean_swa - sem_swa)[idx],
                (mean_swa + sem_swa)[idx],
                alpha=0.3,
                label="Val SWA SEM",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Shape-Weighted Accuracy")
            plt.title(
                f"{ds_name}: Aggregated Validation SWA\n(Mean ± SEM over {len(val_swas)} runs)"
            )
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name}_agg_val_swa.png")
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated SWA plot for {ds_name}: {e}")
        plt.close()

# ---------- 3. dataset comparison of best SWA (bar plot) ----------
try:
    if best_swa_summary:
        names = sorted(best_swa_summary.keys())
        means = [np.mean(best_swa_summary[n]) for n in names]
        sems = [
            np.std(best_swa_summary[n]) / sqrt(len(best_swa_summary[n])) for n in names
        ]
        plt.figure()
        plt.bar(range(len(names)), means, yerr=sems, capsize=5)
        plt.xticks(range(len(names)), names, rotation=45, ha="right")
        plt.ylabel("Best Validation SWA")
        plt.title("Dataset Comparison: Best Validation SWA (Mean ± SEM across runs)")
        plt.tight_layout()
        fname = os.path.join(working_dir, "datasets_best_val_swa_comparison_agg.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated cross-dataset comparison plot: {e}")
    plt.close()

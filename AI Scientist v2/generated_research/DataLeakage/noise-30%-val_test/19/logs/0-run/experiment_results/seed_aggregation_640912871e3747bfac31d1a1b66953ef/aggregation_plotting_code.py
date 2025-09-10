import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------
# 1. load every experiment_data.npy that is present
# -------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-17_18-48-09_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_d22e800bef6143118436c899d34255df_proc_3327803/experiment_data.npy",
    "experiments/2025-08-17_18-48-09_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_4a029c19df794be3b3d7c5345fe36f20_proc_3327804/experiment_data.npy",
    "experiments/2025-08-17_18-48-09_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_f5defbc7deb84f229b32c622ab3dbc8f_proc_3327801/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        exp = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_experiment_data.append(exp)
    except Exception as e:
        print(f"Error loading {p}: {e}")

# -------------------------------------------------
# 2. aggregate keys (datasets) that appear anywhere
# -------------------------------------------------
all_datasets = set()
for exp in all_experiment_data:
    all_datasets.update(exp.keys())


# Helper to stack variable-length lists into 2-D array with NaNs
def stack_with_nan(list_of_lists):
    if not list_of_lists:
        return np.empty((0, 0))
    max_len = max(len(x) for x in list_of_lists)
    arr = np.full((len(list_of_lists), max_len), np.nan, dtype=float)
    for i, seq in enumerate(list_of_lists):
        arr[i, : len(seq)] = seq
    return arr


# -------------------------------------------------
# 3. create aggregate plots
# -------------------------------------------------
for dset in all_datasets:
    # collect per-run sequences
    epochs_list, tr_loss_list, val_loss_list = [], [], []
    tr_f1_list, val_f1_list = [], []
    for exp in all_experiment_data:
        if dset not in exp:
            continue
        info = exp[dset]
        epochs_list.append(info.get("epochs", []))
        tr_loss_list.append(info.get("losses", {}).get("train", []))
        val_loss_list.append(info.get("losses", {}).get("val", []))
        tr_f1_list.append(info.get("metrics", {}).get("train_macro_f1", []))
        val_f1_list.append(info.get("metrics", {}).get("val_macro_f1", []))

    # skip if no runs actually carried data
    if not tr_loss_list:
        print(f"No data for dataset {dset}, skipping.")
        continue

    # convert to arrays with NaN padding
    tr_loss_arr = stack_with_nan(tr_loss_list)
    val_loss_arr = stack_with_nan(val_loss_list)
    tr_f1_arr = stack_with_nan(tr_f1_list)
    val_f1_arr = stack_with_nan(val_f1_list)

    # common epoch index (assume 0..n-1)
    max_epoch = tr_loss_arr.shape[1] if tr_loss_arr.size else 0
    epoch_axis = np.arange(max_epoch)

    # mean & standard error (ignoring NaNs)
    def mean_sem(a):
        mean = np.nanmean(a, axis=0)
        sem = np.nanstd(a, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(a), axis=0))
        return mean, sem

    tr_loss_mean, tr_loss_sem = mean_sem(tr_loss_arr)
    val_loss_mean, val_loss_sem = mean_sem(val_loss_arr)
    tr_f1_mean, tr_f1_sem = mean_sem(tr_f1_arr)
    val_f1_mean, val_f1_sem = mean_sem(val_f1_arr)

    # --------------- aggregated loss plot ----------------
    try:
        plt.figure()
        plt.plot(epoch_axis, tr_loss_mean, label="Train mean")
        plt.fill_between(
            epoch_axis,
            tr_loss_mean - tr_loss_sem,
            tr_loss_mean + tr_loss_sem,
            alpha=0.3,
            label="Train ±1 SEM",
        )
        plt.plot(epoch_axis, val_loss_mean, label="Val mean")
        plt.fill_between(
            epoch_axis,
            val_loss_mean - val_loss_sem,
            val_loss_mean + val_loss_sem,
            alpha=0.3,
            label="Val ±1 SEM",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dset}: Aggregated Train/Val Loss (n={len(tr_loss_list)} runs)")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, f"{dset}_aggregated_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss curve for {dset}: {e}")
        plt.close()

    # --------------- aggregated F1 plot ------------------
    try:
        plt.figure()
        plt.plot(epoch_axis, tr_f1_mean, label="Train mean")
        plt.fill_between(
            epoch_axis,
            tr_f1_mean - tr_f1_sem,
            tr_f1_mean + tr_f1_sem,
            alpha=0.3,
            label="Train ±1 SEM",
        )
        plt.plot(epoch_axis, val_f1_mean, label="Val mean")
        plt.fill_between(
            epoch_axis,
            val_f1_mean - val_f1_sem,
            val_f1_mean + val_f1_sem,
            alpha=0.3,
            label="Val ±1 SEM",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title(f"{dset}: Aggregated Train/Val Macro-F1 (n={len(tr_f1_list)} runs)")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, f"{dset}_aggregated_macro_f1_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated F1 curve for {dset}: {e}")
        plt.close()

print("Aggregate plots saved to", working_dir)

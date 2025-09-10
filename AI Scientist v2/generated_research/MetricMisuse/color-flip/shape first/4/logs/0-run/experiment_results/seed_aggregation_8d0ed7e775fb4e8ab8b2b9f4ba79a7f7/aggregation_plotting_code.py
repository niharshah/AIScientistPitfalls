import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
# basic set-up
# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# load every experiment_data.npy that the user listed
# ------------------------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-15_23-37-14_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_43ebeb7700684ccdb5bbc99b916c8118_proc_3023916/experiment_data.npy",
    "experiments/2025-08-15_23-37-14_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_4c374067d39241aa8a61d322310096f4_proc_3023915/experiment_data.npy",
    "experiments/2025-08-15_23-37-14_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_90429db0f47349c7a7039a95ac1deeba_proc_3023914/experiment_data.npy",
]

all_runs = []  # each entry is a dict with keys: losses.train, losses.val, metrics.val
dataset_name = "SPR_BENCH"

for p in experiment_data_path_list:
    try:
        data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        runs_dict = data.get("epochs", {}).get(dataset_name, {})
        for _, rec in runs_dict.items():
            all_runs.append(rec)
    except Exception as e:
        print(f"Error loading {p}: {e}")

if not all_runs:
    print("No experiment data found — nothing to plot.")
    exit()


# ------------------------------------------------------------------
# helper to aggregate list of 1-D arrays of possibly different length
# ------------------------------------------------------------------
def aggregate(list_of_arrays):
    max_len = max(len(arr) for arr in list_of_arrays)
    mat = np.full((len(list_of_arrays), max_len), np.nan, dtype=float)
    for i, arr in enumerate(list_of_arrays):
        mat[i, : len(arr)] = arr
    mean = np.nanmean(mat, axis=0)
    se = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(mat), axis=0))
    return mean, se, np.arange(1, max_len + 1)


# ------------------------------------------------------------------
# collect per-run sequences
# ------------------------------------------------------------------
train_losses = [
    np.asarray(r["losses"]["train"], dtype=float) for r in all_runs if "losses" in r
]
val_losses = [
    np.asarray(r["losses"]["val"], dtype=float) for r in all_runs if "losses" in r
]
val_hwa = [
    np.asarray(r["metrics"]["val"], dtype=float) for r in all_runs if "metrics" in r
]

# ------------------------------------------------------------------
# PLOT 1 : aggregated loss curves with SE bands
# ------------------------------------------------------------------
try:
    mean_train, se_train, epochs_train = aggregate(train_losses)
    mean_val, se_val, epochs_val = aggregate(val_losses)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs_train, mean_train, label="Train mean", color="tab:blue")
    plt.fill_between(
        epochs_train,
        mean_train - se_train,
        mean_train + se_train,
        color="tab:blue",
        alpha=0.25,
        label="Train ±1 SE",
    )

    plt.plot(epochs_val, mean_val, label="Val mean", color="tab:orange")
    plt.fill_between(
        epochs_val,
        mean_val - se_val,
        mean_val + se_val,
        color="tab:orange",
        alpha=0.25,
        label="Val ±1 SE",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"{dataset_name}: Mean ± SE Training / Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, f"{dataset_name}_aggregated_loss_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
finally:
    plt.close()

# ------------------------------------------------------------------
# PLOT 2 : aggregated validation HWA with SE bands
# ------------------------------------------------------------------
try:
    mean_hwa, se_hwa, epochs_hwa = aggregate(val_hwa)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs_hwa, mean_hwa, label="Val HWA mean", color="tab:green")
    plt.fill_between(
        epochs_hwa,
        mean_hwa - se_hwa,
        mean_hwa + se_hwa,
        color="tab:green",
        alpha=0.25,
        label="Val HWA ±1 SE",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Accuracy")
    plt.title(f"{dataset_name}: Mean ± SE Validation HWA")
    plt.legend()
    fname = os.path.join(working_dir, f"{dataset_name}_aggregated_HWA_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating aggregated HWA plot: {e}")
finally:
    plt.close()

# ------------------------------------------------------------------
# Print final-epoch statistics for quick inspection
# ------------------------------------------------------------------
try:
    final_hwa_values = [arr[-1] for arr in val_hwa if len(arr) > 0]
    if final_hwa_values:
        mean_final = np.mean(final_hwa_values)
        std_final = np.std(final_hwa_values, ddof=1)
        print(
            f"Final-epoch HWA across runs: {mean_final:.4f} ± {std_final:.4f} (mean ± std, n={len(final_hwa_values)})"
        )
except Exception as e:
    print(f"Error computing final HWA statistics: {e}")

import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
# Basic set-up
# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# 1) Load every experiment_data.npy that the runner provided
# ------------------------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-30_20-55-31_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_731ee1f27d6241839fe2bceae68adb10_proc_1488339/experiment_data.npy",
    "experiments/2025-08-30_20-55-31_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_a869c7077e334db8a4a9b5c535247103_proc_1488338/experiment_data.npy",
    "experiments/2025-08-30_20-55-31_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_d3b0ebcec02d4e5aa044bf37ad9447af_proc_1488341/experiment_data.npy",
]

all_runs = []
for path in experiment_data_path_list:
    try:
        run_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), path), allow_pickle=True
        ).item()
        all_runs.append(run_data)
    except Exception as e:
        print(f"Error loading {path}: {e}")

if not all_runs:
    print("No experiment data loaded — aborting plotting script.")
    exit(0)

# ------------------------------------------------------------------
# 2) Reorganize: dict[dataset_name] -> list_of_run_logs
# ------------------------------------------------------------------
per_dataset = {}
for run in all_runs:
    for dset_name, log in run.items():
        per_dataset.setdefault(dset_name, []).append(log)


# ------------------------------------------------------------------
# Helper to compute mean & SEM while ignoring ragged tails
# ------------------------------------------------------------------
def stack_and_trim(list_of_lists):
    """
    Takes e.g. [[...len e1...], [...len e2...], ...]  -> 2D array (n_runs, min_len)
    """
    if not list_of_lists:
        return np.array([])
    min_len = min(len(x) for x in list_of_lists)
    if min_len == 0:
        return np.array([])
    trimmed = np.stack([np.asarray(x[:min_len]) for x in list_of_lists], axis=0)
    return trimmed


# ------------------------------------------------------------------
# 3) Iterate over datasets and create aggregated plots
# ------------------------------------------------------------------
for dset_name, run_logs in per_dataset.items():
    # Collect per-run sequences
    losses_tr_runs = [rl.get("losses", {}).get("train", []) for rl in run_logs]
    losses_val_runs = [rl.get("losses", {}).get("val", []) for rl in run_logs]
    acc_tr_runs = [rl.get("metrics", {}).get("train", []) for rl in run_logs]
    acc_val_runs = [rl.get("metrics", {}).get("val", []) for rl in run_logs]
    epochs_runs = [rl.get("epochs", []) for rl in run_logs]

    # Use shortest sequence length so shapes agree
    epoch_mat = stack_and_trim(epochs_runs)
    if epoch_mat.size == 0:
        print(f"No epoch data for {dset_name}, skipping.")
        continue
    epochs = epoch_mat[0]  # same for all after trimming

    # ------------------------------------------------------------------
    # Aggregate Loss ----------------------------------------------------
    # ------------------------------------------------------------------
    try:
        loss_tr_mat = stack_and_trim(losses_tr_runs)
        loss_val_mat = stack_and_trim(losses_val_runs)
        if loss_tr_mat.size and loss_val_mat.size:
            # Mean & SEM
            mean_tr = loss_tr_mat.mean(axis=0)
            sem_tr = loss_tr_mat.std(axis=0, ddof=1) / np.sqrt(loss_tr_mat.shape[0])
            mean_val = loss_val_mat.mean(axis=0)
            sem_val = loss_val_mat.std(axis=0, ddof=1) / np.sqrt(loss_val_mat.shape[0])

            plt.figure()
            plt.plot(epochs, mean_tr, label="Train (mean)")
            plt.fill_between(
                epochs,
                mean_tr - sem_tr,
                mean_tr + sem_tr,
                alpha=0.3,
                label="Train (±SE)",
            )
            plt.plot(epochs, mean_val, label="Val (mean)")
            plt.fill_between(
                epochs,
                mean_val - sem_val,
                mean_val + sem_val,
                alpha=0.3,
                label="Val (±SE)",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(
                f"{dset_name}: Mean ± SE Training vs Validation Loss (n={loss_tr_mat.shape[0]})"
            )
            plt.legend()
            fname = os.path.join(working_dir, f"{dset_name}_aggregated_loss_curve.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss curve for {dset_name}: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # Aggregate Accuracy ------------------------------------------------
    # ------------------------------------------------------------------
    try:
        acc_tr_mat = stack_and_trim(acc_tr_runs)
        acc_val_mat = stack_and_trim(acc_val_runs)
        if acc_tr_mat.size and acc_val_mat.size:
            mean_tr = acc_tr_mat.mean(axis=0)
            sem_tr = acc_tr_mat.std(axis=0, ddof=1) / np.sqrt(acc_tr_mat.shape[0])
            mean_val = acc_val_mat.mean(axis=0)
            sem_val = acc_val_mat.std(axis=0, ddof=1) / np.sqrt(acc_val_mat.shape[0])

            plt.figure()
            plt.plot(epochs, mean_tr, label="Train (mean)")
            plt.fill_between(
                epochs,
                mean_tr - sem_tr,
                mean_tr + sem_tr,
                alpha=0.3,
                label="Train (±SE)",
            )
            plt.plot(epochs, mean_val, label="Val (mean)")
            plt.fill_between(
                epochs,
                mean_val - sem_val,
                mean_val + sem_val,
                alpha=0.3,
                label="Val (±SE)",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Complexity-Weighted Accuracy")
            plt.title(
                f"{dset_name}: Mean ± SE Training vs Validation Accuracy (n={acc_tr_mat.shape[0]})"
            )
            plt.legend()
            fname = os.path.join(
                working_dir, f"{dset_name}_aggregated_accuracy_curve.png"
            )
            plt.savefig(fname)
            print(f"Saved {fname}")
            plt.close()

            # Print final-epoch validation accuracy
            final_mean = mean_val[-1]
            final_se = sem_val[-1]
            print(
                f"{dset_name}: final validation accuracy = {final_mean:.4f} ± {final_se:.4f} (SE)"
            )
    except Exception as e:
        print(f"Error creating aggregated accuracy curve for {dset_name}: {e}")
        plt.close()

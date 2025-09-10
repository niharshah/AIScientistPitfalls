import matplotlib.pyplot as plt
import numpy as np
import os

# -----------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -----------------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-15_01-36-11_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_473b3a2d7fca42898591ac718ee0c708_proc_2795687/experiment_data.npy",
    "experiments/2025-08-15_01-36-11_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_8e0956f3e945428d8b6e404a59f2053b_proc_2795689/experiment_data.npy",
    "experiments/2025-08-15_01-36-11_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_cbaf9b4b72544901a014942daafe7426_proc_2795686/experiment_data.npy",
]

all_experiment_data = []
try:
    for exp_path in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), exp_path)
        exp_data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []


# -----------------------------------------------------------
def pad_to_max(list_of_lists, pad_val=np.nan):
    max_len = max(len(l) for l in list_of_lists)
    out = []
    for l in list_of_lists:
        if len(l) < max_len:
            l = list(l) + [pad_val] * (max_len - len(l))
        out.append(l)
    return np.array(out, dtype=float)


for dataset in {k for exp in all_experiment_data for k in exp.keys()}:
    # Gather metrics across runs ------------------------------------------
    train_runs, val_runs, hwa_runs, test_hwa_runs = [], [], [], []
    for exp in all_experiment_data:
        if dataset not in exp:
            continue
        data = exp[dataset]
        train_runs.append(data.get("losses", {}).get("train", []))
        val_runs.append(data.get("losses", {}).get("val", []))
        hwa_runs.append(data.get("metrics", {}).get("val", []))
        if "test" in data.get("metrics", {}):
            test_hwa_runs.append(data["metrics"]["test"])

    # Skip dataset if nothing was gathered
    if not (train_runs or val_runs or hwa_runs):
        continue

    # Pad with NaNs so arrays are same length ------------------------------
    if train_runs:
        train_arr = pad_to_max(train_runs)
        train_mean = np.nanmean(train_arr, axis=0)
        train_se = np.nanstd(train_arr, axis=0, ddof=1) / np.sqrt(train_arr.shape[0])
    if val_runs:
        val_arr = pad_to_max(val_runs)
        val_mean = np.nanmean(val_arr, axis=0)
        val_se = np.nanstd(val_arr, axis=0, ddof=1) / np.sqrt(val_arr.shape[0])
    if hwa_runs:
        hwa_arr = pad_to_max(hwa_runs)
        hwa_mean = np.nanmean(hwa_arr, axis=0)
        hwa_se = np.nanstd(hwa_arr, axis=0, ddof=1) / np.sqrt(hwa_arr.shape[0])

    # ---------------------------------------------------------------------
    # Plot 1: Loss curves with SE bands
    try:
        plt.figure()
        epochs = (
            np.arange(1, len(train_mean) + 1)
            if train_runs
            else np.arange(1, len(val_mean) + 1)
        )
        if train_runs:
            plt.plot(epochs, train_mean, label="Train Mean", color="tab:blue")
            plt.fill_between(
                epochs,
                train_mean - train_se,
                train_mean + train_se,
                alpha=0.3,
                color="tab:blue",
                label="Train SE",
            )
        if val_runs:
            plt.plot(epochs, val_mean, label="Val Mean", color="tab:orange")
            plt.fill_between(
                epochs,
                val_mean - val_se,
                val_mean + val_se,
                alpha=0.3,
                color="tab:orange",
                label="Val SE",
            )
        plt.title(f"{dataset}: Training vs Validation Loss (Mean ± SE)")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.tight_layout()
        fname = f"{dataset}_agg_loss_curve.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss curve for {dataset}: {e}")
        plt.close()

    # ---------------------------------------------------------------------
    # Plot 2: Validation HWA curve with SE bands
    try:
        if hwa_runs:
            plt.figure()
            epochs_hwa = np.arange(1, len(hwa_mean) + 1)
            plt.plot(
                epochs_hwa,
                hwa_mean,
                marker="o",
                label="Val HWA Mean",
                color="tab:green",
            )
            plt.fill_between(
                epochs_hwa,
                hwa_mean - hwa_se,
                hwa_mean + hwa_se,
                alpha=0.3,
                color="tab:green",
                label="Val HWA SE",
            )
            subtitle = ""
            if test_hwa_runs:
                test_hwa_mean = np.mean(test_hwa_runs)
                test_hwa_se = np.std(test_hwa_runs, ddof=1) / np.sqrt(
                    len(test_hwa_runs)
                )
                subtitle = f"(Final Test HWA: {test_hwa_mean:.4f} ± {test_hwa_se:.4f})"
            plt.title(f"{dataset}: Validation HWA over Epochs {subtitle}")
            plt.xlabel("Epoch")
            plt.ylabel("Harmonic-Weighted Accuracy")
            plt.ylim(0, 1)
            plt.legend()
            plt.tight_layout()
            fname = f"{dataset}_agg_hwa_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated HWA curve for {dataset}: {e}")
        plt.close()

    # ---------------------------------------------------------------------
    # Print aggregated test metric
    if test_hwa_runs:
        print(
            f"{dataset} Final Test HWA: {np.mean(test_hwa_runs):.4f} ± "
            f"{np.std(test_hwa_runs, ddof=1)/np.sqrt(len(test_hwa_runs)):.4f}"
        )

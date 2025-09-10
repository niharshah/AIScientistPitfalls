import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------- load ALL experiment_data -----------------
try:
    experiment_data_path_list = [
        "experiments/2025-08-17_22-28-20_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_03dddee9063945c78a9e952454ebd4b8_proc_3448830/experiment_data.npy",
        "experiments/2025-08-17_22-28-20_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_954a5ed87064423f96659d52a86570d1_proc_3448832/experiment_data.npy",
        "experiments/2025-08-17_22-28-20_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_aa44a7465317479e9722d17c3f983963_proc_3448831/experiment_data.npy",
    ]
    all_experiment_data = []
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []


# helper ----------------------------------------------------
def pad_to_max(list_of_lists, pad_val=np.nan):
    max_len = max(len(x) for x in list_of_lists)
    arr = np.full((len(list_of_lists), max_len), pad_val, dtype=float)
    for i, l in enumerate(list_of_lists):
        arr[i, : len(l)] = l
    return arr


# ---------------- iterate over datasets --------------------
# collect every dataset key that appears anywhere
dataset_keys = set()
for exp in all_experiment_data:
    for run_key in exp:
        dataset_keys.update(exp[run_key].keys())

for dset in dataset_keys:
    # ---------- aggregate losses -----------
    try:
        train_losses_runs, val_losses_runs = [], []
        for exp in all_experiment_data:
            for run_key in exp:
                run = exp[run_key].get(dset, {})
                losses = run.get("losses", {})
                if losses:
                    train_losses_runs.append(list(losses.get("train", [])))
                    val_losses_runs.append(list(losses.get("val", [])))
        if train_losses_runs and val_losses_runs:
            train_arr = pad_to_max(train_losses_runs)
            val_arr = pad_to_max(val_losses_runs)
            epochs = np.arange(1, train_arr.shape[1] + 1)

            train_mean = np.nanmean(train_arr, axis=0)
            val_mean = np.nanmean(val_arr, axis=0)
            train_se = np.nanstd(train_arr, axis=0, ddof=1) / np.sqrt(
                np.sum(~np.isnan(train_arr), axis=0)
            )
            val_se = np.nanstd(val_arr, axis=0, ddof=1) / np.sqrt(
                np.sum(~np.isnan(val_arr), axis=0)
            )

            plt.figure()
            plt.plot(epochs, train_mean, label="Train Loss (mean)", color="tab:blue")
            plt.fill_between(
                epochs,
                train_mean - train_se,
                train_mean + train_se,
                color="tab:blue",
                alpha=0.3,
                label="Train ± SE",
            )
            plt.plot(epochs, val_mean, label="Val Loss (mean)", color="tab:orange")
            plt.fill_between(
                epochs,
                val_mean - val_se,
                val_mean + val_se,
                color="tab:orange",
                alpha=0.3,
                label="Val ± SE",
            )
            plt.title(
                f"{dset} Aggregate Loss Curves\nMean ±1 SE across {train_arr.shape[0]} runs"
            )
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset}_loss_curves_aggregate.png")
            plt.savefig(fname)
            plt.close()
            print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating aggregate loss plot for {dset}: {e}")
        plt.close()

    # ---------- aggregate val metrics -----------
    try:
        macro_runs, cwa_runs = [], []
        for exp in all_experiment_data:
            for run_key in exp:
                run = exp[run_key].get(dset, {})
                metrics_seq = run.get("metrics", {}).get("val", [])
                if metrics_seq:
                    macro_runs.append([m.get("macro_f1") for m in metrics_seq if m])
                    cwa_runs.append([m.get("cwa") for m in metrics_seq if m])
        if macro_runs and cwa_runs:
            macro_arr = pad_to_max(macro_runs)
            cwa_arr = pad_to_max(cwa_runs)
            epochs = np.arange(1, macro_arr.shape[1] + 1)

            macro_mean = np.nanmean(macro_arr, axis=0)
            cwa_mean = np.nanmean(cwa_arr, axis=0)
            macro_se = np.nanstd(macro_arr, axis=0, ddof=1) / np.sqrt(
                np.sum(~np.isnan(macro_arr), axis=0)
            )
            cwa_se = np.nanstd(cwa_arr, axis=0, ddof=1) / np.sqrt(
                np.sum(~np.isnan(cwa_arr), axis=0)
            )

            plt.figure()
            ax1 = plt.gca()
            ax1.plot(epochs, macro_mean, color="b", label="Macro-F1 (mean)")
            ax1.fill_between(
                epochs,
                macro_mean - macro_se,
                macro_mean + macro_se,
                color="b",
                alpha=0.3,
                label="Macro-F1 ± SE",
            )
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Macro-F1", color="b")
            ax2 = ax1.twinx()
            ax2.plot(epochs, cwa_mean, color="r", linestyle="--", label="CWA (mean)")
            ax2.fill_between(
                epochs,
                cwa_mean - cwa_se,
                cwa_mean + cwa_se,
                color="r",
                alpha=0.3,
                label="CWA ± SE",
            )
            ax2.set_ylabel("CWA", color="r")
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            plt.legend(lines + lines2, labels + labels2, loc="best")
            plt.title(
                f"{dset} Validation Metrics\nMean ±1 SE across {macro_arr.shape[0]} runs"
            )
            fname = os.path.join(working_dir, f"{dset}_val_metrics_aggregate.png")
            plt.savefig(fname)
            plt.close()
            print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating aggregate val metric plot for {dset}: {e}")
        plt.close()

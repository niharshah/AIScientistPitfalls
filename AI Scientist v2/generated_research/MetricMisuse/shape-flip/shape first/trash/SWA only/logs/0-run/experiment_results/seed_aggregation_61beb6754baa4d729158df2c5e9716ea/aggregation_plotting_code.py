import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------------------------------------------------
# Load all experiment_data files that are listed in the specification
# --------------------------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-14_01-55-43_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_74ebb360709841f0b406cd1602f86a81_proc_2602725/experiment_data.npy",
    "experiments/2025-08-14_01-55-43_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_dd83e22a63b54cb1abe554b4d385c3b0_proc_2602726/experiment_data.npy",
    "experiments/2025-08-14_01-55-43_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_ee2ddcd6d60e4044ae81871fe1ae2d3e_proc_2602724/experiment_data.npy",
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

if not all_experiment_data:
    print("No experiment files could be loaded, aborting plots.")
else:
    # ---------------------------------------------------------------
    # Gather data across runs
    # ---------------------------------------------------------------
    # Collect every unique weight_decay across all runs
    wd_set = set()
    for exp in all_experiment_data:
        wd_set.update(exp["weight_decay"].keys())
    wd_keys = sorted(wd_set, key=float)

    # Helper to accumulate per-run lists
    dev_curves_dict = {wd: [] for wd in wd_keys}
    test_acc_dict = {wd: [] for wd in wd_keys}
    swa_dict = {wd: [] for wd in wd_keys}
    cwa_dict = {wd: [] for wd in wd_keys}

    for exp in all_experiment_data:
        for wd in wd_keys:
            if wd not in exp["weight_decay"]:
                continue  # skip if this run lacks the wd value
            metrics = exp["weight_decay"][wd]["metrics"]
            dev_curves_dict[wd].append(np.asarray(metrics["dev"], dtype=float))
            test_acc_dict[wd].append(float(metrics["test"]["acc"]))
            swa_dict[wd].append(float(metrics["test"]["swa"]))
            cwa_dict[wd].append(float(metrics["test"]["cwa"]))

    # Utility to compute mean & stderr safely
    def mean_stderr(lst):
        arr = np.asarray(lst, dtype=float)
        if arr.size == 0:
            return np.nan, np.nan
        mean = arr.mean()
        stderr = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        return mean, stderr

    # -------------------------------------------------------------
    # 1) Dev-accuracy curves with stderr ribbon
    # -------------------------------------------------------------
    try:
        plt.figure()
        for wd in wd_keys:
            runs = dev_curves_dict[wd]
            if not runs:
                continue
            # Trim to minimum common length
            min_len = min(len(r) for r in runs)
            runs_trim = np.stack([r[:min_len] for r in runs], axis=0)
            mean_curve = runs_trim.mean(axis=0)
            stderr_curve = (
                runs_trim.std(axis=0, ddof=1) / np.sqrt(runs_trim.shape[0])
                if runs_trim.shape[0] > 1
                else np.zeros_like(mean_curve)
            )
            epochs = np.arange(1, min_len + 1)
            plt.plot(epochs, mean_curve, label=f"wd={wd} (mean)")
            plt.fill_between(
                epochs, mean_curve - stderr_curve, mean_curve + stderr_curve, alpha=0.2
            )
        plt.title("Synthetic SPR – Dev Accuracy vs Epochs (mean ± stderr)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, "synthetic_spr_dev_accuracy_mean_stderr.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated dev accuracy plot: {e}")
        plt.close()

    # -------------------------------------------------------------
    # Helper to build bar-plot statistics
    # -------------------------------------------------------------
    def prepare_bar_data(data_dict):
        means = []
        stderrs = []
        valid_wd = []
        for wd in wd_keys:
            m, s = mean_stderr(data_dict[wd])
            if np.isnan(m):
                continue
            means.append(m)
            stderrs.append(s)
            valid_wd.append(wd)
        return valid_wd, np.asarray(means), np.asarray(stderrs)

    # -------------------------------------------------------------
    # 2) Test accuracy bar chart (mean ± stderr)
    # -------------------------------------------------------------
    try:
        v_wd, means, stderrs = prepare_bar_data(test_acc_dict)
        x = np.arange(len(v_wd))
        plt.figure()
        plt.bar(x, means, yerr=stderrs, capsize=5)
        plt.title("Synthetic SPR – Test Accuracy by Weight Decay (mean ± stderr)")
        plt.xlabel("Weight Decay")
        plt.ylabel("Accuracy")
        plt.xticks(x, v_wd)
        plt.tight_layout()
        fname = os.path.join(working_dir, "synthetic_spr_test_accuracy_mean_stderr.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated test accuracy bar: {e}")
        plt.close()

    # -------------------------------------------------------------
    # 3) Shape-Weighted Accuracy (SWA)
    # -------------------------------------------------------------
    try:
        v_wd, means, stderrs = prepare_bar_data(swa_dict)
        x = np.arange(len(v_wd))
        plt.figure()
        plt.bar(x, means, yerr=stderrs, capsize=5)
        plt.title(
            "Synthetic SPR – Shape-Weighted Accuracy (SWA) by Weight Decay (mean ± stderr)"
        )
        plt.xlabel("Weight Decay")
        plt.ylabel("SWA")
        plt.xticks(x, v_wd)
        plt.tight_layout()
        fname = os.path.join(working_dir, "synthetic_spr_swa_mean_stderr.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated SWA bar: {e}")
        plt.close()

    # -------------------------------------------------------------
    # 4) Color-Weighted Accuracy (CWA)
    # -------------------------------------------------------------
    try:
        v_wd, means, stderrs = prepare_bar_data(cwa_dict)
        x = np.arange(len(v_wd))
        plt.figure()
        plt.bar(x, means, yerr=stderrs, capsize=5)
        plt.title(
            "Synthetic SPR – Color-Weighted Accuracy (CWA) by Weight Decay (mean ± stderr)"
        )
        plt.xlabel("Weight Decay")
        plt.ylabel("CWA")
        plt.xticks(x, v_wd)
        plt.tight_layout()
        fname = os.path.join(working_dir, "synthetic_spr_cwa_mean_stderr.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated CWA bar: {e}")
        plt.close()

    # -------------------------------------------------------------
    # Print best configuration according to mean test accuracy
    # -------------------------------------------------------------
    best_wd = None
    best_mean = -np.inf
    best_stderr = 0.0
    for wd in wd_keys:
        m, s = mean_stderr(test_acc_dict[wd])
        if np.isnan(m):
            continue
        if m > best_mean:
            best_mean, best_stderr, best_wd = m, s, wd
    if best_wd is not None:
        print(
            f"Best weight_decay={best_wd} with mean test accuracy={best_mean:.3f} ± {best_stderr:.3f}"
        )

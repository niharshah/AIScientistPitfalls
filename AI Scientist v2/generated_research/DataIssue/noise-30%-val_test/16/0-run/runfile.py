import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------- paths & constants ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Experiment data (relative to $AI_SCIENTIST_ROOT)
experiment_data_path_list = [
    "experiments/2025-08-17_18-47-55_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_3404b88306e94a93bb791755a8841632_proc_3338338/experiment_data.npy",
    "experiments/2025-08-17_18-47-55_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_ad156518f1b945e7aa57e43a6ebb5ff5_proc_3338340/experiment_data.npy",
    "experiments/2025-08-17_18-47-55_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_cdd18529ee024526a1fab484c64aba23_proc_3338341/experiment_data.npy",
]

# ---------------- load every run ----------------
all_runs = []
for p in experiment_data_path_list:
    try:
        full_p = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp = np.load(full_p, allow_pickle=True).item()
        if "SPR_BENCH" in exp:
            all_runs.append(exp["SPR_BENCH"])
        else:
            print(f"Warning: SPR_BENCH entry missing in {p}")
    except Exception as e:
        print(f"Error loading {p}: {e}")

if len(all_runs) == 0:
    raise SystemExit("No valid runs found – aborting.")

n_runs = len(all_runs)
print(f"Loaded {n_runs} SPR_BENCH run(s).")


# ---------------- helper to collect & align ----------------
def stack_metric(path, subkey):
    """Return epochs, 2-D array (runs×epochs), or ([],[]) if nothing found."""
    series = []
    for run in all_runs:
        arr = np.array(run[path].get(subkey, [])) if path in run else np.array([])
        series.append(arr)
    min_len = min(len(a) for a in series)
    if min_len == 0:
        return np.array([]), np.array([[]])
    trimmed = np.stack([a[:min_len] for a in series])  # runs × epochs
    epochs = np.array(all_runs[0]["epochs"][:min_len])
    return epochs, trimmed


# ---------------- aggregate curves ----------------
plots = [
    (
        "Loss Curves",
        ("losses", "train"),
        ("losses", "val"),
        "BCE Loss",
        "spr_bench_aggregated_loss_curves.png",
    ),
    (
        "Accuracy Curves",
        ("metrics", "train", "acc"),
        ("metrics", "val", "acc"),
        "Accuracy",
        "spr_bench_aggregated_accuracy_curves.png",
    ),
    (
        "MCC Curves",
        ("metrics", "train", "MCC"),
        ("metrics", "val", "MCC"),
        "Matthews CorrCoef",
        "spr_bench_aggregated_mcc_curves.png",
    ),
]


def retrieve(run_dict, keys):
    """Nested get with default {}→[] to avoid KeyErrors."""
    d = run_dict
    for k in keys:
        d = d.get(k, {})
    return d


# plotting loop --------------------------------------------------------------
for title, train_path, val_path, ylabel, fname in plots:
    try:
        # unpack keys: ("losses","train") etc.
        if len(train_path) == 2:
            t_epochs, train_mat = stack_metric(train_path[0], train_path[1])
            v_epochs, val_mat = stack_metric(val_path[0], val_path[1])
        else:  # metrics / train / acc
            t_epochs, train_mat = stack_metric(train_path[0], train_path[2])
            v_epochs, val_mat = stack_metric(val_path[0], val_path[2])

        if t_epochs.size and v_epochs.size:
            # train
            train_mean = np.nanmean(train_mat, axis=0)
            train_sem = np.nanstd(train_mat, axis=0, ddof=1) / np.sqrt(n_runs)
            # val
            val_mean = np.nanmean(val_mat, axis=0)
            val_sem = np.nanstd(val_mat, axis=0, ddof=1) / np.sqrt(n_runs)

            plt.figure()
            plt.plot(t_epochs, train_mean, label="Train μ")
            plt.fill_between(
                t_epochs,
                train_mean - train_sem,
                train_mean + train_sem,
                alpha=0.3,
                label="Train ±SEM",
            )
            plt.plot(v_epochs, val_mean, label="Val μ")
            plt.fill_between(
                v_epochs,
                val_mean - val_sem,
                val_mean + val_sem,
                alpha=0.3,
                label="Val ±SEM",
            )
            plt.title(f"SPR_BENCH {title}\nMean ± SEM across {n_runs} runs")
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.legend()
            plt.savefig(os.path.join(working_dir, fname))
            print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating plot {fname}: {e}")
        plt.close()

# ---------------- aggregate test metrics ----------------
try:
    test_metrics_keys = set().union(
        *(run.get("test_metrics", {}).keys() for run in all_runs)
    )
    if test_metrics_keys:
        print("\n===== AGGREGATED TEST METRICS =====")
        for key in sorted(test_metrics_keys):
            vals = [run.get("test_metrics", {}).get(key, np.nan) for run in all_runs]
            vals = np.array(vals, dtype=float)
            mean = np.nanmean(vals)
            std = np.nanstd(vals, ddof=1)
            print(f"{key}: {mean:.4f} ± {std:.4f} (n={n_runs})")
except Exception as e:
    print(f"Error aggregating test metrics: {e}")

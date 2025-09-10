import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load all experiment_data.npy ----------
experiment_data_path_list = [
    "experiments/2025-08-17_18-48-06_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_b9757632f0544054b7dd88cd2cf4e7dd_proc_3345783/experiment_data.npy",
    "experiments/2025-08-17_18-48-06_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_e09938cb37b84e3db10677797547944a_proc_3345785/experiment_data.npy",
    "experiments/2025-08-17_18-48-06_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_33c71e5bb62544f9965c0085f58d24d2_proc_3345784/experiment_data.npy",
]
all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        d = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(d)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# ---------- aggregate by dataset ----------
datasets_runs = {}  # {dataset_name: [run1_dict, run2_dict, ...]}
for run_dict in all_experiment_data:
    for dset, ddata in run_dict.items():
        datasets_runs.setdefault(dset, []).append(ddata)


def _stack_and_trim(list_of_lists):
    """
    Stack 1-D arrays from different runs, trimming to shortest length.
    Returns np.ndarray shape (n_runs, min_len)
    """
    min_len = min(len(x) for x in list_of_lists)
    return np.stack([np.asarray(x)[:min_len] for x in list_of_lists], axis=0)


# ---------- iterate datasets ----------
for dset, runs in datasets_runs.items():
    # ===== aggregate losses =====
    try:
        train_losses = _stack_and_trim(
            [r.get("losses", {}).get("train", []) for r in runs]
        )
        val_losses = _stack_and_trim([r.get("losses", {}).get("val", []) for r in runs])

        epochs = np.arange(1, train_losses.shape[1] + 1)
        n_runs = train_losses.shape[0]

        # stats
        train_mean = np.mean(train_losses, axis=0)
        train_sem = np.std(train_losses, axis=0, ddof=1) / np.sqrt(n_runs)
        val_mean = np.mean(val_losses, axis=0)
        val_sem = np.std(val_losses, axis=0, ddof=1) / np.sqrt(n_runs)

        plt.figure()
        plt.plot(epochs, train_mean, label="train mean", color="tab:blue")
        plt.fill_between(
            epochs,
            train_mean - train_sem,
            train_mean + train_sem,
            color="tab:blue",
            alpha=0.2,
            label="train ± SEM",
        )
        plt.plot(epochs, val_mean, label="val mean", color="tab:orange", linestyle="--")
        plt.fill_between(
            epochs,
            val_mean - val_sem,
            val_mean + val_sem,
            color="tab:orange",
            alpha=0.2,
            label="val ± SEM",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dset}: Aggregated Training vs Validation Loss\n(N={n_runs} runs)")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset}_loss_aggregate.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dset}: {e}")
        plt.close()

    # ===== aggregate each metric =====
    try:
        # collect all metric keys that start with 'train_'
        metric_keys = set()
        for r in runs:
            metric_keys.update(
                [k for k in r.get("metrics", {}).keys() if k.startswith("train_")]
            )

        for train_key in metric_keys:
            metric_name = train_key[len("train_") :]
            val_key = f"val_{metric_name}"

            train_metrics_runs = [r.get("metrics", {}).get(train_key, []) for r in runs]
            val_metrics_runs = [r.get("metrics", {}).get(val_key, []) for r in runs]

            # skip if any run has no data
            if not all(len(arr) for arr in train_metrics_runs):
                continue

            train_stack = _stack_and_trim(train_metrics_runs)
            val_stack = _stack_and_trim(val_metrics_runs)

            epochs = np.arange(1, train_stack.shape[1] + 1)
            n_runs = train_stack.shape[0]

            train_mean = np.mean(train_stack, axis=0)
            train_sem = np.std(train_stack, axis=0, ddof=1) / np.sqrt(n_runs)
            val_mean = np.mean(val_stack, axis=0)
            val_sem = np.std(val_stack, axis=0, ddof=1) / np.sqrt(n_runs)

            plt.figure()
            plt.plot(epochs, train_mean, label="train mean", color="tab:green")
            plt.fill_between(
                epochs,
                train_mean - train_sem,
                train_mean + train_sem,
                color="tab:green",
                alpha=0.2,
                label="train ± SEM",
            )
            plt.plot(
                epochs, val_mean, label="val mean", color="tab:red", linestyle="--"
            )
            plt.fill_between(
                epochs,
                val_mean - val_sem,
                val_mean + val_sem,
                color="tab:red",
                alpha=0.2,
                label="val ± SEM",
            )
            plt.xlabel("Epoch")
            plt.ylabel(metric_name.upper())
            plt.title(f"{dset}: Aggregated {metric_name.upper()} (N={n_runs})")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset}_{metric_name}_aggregate.png")
            plt.savefig(fname, dpi=150)
            plt.close()

            # ---- console numeric summary of final epoch ----
            final_val_vals = val_stack[:, -1]
            mean_final = np.mean(final_val_vals)
            sem_final = np.std(final_val_vals, ddof=1) / np.sqrt(n_runs)
            print(
                f"{dset} | Final val {metric_name}: {mean_final:.4f} ± {sem_final:.4f} (SEM)"
            )
    except Exception as e:
        print(f"Error creating aggregated metric plots for {dset}: {e}")
        plt.close()

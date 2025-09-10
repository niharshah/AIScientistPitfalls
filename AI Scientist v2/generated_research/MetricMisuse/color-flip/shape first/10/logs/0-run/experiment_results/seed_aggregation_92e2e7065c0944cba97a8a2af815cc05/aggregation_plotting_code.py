import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------ #
# set up I/O
# ------------------------------------------------------------------ #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# All experiment result files that really exist
experiment_data_path_list = [
    "experiments/2025-08-16_02-32-02_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_615df3da2e60464b87a735f974f7f678_proc_3101767/experiment_data.npy",
    "experiments/2025-08-16_02-32-02_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_62f1e69283f94766bd53544b7a5a103f_proc_3101765/experiment_data.npy",
    "experiments/2025-08-16_02-32-02_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_731c79f4dd9044e6a39310c8e4fbc390_proc_3101764/experiment_data.npy",
]

all_experiment_data = []
try:
    for exp_path in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), exp_path)
        if not os.path.isfile(full_path):
            print(f"File not found, skipping: {full_path}")
            continue
        data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(data)
    if not all_experiment_data:
        raise RuntimeError("No experiment data files could be loaded.")
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []


# ------------------------------------------------------------------ #
# helper to aggregate arrays across runs
# ------------------------------------------------------------------ #
def aggregate_metric(run_dicts, key_chain):
    """
    run_dicts: list of dictionaries (one per run) for the same dataset
    key_chain: tuple of keys leading to the metric (e.g. ('losses','train_sup'))
    Returns:
        epochs (1-D np.array), mean_vals, sem_vals
    """
    series_dict = {}
    for rd in run_dicts:
        cursor = rd
        for k in key_chain:
            cursor = cursor.get(k, {})
        arr = np.array(cursor)
        if arr.size == 0:
            continue
        epochs = np.arange(1, len(arr) + 1)
        for ep, v in zip(epochs, arr):
            series_dict.setdefault(ep, []).append(v)

    if not series_dict:
        return None, None, None

    sorted_epochs = np.array(sorted(series_dict.keys()))
    means = np.array([np.mean(series_dict[ep]) for ep in sorted_epochs])
    sems = np.array(
        [
            (
                np.std(series_dict[ep], ddof=1) / np.sqrt(len(series_dict[ep]))
                if len(series_dict[ep]) > 1
                else 0.0
            )
            for ep in sorted_epochs
        ]
    )
    return sorted_epochs, means, sems


# ------------------------------------------------------------------ #
# Aggregate and plot
# ------------------------------------------------------------------ #
metric_specs = {
    ("losses", "contrastive"): "NT-Xent Loss",
    ("losses", "train_sup"): "Cross-Entropy Loss (Train)",
    ("losses", "val_sup"): "Cross-Entropy Loss (Val)",
    ("metrics", "val_ACS"): "Augmentation-Consistency Score (ACS)",
}

# collect dataset names that appear in ANY run
dataset_names = set()
for run in all_experiment_data:
    dataset_names.update(run.keys())

for dname in dataset_names:
    # gather all runs for this dataset
    runs_for_dataset = [run[dname] for run in all_experiment_data if dname in run]

    for key_chain, ylabel in metric_specs.items():
        try:
            epochs, mean_vals, sem_vals = aggregate_metric(runs_for_dataset, key_chain)
            if epochs is None:
                continue  # metric absent
            plt.figure()
            plt.plot(epochs, mean_vals, label="Mean", color="tab:blue")
            plt.fill_between(
                epochs,
                mean_vals - sem_vals,
                mean_vals + sem_vals,
                color="tab:blue",
                alpha=0.3,
                label="± SEM",
            )
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            title_metric = key_chain[-1]
            plt.title(
                f"{dname}: {ylabel} (Mean ± SEM over {len(runs_for_dataset)} runs)"
            )
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_{title_metric}_mean_sem.png")
            plt.savefig(fname)
            print("Saved", fname)
            plt.close()

            # print final epoch summary
            final_mean = mean_vals[-1]
            final_sem = sem_vals[-1]
            print(
                f"{dname} | {title_metric} | final epoch mean ± SEM: "
                f"{final_mean:.4f} ± {final_sem:.4f}"
            )
        except Exception as e:
            print(f"Error plotting {dname} {key_chain}: {e}")
            plt.close()

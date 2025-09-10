import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------------------------------------------------------
# Load every run
experiment_data_path_list = [
    "experiments/2025-08-15_01-36-11_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_241dae6d4d064224b83504d3c91bffdb_proc_2797186/experiment_data.npy",
    "experiments/2025-08-15_01-36-11_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_aaf4225adc8b499f9236c4e994de0f79_proc_2797190/experiment_data.npy",
    "experiments/2025-08-15_01-36-11_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_4820eac4b65d474ab425a9320e85f89b_proc_2797187/experiment_data.npy",
]

all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        all_experiment_data.append(np.load(full_path, allow_pickle=True).item())
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# --------------------------------------------------------------------------
# Aggregate across runs
metrics_collector = (
    {}
)  # {batch_size: {"train":[], "val":[], "hwa_val":[], "hwa_test":[]}}
for run_data in all_experiment_data:
    try:
        spr_runs = run_data["batch_size"]["SPR_BENCH"]
    except Exception:
        continue
    for bs, res in spr_runs.items():
        m = metrics_collector.setdefault(
            bs, {"train": [], "val": [], "hwa_val": [], "hwa_test": []}
        )
        m["train"].append(np.asarray(res["losses"]["train"], dtype=float))
        m["val"].append(np.asarray(res["losses"]["val"], dtype=float))
        m["hwa_val"].append(np.asarray(res["metrics"]["val"], dtype=float))
        m["hwa_test"].append(float(res["test_metrics"]))

batch_sizes = sorted(metrics_collector.keys())
if not batch_sizes:
    print("No metrics found to plot.")
    exit()


# Helper to stack arrays up to shortest length
def stack_and_sem(arr_list):
    if len(arr_list) == 0:
        return None, None
    min_len = min(len(a) for a in arr_list)
    arr = np.stack([a[:min_len] for a in arr_list], axis=0)  # shape (n_runs, epochs)
    mean = arr.mean(axis=0)
    sem = (
        arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        if arr.shape[0] > 1
        else np.zeros_like(mean)
    )
    return mean, sem


# --------------------------------------------------------------------------
# 1. Mean loss curves with SEM shading
try:
    plt.figure(figsize=(6, 4))
    for bs in batch_sizes:
        mean_train, sem_train = stack_and_sem(metrics_collector[bs]["train"])
        mean_val, sem_val = stack_and_sem(metrics_collector[bs]["val"])
        if mean_train is None or mean_val is None:
            continue
        epochs = np.arange(1, len(mean_train) + 1)
        # Train
        plt.plot(epochs, mean_train, label=f"train μ bs={bs}")
        plt.fill_between(
            epochs, mean_train - sem_train, mean_train + sem_train, alpha=0.2
        )
        # Val
        plt.plot(epochs, mean_val, linestyle="--", label=f"val μ bs={bs}")
        plt.fill_between(epochs, mean_val - sem_val, mean_val + sem_val, alpha=0.2)
    plt.title("SPR_BENCH Loss (Mean ± SEM)\nLeft: Train, Right: Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend(fontsize=6)
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_mean_loss_curves_all_bs.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

# --------------------------------------------------------------------------
# 2. Mean validation-HWA curves with SEM shading
try:
    plt.figure(figsize=(6, 4))
    for bs in batch_sizes:
        mean_hwa, sem_hwa = stack_and_sem(metrics_collector[bs]["hwa_val"])
        if mean_hwa is None:
            continue
        epochs = np.arange(1, len(mean_hwa) + 1)
        plt.plot(epochs, mean_hwa, label=f"μ bs={bs}")
        plt.fill_between(epochs, mean_hwa - sem_hwa, mean_hwa + sem_hwa, alpha=0.2)
    plt.title("SPR_BENCH Validation HWA (Mean ± SEM)")
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.ylim(0, 1.05)
    plt.legend(fontsize=6)
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_mean_val_hwa_curves_all_bs.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated HWA curve plot: {e}")
    plt.close()

# --------------------------------------------------------------------------
# 3. Test-HWA bar plot with SEM
try:
    plt.figure(figsize=(5, 3))
    means = [np.mean(metrics_collector[bs]["hwa_test"]) for bs in batch_sizes]
    sems = [
        (
            np.std(metrics_collector[bs]["hwa_test"], ddof=1)
            / np.sqrt(len(metrics_collector[bs]["hwa_test"]))
            if len(metrics_collector[bs]["hwa_test"]) > 1
            else 0.0
        )
        for bs in batch_sizes
    ]
    plt.bar(
        range(len(batch_sizes)), means, yerr=sems, capsize=4, tick_label=batch_sizes
    )
    plt.title("SPR_BENCH Test HWA vs Batch Size (Mean ± SEM)")
    plt.xlabel("Batch Size")
    plt.ylabel("HWA")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_mean_test_hwa_vs_bs.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated test HWA bar plot: {e}")
    plt.close()

# --------------------------------------------------------------------------
# Print evaluation summary
print("SPR_BENCH  Test HWA  (mean ± SEM)")
for bs, m in zip(batch_sizes, means):
    print(f"  Batch {bs}: {m:.4f} ± {sems[batch_sizes.index(bs)]:.4f}")

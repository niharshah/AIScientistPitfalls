import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- basic setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
saved_files = []

# ---------- load every experiment_data.npy that actually exists ----------
experiment_data_path_list = [
    "experiments/2025-08-14_21-45-52_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_50e9d73a6a3847428f28acf9015fb60b_proc_2753348/experiment_data.npy",
    "experiments/2025-08-14_21-45-52_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_5a37ba340bc448499729f60826d49df2_proc_2753346/experiment_data.npy",
    "experiments/2025-08-14_21-45-52_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_b952af9d95ff4037a35a7e120c8bdaf7_proc_2753349/experiment_data.npy",
]

all_runs = []  # list of dicts with keys: metrics, test_acc, test_ura
try:
    for rel_path in experiment_data_path_list:
        abs_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), rel_path)
        if os.path.isfile(abs_path):
            exp_data = np.load(abs_path, allow_pickle=True).item()
            for run_key, run_data in exp_data.get("EPOCHS", {}).items():
                all_runs.append(run_data)
        else:
            print(f"Warning: file not found – {abs_path}")
except Exception as e:
    print(f"Error loading experiment data: {e}")

# ---------- aggregate per-epoch curves ----------
try:
    metrics_to_plot = ["train_acc", "val_acc", "val_ura"]
    # collect arrays per metric
    metric_dict = {m: [] for m in metrics_to_plot}
    for run in all_runs:
        for m in metrics_to_plot:
            arr = run.get("metrics", {}).get(m, None)
            if arr is not None and len(arr) > 0:
                metric_dict[m].append(np.asarray(arr, dtype=float))
    # ensure we have at least one run for every metric
    min_len = None
    for m, lst in metric_dict.items():
        if len(lst) == 0:
            print(f"No data for metric {m}; it will be skipped.")
            metric_dict[m] = []
        else:
            lmin = min(len(a) for a in lst)
            min_len = lmin if min_len is None else min(min_len, lmin)

    if min_len and min_len > 0:
        epochs = np.arange(1, min_len + 1)
        plt.figure(figsize=(7, 4))
        for m, lst in metric_dict.items():
            if not lst:
                continue
            stacked = np.stack([a[:min_len] for a in lst], axis=0)
            mean = stacked.mean(axis=0)
            stderr = stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0])
            plt.plot(epochs, mean, label=f"{m.replace('_', ' ').title()} (mean)")
            plt.fill_between(
                epochs,
                mean - stderr,
                mean + stderr,
                alpha=0.2,
                label=f"{m.replace('_', ' ').title()} (±SEM)",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR_BENCH – Aggregate Epoch Curves\nMean ± Standard Error")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_aggregate_epoch_curves.png")
        plt.tight_layout()
        plt.savefig(fname)
        saved_files.append(fname)
        plt.close()
    else:
        print("No per-epoch metrics found to aggregate.")
except Exception as e:
    print(f"Error creating aggregate epoch plot: {e}")
    plt.close()

# ---------- aggregate final test metrics ----------
try:
    test_accs = [r.get("test_acc") for r in all_runs if r.get("test_acc") is not None]
    test_uras = [r.get("test_ura") for r in all_runs if r.get("test_ura") is not None]

    if test_accs and test_uras:
        means = [np.mean(test_accs), np.mean(test_uras)]
        sems = [
            np.std(test_accs, ddof=1) / np.sqrt(len(test_accs)),
            np.std(test_uras, ddof=1) / np.sqrt(len(test_uras)),
        ]
        x = np.arange(2)
        plt.figure(figsize=(6, 4))
        plt.bar(x, means, yerr=sems, capsize=5, tick_label=["Test Acc", "Test URA"])
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title("SPR_BENCH – Aggregate Test Metrics\nMean ± Standard Error")
        fname = os.path.join(working_dir, "SPR_BENCH_aggregate_test_metrics.png")
        plt.tight_layout()
        plt.savefig(fname)
        saved_files.append(fname)
        plt.close()
    else:
        print("No final test metrics found to aggregate.")
except Exception as e:
    print(f"Error creating aggregate test plot: {e}")
    plt.close()

print("Saved plots:", saved_files)

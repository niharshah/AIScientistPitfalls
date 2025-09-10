import matplotlib.pyplot as plt
import numpy as np
import os

# -------- paths & constants --------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data_path_list = [
    "experiments/2025-08-17_23-44-14_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_06e63572428c4513bfddccfe7ed13e48_proc_3469362/experiment_data.npy",
    "experiments/2025-08-17_23-44-14_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_eeaa67f3d528495caa16e34cf938b84e_proc_3469363/experiment_data.npy",
    "experiments/2025-08-17_23-44-14_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_86c937140d4843feb5286f8cd3692d80_proc_3469364/experiment_data.npy",
]


# -------- helper to stack runs --------
def stack_and_aggregate(run_dicts, key_chain):
    """run_dicts is list of dicts for one batch size; key_chain is list of nested keys.
    Returns mean, stderr arrays over runs clipped to common min length."""
    series = []
    for d in run_dicts:
        val = d
        try:
            for k in key_chain:
                val = val[k]
            series.append(np.array(val, dtype=float))
        except KeyError:
            continue
    if not series:
        return None, None
    # Trim to common length
    min_len = min(map(len, series))
    series = [s[:min_len] for s in series]
    arr = np.stack(series, axis=0)
    mean = arr.mean(axis=0)
    stderr = (
        arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        if arr.shape[0] > 1
        else np.zeros_like(mean)
    )
    return mean, stderr


# -------- load all experiment dicts --------
all_experiment_data = []
for path in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), path)
        exp = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp)
    except Exception as e:
        print(f"Error loading {path}: {e}")

# -------- regroup per batch size --------
batch_runs = {}  # {bs: [stats_dict_from_each_run]}
for exp in all_experiment_data:
    for bs, stats in exp.get("batch_size", {}).items():
        batch_runs.setdefault(bs, []).append(stats)

# keys we will aggregate
metric_specs = {
    "train_loss": ["losses", "train"],
    "val_loss": ["losses", "val"],
    "val_f1": ["metrics", "val_f1"],
}

# -------- aggregate curves --------
aggregated = {}  # bs -> metric_name -> (mean, stderr, epochs)
for bs, run_dicts in batch_runs.items():
    aggregated[bs] = {}
    # find minimum epochs across runs to align
    min_epochs_len = min(len(r["epochs"]) for r in run_dicts)
    epochs = np.array(run_dicts[0]["epochs"][:min_epochs_len])
    for mname, kchain in metric_specs.items():
        mean, stderr = stack_and_aggregate(run_dicts, kchain)
        if mean is None:
            continue
        aggregated[bs][mname] = (mean, stderr, epochs)


# -------- plotting helpers --------
def plot_curve(metric_name, ylabel, filename_suffix):
    try:
        plt.figure()
        for bs, mdict in aggregated.items():
            if metric_name not in mdict:
                continue
            mean, stderr, epochs = mdict[metric_name]
            plt.plot(epochs, mean, label=f"bs={bs}")
            plt.fill_between(epochs, mean - stderr, mean + stderr, alpha=0.3)
        if plt.gca().has_data():
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.title(f"SPR_BENCH {ylabel} vs Epoch (mean ± stderr)")
            plt.legend()
            fname = os.path.join(working_dir, f"spr_{filename_suffix}_mean_stderr.png")
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating {metric_name} plot: {e}")
        plt.close()


plot_curve("train_loss", "Training Loss", "train_loss")
plot_curve("val_loss", "Validation Loss", "val_loss")
plot_curve("val_f1", "Validation Macro-F1", "val_f1")

# -------- best F1 bar plot --------
try:
    bs_vals, mean_best, stderr_best = [], [], []
    for bs, run_dicts in batch_runs.items():
        bests = []
        for rd in run_dicts:
            if "val_f1" in metric_specs:  # only if exists
                try:
                    bests.append(np.max(rd["metrics"]["val_f1"]))
                except KeyError:
                    continue
        if bests:
            bs_vals.append(bs)
            mean_best.append(np.mean(bests))
            stderr_best.append(
                np.std(bests, ddof=1) / np.sqrt(len(bests)) if len(bests) > 1 else 0.0
            )
    if bs_vals:
        plt.figure()
        x = np.arange(len(bs_vals))
        plt.bar(x, mean_best, yerr=stderr_best, capsize=5)
        plt.xticks(x, bs_vals)
        plt.xlabel("Batch Size")
        plt.ylabel("Best Validation Macro-F1")
        plt.title("SPR_BENCH Best Validation Macro-F1 by Batch Size (mean ± stderr)")
        plt.savefig(os.path.join(working_dir, "spr_best_f1_bar_mean_stderr.png"))
        plt.close()
    else:
        plt.close()
except Exception as e:
    print(f"Error creating aggregated best-F1 bar plot: {e}")
    plt.close()

# -------- numeric summary --------
for bs, mb, se in zip(bs_vals, mean_best, stderr_best):
    print(f"Batch size {bs:>3}: best val Macro-F1 = {mb:.4f} ± {se:.4f}")

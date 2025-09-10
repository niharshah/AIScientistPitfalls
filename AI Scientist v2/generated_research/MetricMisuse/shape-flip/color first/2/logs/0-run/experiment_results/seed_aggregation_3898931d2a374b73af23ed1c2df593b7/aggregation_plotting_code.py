import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
# Basic I/O setup
# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# Load all experiment_data dictionaries
# ------------------------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-30_17-49-38_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_82d470518d7b4bc6be574a2b8a51d291_proc_1441385/experiment_data.npy",
    "experiments/2025-08-30_17-49-38_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_71b534c25dc3465ba38a6d3287a9cd5d_proc_1441387/experiment_data.npy",
    "experiments/2025-08-30_17-49-38_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_066a13ae404c43ad9fcdc4423c6d4ab5_proc_1441386/experiment_data.npy",
]

all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp_dict = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_dict)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# ------------------------------------------------------------------
# Aggregate per-epoch metrics across runs
# ------------------------------------------------------------------
agg = {}  # {pooling_type: {metric: list_of_arrays_from_runs}}
for exp in all_experiment_data:
    for pooling_type, datasets in exp.get("pooling_type", {}).items():
        log = datasets.get("SPR_BENCH", {})
        tl = np.array([v for _, v in log.get("losses", {}).get("train", [])])
        vl = np.array([v for _, v in log.get("losses", {}).get("val", [])])
        dwa = np.array([v for _, v in log.get("metrics", {}).get("val", [])])

        if pooling_type not in agg:
            agg[pooling_type] = {"train_loss": [], "val_loss": [], "dwa": []}

        # Only add if we actually have data
        if tl.size:
            agg[pooling_type]["train_loss"].append(tl)
        if vl.size:
            agg[pooling_type]["val_loss"].append(vl)
        if dwa.size:
            agg[pooling_type]["dwa"].append(dwa)


# Helper to stack runs to same length (min length across runs)
def stack_and_crop(list_of_1d_arrays):
    if not list_of_1d_arrays:
        return np.empty((0, 0))
    min_len = min(arr.shape[0] for arr in list_of_1d_arrays)
    cropped = np.stack([arr[:min_len] for arr in list_of_1d_arrays], axis=0)
    return cropped  # shape (n_runs, min_len)


# Prepare summary stats
summary = {}  # {pooling_type: {metric: {"mean":1d, "se":1d}}}
for p, metrics in agg.items():
    summary[p] = {}
    for m, runs in metrics.items():
        data = stack_and_crop(runs)
        if data.size == 0:
            continue
        mean = data.mean(axis=0)
        se = (
            data.std(axis=0, ddof=1) / np.sqrt(data.shape[0])
            if data.shape[0] > 1
            else np.zeros_like(mean)
        )
        summary[p][m] = {"mean": mean, "se": se}

# ------------------------------------------------------------------
# 1) Training & Validation Loss curves with standard error
# ------------------------------------------------------------------
try:
    plt.figure(figsize=(8, 5))
    for p, metrics in summary.items():
        if "train_loss" not in metrics or "val_loss" not in metrics:
            continue
        ep = np.arange(1, len(metrics["train_loss"]["mean"]) + 1)
        # Train
        plt.plot(
            ep, metrics["train_loss"]["mean"], linestyle="--", label=f"{p}-train mean"
        )
        plt.fill_between(
            ep,
            metrics["train_loss"]["mean"] - metrics["train_loss"]["se"],
            metrics["train_loss"]["mean"] + metrics["train_loss"]["se"],
            alpha=0.25,
        )
        # Val
        plt.plot(ep, metrics["val_loss"]["mean"], linestyle="-", label=f"{p}-val mean")
        plt.fill_between(
            ep,
            metrics["val_loss"]["mean"] - metrics["val_loss"]["se"],
            metrics["val_loss"]["mean"] + metrics["val_loss"]["se"],
            alpha=0.25,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss (mean ± SE) - SPR_BENCH")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves_mean_se.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss curves plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 2) Validation DWA curves with standard error
# ------------------------------------------------------------------
try:
    plt.figure(figsize=(8, 5))
    for p, metrics in summary.items():
        if "dwa" not in metrics:
            continue
        ep = np.arange(1, len(metrics["dwa"]["mean"]) + 1)
        plt.plot(ep, metrics["dwa"]["mean"], label=f"{p} mean")
        plt.fill_between(
            ep,
            metrics["dwa"]["mean"] - metrics["dwa"]["se"],
            metrics["dwa"]["mean"] + metrics["dwa"]["se"],
            alpha=0.25,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Dual Weighted Accuracy")
    plt.title("Validation DWA (mean ± SE) - SPR_BENCH")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_dwa_curves_mean_se.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated DWA curves plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 3) Final-epoch DWA bar chart with error bars
# ------------------------------------------------------------------
try:
    plt.figure(figsize=(6, 4))
    poolings, means, ses = [], [], []
    for p, metrics in summary.items():
        if "dwa" in metrics:
            poolings.append(p)
            means.append(metrics["dwa"]["mean"][-1])
            ses.append(metrics["dwa"]["se"][-1])
    if poolings:
        x = np.arange(len(poolings))
        plt.bar(x, means, yerr=ses, capsize=5, color="skyblue")
        plt.xticks(x, poolings)
        plt.ylabel("Final Dual Weighted Accuracy")
        plt.title("Final DWA by Pooling Type (mean ± SE) - SPR_BENCH")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_dwa_bar_mean_se.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated final DWA bar plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# Print aggregated final metrics for quick inspection
# ------------------------------------------------------------------
for p, metrics in summary.items():
    if "dwa" in metrics:
        print(
            f"{p}: final DWA mean={metrics['dwa']['mean'][-1]:.4f} ± SE={metrics['dwa']['se'][-1]:.4f}"
        )

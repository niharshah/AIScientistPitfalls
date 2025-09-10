import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- set up ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# paths provided by the system
experiment_data_path_list = [
    "experiments/2025-08-16_02-31-02_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_c41dc31e27224cfea6363fcb6a71c298_proc_3098140/experiment_data.npy",
    "experiments/2025-08-16_02-31-02_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_a79c6f3b2c534c0cbf387ca227b5516c_proc_3098141/experiment_data.npy",
    "experiments/2025-08-16_02-31-02_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_7916c64f30854c75a8cc8839da03b88a_proc_3098142/experiment_data.npy",
]

all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        run_data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(run_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")


# helper
def get_nested(d, keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


dataset_name = "SPR_BENCH"

# -------------- aggregate data -----------------
agg = {}  # {batch_size: {metric_name: [list_per_run]}}
for run_data in all_experiment_data:
    batch_dict = get_nested(run_data, ["BATCH_SIZE", dataset_name], {})
    for bsz, logs in batch_dict.items():
        bsz = int(bsz)
        cur = agg.setdefault(bsz, {"train_loss": [], "val_loss": [], "val_SCWA": []})
        tl = get_nested(logs, ["losses", "train"], [])
        vl = get_nested(logs, ["losses", "val"], [])
        sc = get_nested(logs, ["metrics", "val_SCWA"], [])
        if tl:
            cur["train_loss"].append(np.asarray(tl, dtype=float))
        if vl:
            cur["val_loss"].append(np.asarray(vl, dtype=float))
        if sc:
            cur["val_SCWA"].append(np.asarray(sc, dtype=float))


# utility to compute mean and sem trimmed to common length
def mean_sem(list_of_arr):
    if not list_of_arr:
        return None, None
    min_len = min(a.shape[0] for a in list_of_arr)
    stacked = np.stack([a[:min_len] for a in list_of_arr], axis=0)
    mean = stacked.mean(axis=0)
    sem = stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0])
    return mean, sem


# ---------------- plotting -----------------
for bsz, metrics in list(agg.items())[:3]:  # safety: at most 3 batch sizes
    # 1) aggregated loss curves
    try:
        tr_mean, tr_sem = mean_sem(metrics["train_loss"])
        vl_mean, vl_sem = mean_sem(metrics["val_loss"])
        if tr_mean is not None and vl_mean is not None:
            epochs = np.arange(1, len(tr_mean) + 1)
            plt.figure()
            plt.plot(epochs, tr_mean, label="Train Loss (mean)")
            plt.fill_between(
                epochs, tr_mean - tr_sem, tr_mean + tr_sem, alpha=0.3, label="Train SEM"
            )
            plt.plot(epochs, vl_mean, label="Val Loss (mean)")
            plt.fill_between(
                epochs, vl_mean - vl_sem, vl_mean + vl_sem, alpha=0.3, label="Val SEM"
            )
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{dataset_name} – Aggregated Loss Curves (Batch {bsz})")
            plt.legend()
            fname = f"{dataset_name}_batch{bsz}_agg_loss.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for batch {bsz}: {e}")
        plt.close()

    # 2) aggregated val SCWA curves
    try:
        sc_mean, sc_sem = mean_sem(metrics["val_SCWA"])
        if sc_mean is not None:
            epochs = np.arange(1, len(sc_mean) + 1)
            plt.figure()
            plt.plot(epochs, sc_mean, marker="o", label="Val SCWA (mean)")
            plt.fill_between(
                epochs, sc_mean - sc_sem, sc_mean + sc_sem, alpha=0.3, label="SEM"
            )
            plt.xlabel("Epoch")
            plt.ylabel("SCWA")
            plt.title(f"{dataset_name} – Aggregated Val SCWA (Batch {bsz})")
            plt.legend()
            fname = f"{dataset_name}_batch{bsz}_agg_val_SCWA.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated SCWA plot for batch {bsz}: {e}")
        plt.close()

# 3) final SCWA vs batch size with error bars
try:
    batch_sizes, means, sems = [], [], []
    for bsz, metrics in agg.items():
        vals = metrics["val_SCWA"]
        if vals:
            finals = [v[-1] for v in vals]
            batch_sizes.append(bsz)
            means.append(np.mean(finals))
            sems.append(np.std(finals, ddof=1) / np.sqrt(len(finals)))
    if batch_sizes:
        inds = np.argsort(batch_sizes)
        batch_sizes = np.array(batch_sizes)[inds]
        means = np.array(means)[inds]
        sems = np.array(sems)[inds]
        plt.figure()
        plt.errorbar(
            batch_sizes, means, yerr=sems, fmt="s-", capsize=4, label="Final SCWA"
        )
        for x, y in zip(batch_sizes, means):
            plt.text(x, y, f"{y:.3f}")
        plt.xlabel("Batch Size")
        plt.ylabel("Final SCWA (mean ± SEM)")
        plt.title(f"{dataset_name} – Final SCWA vs. Batch Size (Aggregated)")
        plt.legend()
        fname = f"{dataset_name}_agg_final_SCWA_vs_batch.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
except Exception as e:
    print(f"Error creating aggregated summary SCWA plot: {e}")
    plt.close()

# -------------- console summary -------------
print("\nAggregated Final SCWA:")
for bsz, metrics in sorted(agg.items()):
    vals = metrics["val_SCWA"]
    if vals:
        finals = [v[-1] for v in vals]
        mean = np.mean(finals)
        sem = np.std(finals, ddof=1) / np.sqrt(len(finals))
        print(f"  Batch {bsz}: {mean:.4f} ± {sem:.4f} (n={len(finals)})")

print("\nPlotting complete; aggregated figures saved to", working_dir)

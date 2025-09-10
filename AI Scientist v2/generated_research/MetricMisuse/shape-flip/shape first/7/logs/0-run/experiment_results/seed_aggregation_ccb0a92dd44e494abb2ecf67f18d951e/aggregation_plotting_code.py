import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# List of experiment result files (relative to $AI_SCIENTIST_ROOT)
experiment_data_path_list = [
    "experiments/2025-08-14_19-19-19_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_cc7b62d66fea4b4cb153329ca4cd764a_proc_2717961/experiment_data.npy",
    "experiments/2025-08-14_19-19-19_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_14e8d09f317240248a596d30b8279973_proc_2717959/experiment_data.npy",
    "experiments/2025-08-14_19-19-19_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_36724a5812314db28cf4894f862f1ce8_proc_2717960/experiment_data.npy",
]

# ------------------------------------------------------------------
# Load every experiment file
all_experiment_data = []
for p in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp_dat = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_dat)
    except Exception as e:
        print(f"Error loading {p}: {e}")

if not all_experiment_data:
    print("No experiment data loaded; aborting plots.")
    exit()

# ------------------------------------------------------------------
# Aggregate by embedding dimension
agg = {}  # {embed_key: {"loss_tr": [runs], "loss_val": [runs], ...}}
for exp_dat in all_experiment_data:
    for ed_key, ed_dict in exp_dat.get("embed_dim", {}).items():
        if ed_key not in agg:
            agg[ed_key] = {
                "loss_tr": [],
                "loss_val": [],
                "hwa_tr": [],
                "hwa_val": [],
                "test_acc": [],
            }
        # extract lists of (epoch, value)
        lt = [v for _, v in ed_dict["losses"]["train"]]
        lv = [v for _, v in ed_dict["losses"]["val"]]
        ht = [v for _, v in ed_dict["metrics"]["train"]]
        hv = [v for _, v in ed_dict["metrics"]["val"]]
        agg[ed_key]["loss_tr"].append(np.array(lt))
        agg[ed_key]["loss_val"].append(np.array(lv))
        agg[ed_key]["hwa_tr"].append(np.array(ht))
        agg[ed_key]["hwa_val"].append(np.array(hv))
        g, p = ed_dict["ground_truth"], ed_dict["predictions"]
        acc = (
            np.nan
            if len(g) == 0
            else sum(int(gt == pr) for gt, pr in zip(g, p)) / len(g)
        )
        agg[ed_key]["test_acc"].append(acc)

# sort keys numerically
exp_keys = sorted(agg.keys(), key=lambda k: int(k.split("ed")[-1]))
epochs = None
for k in exp_keys:
    if len(agg[k]["loss_tr"]) > 0:
        epochs = np.arange(len(agg[k]["loss_tr"][0]))
        break


# helper to compute mean & sem (handles ragged arrays by trimming to min len)
def mean_sem(list_of_arrays):
    if not list_of_arrays:
        return None, None
    min_len = min(len(a) for a in list_of_arrays)
    arr = np.stack([a[:min_len] for a in list_of_arrays], axis=0)
    mean = arr.mean(axis=0)
    sem = (
        arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        if arr.shape[0] > 1
        else np.zeros_like(mean)
    )
    return mean, sem


# ------------------------------------------------------------------
# Figure 1: Loss curves with SEM bands
try:
    plt.figure(figsize=(6, 8))
    plt.subplot(2, 1, 1)
    for k in exp_keys:
        m, s = mean_sem(agg[k]["loss_tr"])
        if m is None:
            continue
        plt.plot(epochs[: len(m)], m, label=k)
        plt.fill_between(epochs[: len(m)], m - s, m + s, alpha=0.2)
    plt.title("SPR_BENCH – Training Loss (mean ± SEM)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy")
    plt.legend(fontsize="small")

    plt.subplot(2, 1, 2)
    for k in exp_keys:
        m, s = mean_sem(agg[k]["loss_val"])
        if m is None:
            continue
        plt.plot(epochs[: len(m)], m, label=k)
        plt.fill_between(epochs[: len(m)], m - s, m + s, alpha=0.2)
    plt.title("Validation Loss (mean ± SEM)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy")
    plt.legend(fontsize="small")

    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "spr_bench_loss_curves_mean_sem.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss curves: {e}")
    plt.close()

# ------------------------------------------------------------------
# Figure 2: HWA curves with SEM bands
try:
    plt.figure(figsize=(6, 8))
    plt.subplot(2, 1, 1)
    for k in exp_keys:
        m, s = mean_sem(agg[k]["hwa_tr"])
        if m is None:
            continue
        plt.plot(epochs[: len(m)], m, label=k)
        plt.fill_between(epochs[: len(m)], m - s, m + s, alpha=0.2)
    plt.title("SPR_BENCH – Training HWA (mean ± SEM)")
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.legend(fontsize="small")

    plt.subplot(2, 1, 2)
    for k in exp_keys:
        m, s = mean_sem(agg[k]["hwa_val"])
        if m is None:
            continue
        plt.plot(epochs[: len(m)], m, label=k)
        plt.fill_between(epochs[: len(m)], m - s, m + s, alpha=0.2)
    plt.title("Validation HWA (mean ± SEM)")
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.legend(fontsize="small")

    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "spr_bench_hwa_curves_mean_sem.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated HWA curves: {e}")
    plt.close()

# ------------------------------------------------------------------
# Figure 3: Test accuracy bar chart with SEM
try:
    dims = [int(k.split("ed")[-1]) for k in exp_keys]
    means = [np.nanmean(agg[k]["test_acc"]) for k in exp_keys]
    sems = [
        (
            np.nanstd(agg[k]["test_acc"], ddof=1) / np.sqrt(len(agg[k]["test_acc"]))
            if len(agg[k]["test_acc"]) > 1
            else 0
        )
        for k in exp_keys
    ]

    plt.figure(figsize=(6, 4))
    plt.bar([str(d) for d in dims], means, yerr=sems, capsize=5, color="skyblue")
    plt.title("SPR_BENCH – Test Accuracy by Embedding Dim (mean ± SEM)")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Accuracy")
    for i, (m, se) in enumerate(zip(means, sems)):
        plt.text(i, m + se + 0.01, f"{m:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "spr_bench_test_accuracy_mean_sem.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated accuracy bar chart: {e}")
    plt.close()

# ------------------------------------------------------------------
# Print evaluation metric
print("Test Accuracy (mean ± SEM) by embed_dim:")
for k, d in zip(exp_keys, dims):
    mean_acc = means[exp_keys.index(k)]
    sem_acc = sems[exp_keys.index(k)]
    print(f"  {k} (dim={d}): {mean_acc:.4f} ± {sem_acc:.4f}")

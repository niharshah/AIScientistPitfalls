import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
import os
import matplotlib.pyplot as plt
import numpy as np

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------ load all npy files
experiment_data_path_list = [
    "experiments/2025-08-16_01-26-03_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_dee5847daef949f2aa9e39ab7d5427cc_proc_3092031/experiment_data.npy",
    "experiments/2025-08-16_01-26-03_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_f34568ed388d445ebc8b95dcde6bc5fd_proc_3092030/experiment_data.npy",
    "experiments/2025-08-16_01-26-03_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_95a0ae473bef48648e4f8702853963ad_proc_3092028/experiment_data.npy",
]

all_experiment_data = []
try:
    root = os.getenv("AI_SCIENTIST_ROOT", os.getcwd())
    for p in experiment_data_path_list:
        full_p = os.path.join(root, p)
        if not os.path.isfile(full_p):
            print(f"File not found: {full_p}")
            continue
        all_experiment_data.append(np.load(full_p, allow_pickle=True).item())
    if len(all_experiment_data) == 0:
        raise RuntimeError("No experiment_data loaded.")
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []


# ------------------------------------------------ aggregate helper
def collect_arrays(key_chain):
    """
    key_chain is a list like ['hidden_dim_tuning','SPR',str(hd),'losses','train']
    returns a dict: hd -> list_of_arrays_across_seeds
    """
    out = {}
    for exp in all_experiment_data:
        try:
            d = exp
            for k in key_chain:
                d = d[k]
            hd = key_chain[2]  # already str
            out.setdefault(hd, []).append(np.asarray(d))
        except KeyError:
            continue
    return out


def agg_mean_sem(list_of_arrays):
    arr = np.stack(list_of_arrays, axis=0)  # (n_seeds, n_epochs)
    min_len = min(a.shape[-1] for a in list_of_arrays)
    arr = arr[:, :min_len]
    mean = arr.mean(axis=0)
    sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
    return mean, sem


# ------------------------------------------------ discover hidden dims present in ALL files
hidden_dims_all = set()
for exp in all_experiment_data:
    try:
        hds = set(exp["hidden_dim_tuning"]["SPR"].keys())
        hidden_dims_all.update(hds)
    except KeyError:
        continue
hidden_dims = sorted(int(hd) for hd in hidden_dims_all)  # ints for order
hidden_dims = [str(hd) for hd in hidden_dims]  # keep as str later
colors = plt.cm.tab10.colors

# ==================================================== PLOTS
# 1) Loss curves (train & val) with mean±SEM
try:
    plt.figure()
    for i, hd in enumerate(hidden_dims):
        # train
        tr_lists = collect_arrays(
            ["hidden_dim_tuning", "SPR", hd, "losses", "train"]
        ).get(hd, [])
        val_lists = collect_arrays(
            ["hidden_dim_tuning", "SPR", hd, "losses", "val"]
        ).get(hd, [])
        if len(tr_lists) == 0 or len(val_lists) == 0:
            continue
        tr_mean, tr_sem = agg_mean_sem(tr_lists)
        val_mean, val_sem = agg_mean_sem(val_lists)
        epochs = np.arange(1, len(tr_mean) + 1)
        plt.plot(
            epochs,
            tr_mean,
            linestyle="--",
            color=colors[i % 10],
            label=f"{hd}-train mean",
        )
        plt.fill_between(
            epochs,
            tr_mean - tr_sem,
            tr_mean + tr_sem,
            color=colors[i % 10],
            alpha=0.2,
        )
        plt.plot(
            epochs,
            val_mean,
            linestyle="-",
            color=colors[i % 10],
            label=f"{hd}-val mean",
        )
        plt.fill_between(
            epochs,
            val_mean - val_sem,
            val_mean + val_sem,
            color=colors[i % 10],
            alpha=0.2,
        )
    plt.xlabel("Fine-tuning Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR Loss Curves (mean ± SEM)\nLeft: Train (--), Right: Validation (—)")
    plt.legend(fontsize=8, ncol=2)
    fname = os.path.join(working_dir, "SPR_loss_mean_sem_vs_epoch_hidden_dims.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

# 2) CWA curves (val) mean ± SEM
try:
    plt.figure()
    for i, hd in enumerate(hidden_dims):
        val_lists = collect_arrays(
            ["hidden_dim_tuning", "SPR", hd, "metrics", "val"]
        ).get(hd, [])
        if len(val_lists) == 0:
            continue
        val_mean, val_sem = agg_mean_sem(val_lists)
        epochs = np.arange(1, len(val_mean) + 1)
        plt.plot(
            epochs,
            val_mean,
            color=colors[i % 10],
            label=f"hd={hd} mean",
        )
        plt.fill_between(
            epochs,
            val_mean - val_sem,
            val_mean + val_sem,
            color=colors[i % 10],
            alpha=0.25,
        )
    plt.xlabel("Fine-tuning Epoch")
    plt.ylabel("Color-Weighted Accuracy")
    plt.title("SPR Validation CWA Across Epochs (mean ± SEM)\nDataset: SPR")
    plt.legend(fontsize=8)
    fname = os.path.join(working_dir, "SPR_val_CWA_mean_sem_vs_epoch_hidden_dims.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating aggregated CWA plot: {e}")
    plt.close()

# 3) AIS curves (val) mean ± SEM
try:
    plt.figure()
    for i, hd in enumerate(hidden_dims):
        ais_lists = collect_arrays(["hidden_dim_tuning", "SPR", hd, "AIS", "val"]).get(
            hd, []
        )
        if len(ais_lists) == 0:
            continue
        ais_mean, ais_sem = agg_mean_sem(ais_lists)
        epochs = np.arange(1, len(ais_mean) + 1)
        plt.plot(
            epochs,
            ais_mean,
            color=colors[i % 10],
            label=f"hd={hd} mean",
        )
        plt.fill_between(
            epochs,
            ais_mean - ais_sem,
            ais_mean + ais_sem,
            color=colors[i % 10],
            alpha=0.25,
        )
    plt.xlabel("Fine-tuning Epoch")
    plt.ylabel("AIS")
    plt.title("SPR Validation AIS Across Epochs (mean ± SEM)\nDataset: SPR")
    plt.legend(fontsize=8)
    fname = os.path.join(working_dir, "SPR_val_AIS_mean_sem_vs_epoch_hidden_dims.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating aggregated AIS plot: {e}")
    plt.close()

# 4) Summary bar plot of best Val CWA per hidden dim
try:
    best_means = []
    best_sems = []
    labels = []
    for hd in hidden_dims:
        val_lists = collect_arrays(
            ["hidden_dim_tuning", "SPR", hd, "metrics", "val"]
        ).get(hd, [])
        if len(val_lists) == 0:
            continue
        best_vals = [v.max() for v in val_lists]
        labels.append(hd)
        best_means.append(np.mean(best_vals))
        best_sems.append(np.std(best_vals, ddof=1) / np.sqrt(len(best_vals)))
        print(
            f"Hidden dim {hd:>4}: best Val CWA mean = {best_means[-1]:.3f} ± {best_sems[-1]:.3f}"
        )
    x = np.arange(len(labels))
    plt.figure()
    plt.bar(x, best_means, yerr=best_sems, capsize=4, color="skyblue", edgecolor="k")
    plt.xticks(x, labels)
    plt.xlabel("Hidden Dimension")
    plt.ylabel("Best Validation CWA")
    plt.title("Best SPR Validation CWA (mean ± SEM across seeds)")
    fname = os.path.join(working_dir, "SPR_best_val_CWA_barplot_mean_sem.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating best CWA bar plot: {e}")
    plt.close()

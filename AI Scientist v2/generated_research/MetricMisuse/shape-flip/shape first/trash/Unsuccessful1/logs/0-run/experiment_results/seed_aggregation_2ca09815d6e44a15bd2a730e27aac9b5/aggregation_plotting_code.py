import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# ------------------ helper to aggregate ---------------------------
def agg_time_series(series_list):
    """
    Given a list of 1-D arrays (potentially different length) return
    mean and standard error truncated to the minimum common length.
    """
    if not series_list:
        return None, None
    min_len = min(len(s) for s in series_list)
    arr = np.stack([s[:min_len] for s in series_list], axis=0)
    mean = arr.mean(axis=0)
    sem = (
        arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        if arr.shape[0] > 1
        else np.zeros_like(mean)
    )
    return mean, sem


# -------------------- load ALL experiment logs --------------------
try:
    experiment_data_path_list = [
        "experiments/2025-08-15_14-47-52_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_c345a3a438b1413cb1f9f75e8086630b_proc_2918275/experiment_data.npy",
        "experiments/2025-08-15_14-47-52_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_c5d4d37a6aaa4598b7049d6608458c99_proc_2918273/experiment_data.npy",
        "experiments/2025-08-15_14-47-52_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_69708e14e60a42d9ba7eb527a3385154_proc_2918274/experiment_data.npy",
    ]
    all_experiment_data = []
    root = os.getenv("AI_SCIENTIST_ROOT", "")
    for p in experiment_data_path_list:
        path_full = os.path.join(root, p)
        all_experiment_data.append(np.load(path_full, allow_pickle=True).item())
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# ------------------ gather per-run values -------------------------
losses_tr = defaultdict(list)
losses_val = defaultdict(list)
hwa_val = defaultdict(list)
test_metrics = defaultdict(list)

for exp_idx, experiment_data in enumerate(all_experiment_data):
    emb_dict = experiment_data.get("embedding_dim", {})
    if not emb_dict:
        continue
    for dim_key, dim_entry in emb_dict.items():
        dim = int(dim_key.split("_")[-1])
        log = dim_entry["SPR_BENCH"]

        losses_tr[dim].append(np.asarray(log["losses"]["train"], dtype=float))
        losses_val[dim].append(np.asarray(log["losses"]["val"], dtype=float))

        # gather per-epoch validation metrics
        h_list = [m["HWA"] for m in log["metrics"]["val"]]
        hwa_val[dim].append(np.asarray(h_list, dtype=float))

        # final test metrics
        test_metrics[dim].append(log["metrics"]["test"])

dims = sorted(losses_tr.keys())
if not dims:
    print("No logs found across provided files.")
    exit(0)

# ------------- compute aggregated statistics ----------------------
train_mean, train_sem = {}, {}
val_mean, val_sem = {}, {}
hwa_mean, hwa_sem = {}, {}

for d in dims:
    train_mean[d], train_sem[d] = agg_time_series(losses_tr[d])
    val_mean[d], val_sem[d] = agg_time_series(losses_val[d])
    hwa_mean[d], hwa_sem[d] = agg_time_series(hwa_val[d])

# aggregate final test metrics
test_mean, test_sem = {}, {}
for d in dims:
    runs = test_metrics[d]
    keys = runs[0].keys()
    test_mean[d] = {k: np.mean([r[k] for r in runs]) for k in keys}
    test_sem[d] = {
        k: (
            np.std([r[k] for r in runs], ddof=1) / np.sqrt(len(runs))
            if len(runs) > 1
            else 0.0
        )
        for k in keys
    }

# --------------------- PLOT 1: loss curves (mean ± sem) -----------
try:
    plt.figure()
    for d in dims:
        epochs = np.arange(1, len(train_mean[d]) + 1)
        # training
        plt.plot(epochs, train_mean[d], "--", label=f"train dim={d}")
        plt.fill_between(
            epochs,
            train_mean[d] - train_sem[d],
            train_mean[d] + train_sem[d],
            alpha=0.2,
        )
        # validation
        plt.plot(epochs, val_mean[d], "-", label=f"val dim={d}")
        plt.fill_between(
            epochs, val_mean[d] - val_sem[d], val_mean[d] + val_sem[d], alpha=0.2
        )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Mean Training vs Validation Loss (±SEM)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves_aggregated.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss curve plot: {e}")
    plt.close()

# --------------------- PLOT 2: validation HWA (mean ± sem) --------
try:
    plt.figure()
    for d in dims:
        epochs = np.arange(1, len(hwa_mean[d]) + 1)
        plt.plot(epochs, hwa_mean[d], marker="o", label=f"dim={d}")
        plt.fill_between(
            epochs, hwa_mean[d] - hwa_sem[d], hwa_mean[d] + hwa_sem[d], alpha=0.2
        )
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.title("SPR_BENCH: Mean Validation HWA (±SEM)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_HWA_curves_aggregated.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated HWA curve plot: {e}")
    plt.close()

# ----------- PLOT 3: final test SWA/CWA/HWA grouped with error ----
try:
    x = np.arange(len(dims))
    width = 0.25
    swa_mean = [test_mean[d]["SWA"] for d in dims]
    cwa_mean = [test_mean[d]["CWA"] for d in dims]
    hwa_mean_bar = [test_mean[d]["HWA"] for d in dims]

    swa_sem = [test_sem[d]["SWA"] for d in dims]
    cwa_sem = [test_sem[d]["CWA"] for d in dims]
    hwa_sem_bar = [test_sem[d]["HWA"] for d in dims]

    plt.figure()
    plt.bar(x - width, swa_mean, width, yerr=swa_sem, capsize=3, label="SWA")
    plt.bar(x, cwa_mean, width, yerr=cwa_sem, capsize=3, label="CWA")
    plt.bar(x + width, hwa_mean_bar, width, yerr=hwa_sem_bar, capsize=3, label="HWA")
    plt.xticks(x, [str(d) for d in dims])
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Score")
    plt.title("SPR_BENCH: Mean Test Metrics (±SEM) by Embedding Dimension")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_test_metrics_grouped_aggregated.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated grouped bar plot: {e}")
    plt.close()

# ------------- PLOT 4: final test HWA only (mean ± sem) -----------
try:
    plt.figure()
    plt.bar(
        [str(d) for d in dims],
        hwa_mean_bar,
        yerr=hwa_sem_bar,
        capsize=3,
        color="steelblue",
    )
    best_dim = dims[int(np.argmax(hwa_mean_bar))]
    plt.xlabel("Embedding Dimension")
    plt.ylabel("HWA")
    plt.title("SPR_BENCH: Mean Test HWA per Embedding Dim (±SEM)")
    fname = os.path.join(working_dir, "SPR_BENCH_test_HWA_bar_aggregated.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated HWA bar plot: {e}")
    plt.close()

# ---------------------- print summary metrics ----------------------
print("Aggregated final test metrics (mean ± std) by embedding dimension:")
for d in dims:
    runs = test_metrics[d]
    std_vals = {k: np.std([r[k] for r in runs], ddof=1) for k in runs[0].keys()}
    print(
        f"dim={d}: "
        + ", ".join(
            [
                f"{k}={test_mean[d][k]:.4f} ± {std_vals[k]:.4f}"
                for k in test_mean[d].keys()
            ]
        )
    )

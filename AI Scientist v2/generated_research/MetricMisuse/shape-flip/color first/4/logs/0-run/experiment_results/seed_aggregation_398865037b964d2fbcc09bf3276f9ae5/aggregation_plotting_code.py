import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------
# 1. Load every experiment_data.npy that the user listed
experiment_data_path_list = [
    "experiments/2025-08-30_19-33-09_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_29404fc29ac5495093823b79b1ba5244_proc_1480339/experiment_data.npy",
    "experiments/2025-08-30_19-33-09_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_a361040fe51b4f7b92ad3c20b15a1dac_proc_1480341/experiment_data.npy",
    "experiments/2025-08-30_19-33-09_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_b7fd56ea0fda468ea051e55b25df9cb6_proc_1480338/experiment_data.npy",
]
all_runs = []
for p in experiment_data_path_list:
    try:
        data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        if "SPR" in data:
            all_runs.append(data["SPR"])
        else:
            print(f"Key 'SPR' not in {p}, skipping.")
    except Exception as e:
        print(f"Error loading {p}: {e}")

n_runs = len(all_runs)
if n_runs == 0:
    print("No runs found — nothing to plot.")
    quit()


# ---------------------------------------------------------------
# Helper
def _save_close(fig_name):
    plt.savefig(os.path.join(working_dir, fig_name))
    plt.close()


# ---------------------------------------------------------------
# 2. Aggregate helper (align to shortest length)
def _stack_metric(list_of_arrays):
    min_len = min(len(a) for a in list_of_arrays)
    trimmed = np.stack([np.asarray(a)[:min_len] for a in list_of_arrays], axis=0)
    mean = trimmed.mean(axis=0)
    sem = (
        trimmed.std(axis=0, ddof=1) / np.sqrt(trimmed.shape[0])
        if trimmed.shape[0] > 1
        else np.zeros_like(mean)
    )
    return mean, sem


# ---------------------------------------------------------------
# 3. Training / validation loss curves (mean ± SEM)
try:
    train_losses = [run["losses"]["train"] for run in all_runs]
    val_losses = [run["losses"]["val"] for run in all_runs]
    tr_mean, tr_sem = _stack_metric(train_losses)
    val_mean, val_sem = _stack_metric(val_losses)
    epochs = np.arange(1, len(tr_mean) + 1)

    plt.figure()
    plt.errorbar(epochs, tr_mean, yerr=tr_sem, label="Train (mean ± SEM)", capsize=2)
    plt.errorbar(epochs, val_mean, yerr=val_sem, label="Val (mean ± SEM)", capsize=2)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR: Aggregated Training vs Validation Loss")
    plt.legend()
    _save_close("SPR_loss_curves_mean_sem.png")
except Exception as e:
    print(f"Error creating aggregated loss curve plot: {e}")
    plt.close()

# ---------------------------------------------------------------
# 4. Validation metrics (CWA, SWA, CompWA) curves with SEM
try:
    cwa_runs, swa_runs, comp_runs = [], [], []
    for run in all_runs:
        vals = run["metrics"]["val"]  # list[dict] per epoch
        cwa_runs.append([ep["CWA"] for ep in vals])
        swa_runs.append([ep["SWA"] for ep in vals])
        comp_runs.append([ep["CompWA"] for ep in vals])

    cwa_mean, cwa_sem = _stack_metric(cwa_runs)
    swa_mean, swa_sem = _stack_metric(swa_runs)
    comp_mean, comp_sem = _stack_metric(comp_runs)
    epochs = np.arange(1, len(cwa_mean) + 1)

    plt.figure()
    plt.errorbar(epochs, cwa_mean, yerr=cwa_sem, label="CWA (mean ± SEM)", capsize=2)
    plt.errorbar(epochs, swa_mean, yerr=swa_sem, label="SWA (mean ± SEM)", capsize=2)
    plt.errorbar(
        epochs, comp_mean, yerr=comp_sem, label="CompWA (mean ± SEM)", capsize=2
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR: Aggregated Validation Weighted Accuracies")
    plt.legend()
    _save_close("SPR_val_weighted_accuracies_mean_sem.png")
except Exception as e:
    print(f"Error creating aggregated val metric plot: {e}")
    plt.close()

# ---------------------------------------------------------------
# 5. Final test metrics bar chart (mean ± SEM)
try:
    metrics_keys = list(all_runs[0]["metrics"]["test"].keys())
    values_per_run = {k: [] for k in metrics_keys}
    for run in all_runs:
        for k in metrics_keys:
            values_per_run[k].append(run["metrics"]["test"][k])

    means = [np.mean(values_per_run[k]) for k in metrics_keys]
    sems = [
        np.std(values_per_run[k], ddof=1) / np.sqrt(n_runs) if n_runs > 1 else 0
        for k in metrics_keys
    ]

    plt.figure()
    x = np.arange(len(metrics_keys))
    plt.bar(
        x, means, yerr=sems, capsize=5, color=["tab:blue", "tab:orange", "tab:green"]
    )
    plt.ylim(0, 1)
    plt.title("SPR: Aggregated Final Test Weighted Accuracies")
    plt.xticks(x, metrics_keys)
    for i, (m, s) in enumerate(zip(means, sems)):
        plt.text(i, m + 0.02, f"{m:.2f}±{s:.2f}", ha="center")
    _save_close("SPR_test_weighted_accuracies_mean_sem.png")

    # print aggregated numbers to console
    print("Aggregated SPR Test Metrics (mean ± SEM):")
    for k, m, s in zip(metrics_keys, means, sems):
        print(f"  {k}: {m:.4f} ± {s:.4f}")
except Exception as e:
    print(f"Error creating aggregated test bar plot: {e}")
    plt.close()

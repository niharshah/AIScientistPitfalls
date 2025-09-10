import matplotlib.pyplot as plt
import numpy as np
import os
from math import sqrt

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# List of experiment_data paths supplied in the prompt
experiment_data_path_list = [
    "experiments/2025-08-30_19-33-09_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_d0ce7bc8b4404b45aa7ca39394c67617_proc_1476164/experiment_data.npy",
    "experiments/2025-08-30_19-33-09_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_faad5059562f436893684732d2b8e1bd_proc_1476163/experiment_data.npy",
    "experiments/2025-08-30_19-33-09_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_7dc05cdfffc941b9b48a706aac26ebd3_proc_1476162/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        full_path = p
        if not os.path.isabs(p):
            root = os.getenv("AI_SCIENTIST_ROOT", "")
            full_path = os.path.join(root, p)
        exp_data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_data)
    except Exception as e:
        print(f"Error loading {p}: {e}")

if len(all_experiment_data) == 0:
    print("No experiment data could be loaded – aborting.")
    import sys

    sys.exit(0)

# ------------------------------------------------------------------
# Collect epoch budgets available across runs
spr_runs = []
for exp in all_experiment_data:
    try:
        spr_runs.append(exp["epochs"]["SPR_BENCH"])
    except Exception as e:
        print(f"SPR_BENCH data not found in one run: {e}")

if len(spr_runs) == 0:
    print("No SPR_BENCH data found – nothing to plot.")
    import sys

    sys.exit(0)

# Use keys from first run as reference, but keep only those present in at least one run
epoch_budgets = set()
for d in spr_runs:
    epoch_budgets.update(d.keys())
epoch_budgets = sorted(epoch_budgets, key=lambda x: int(x))[:4]  # safeguard: max 4

num_runs = len(spr_runs)


# ------------------------------------------------------------------
# Helper to stack metric histories and compute mean & stderr
def aggregate_metric(spr_runs, ep_key, path_list):
    """
    path_list: list of keys to access nested dict from spr_runs[i][ep_key]
               e.g. ['losses', 'train']
    Returns mean, stderr (np.arrays) truncated to minimal length
    """
    series = []
    for spr in spr_runs:
        try:
            node = spr[ep_key]
            for p in path_list:
                node = node[p]
            series.append(np.asarray(node, dtype=float))
        except KeyError:
            continue
    if len(series) == 0:
        return None, None
    min_len = min(len(s) for s in series)
    series = np.stack([s[:min_len] for s in series], axis=0)  # shape (runs, steps)
    mean = series.mean(axis=0)
    stderr = series.std(axis=0, ddof=0) / np.sqrt(series.shape[0])
    return mean, stderr


# ------------------------------------------------------------------
# FIGURES 1-4: aggregated loss curves per epoch budget
for ep_key in epoch_budgets:
    try:
        train_mean, train_se = aggregate_metric(spr_runs, ep_key, ["losses", "train"])
        val_mean, val_se = aggregate_metric(spr_runs, ep_key, ["losses", "val"])

        if train_mean is None or val_mean is None:
            print(f"Skipping loss plot for {ep_key}: missing data")
            continue

        epochs = np.arange(1, len(train_mean) + 1)

        plt.figure()
        # Train loss
        plt.plot(epochs, train_mean, color="tab:blue", label="Train Loss (mean)")
        plt.fill_between(
            epochs,
            train_mean - train_se,
            train_mean + train_se,
            color="tab:blue",
            alpha=0.25,
            label="Train ± SE",
        )
        # Val loss
        plt.plot(epochs, val_mean, color="tab:orange", label="Val Loss (mean)")
        plt.fill_between(
            epochs,
            val_mean - val_se,
            val_mean + val_se,
            color="tab:orange",
            alpha=0.25,
            label="Val ± SE",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"SPR_BENCH Aggregated Train vs Val Loss ({ep_key} Epochs)")
        plt.legend()
        fname = f"SPR_BENCH_agg_loss_{ep_key}epochs.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {ep_key}: {e}")
        plt.close()

# ------------------------------------------------------------------
# FIGURE 5: aggregated validation CWA comparison across budgets
try:
    plt.figure()
    for ep_key in epoch_budgets:
        cwa_mean, cwa_se = aggregate_metric(spr_runs, ep_key, ["metrics", "val_cwa2"])
        if cwa_mean is None:
            print(f"Skipping CWA for {ep_key}: missing data")
            continue
        steps = np.arange(1, len(cwa_mean) + 1)
        plt.plot(steps, cwa_mean, label=f"{ep_key} Epochs (mean)")
        plt.fill_between(steps, cwa_mean - cwa_se, cwa_mean + cwa_se, alpha=0.25)

        # print final epoch aggregated scores
        print(
            f"{ep_key} epochs – final Val CWA2: {cwa_mean[-1]:.4f} ± {cwa_se[-1]:.4f}"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Complexity-Weighted Accuracy")
    plt.title("SPR_BENCH Validation CWA (mean ± SE) Across Epoch Budgets")
    plt.legend()
    fname = "SPR_BENCH_agg_val_CWA_comparison.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated CWA plot: {e}")
    plt.close()

import matplotlib.pyplot as plt
import numpy as np
import os
import math

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------------------------------
# 1. Load every experiment_data.npy that was listed
# -------------------------------------------------------------------------
try:
    experiment_data_path_list = [
        "experiments/2025-08-17_18-48-09_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_4132b97f5b3d4914a0fd75cf44368225_proc_3330988/experiment_data.npy",
        "experiments/2025-08-17_18-48-09_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_f461584c3de14c95a4fe092985d5a596_proc_3330989/experiment_data.npy",
        "experiments/2025-08-17_18-48-09_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_60be6ebb88b242b1ba0754fe3f82a233_proc_3330987/experiment_data.npy",
    ]
    all_experiment_data = []
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        edict = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(edict)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# -------------------------------------------------------------------------
# 2. Gather metrics across runs
# -------------------------------------------------------------------------
from collections import defaultdict

test_f1_by_dropout = defaultdict(list)
train_curves_by_dropout = defaultdict(list)
val_curves_by_dropout = defaultdict(list)
epochs_by_dropout = defaultdict(list)  # keep epoch vectors (assume identical per run)

for run in all_experiment_data:
    drop_dict = run.get("dropout_rate", {})
    for d_rate, rec in drop_dict.items():
        test_f1_by_dropout[d_rate].append(rec["test_macro_f1"])
        train_curves_by_dropout[d_rate].append(
            np.array(rec["metrics"]["train_macro_f1"])
        )
        val_curves_by_dropout[d_rate].append(np.array(rec["metrics"]["val_macro_f1"]))
        epochs_by_dropout[d_rate].append(np.array(rec["epochs"]))

dropouts = sorted(test_f1_by_dropout.keys())


# helper to compute sem
def sem(arr):
    n = len(arr)
    return np.std(arr, ddof=1) / math.sqrt(n) if n > 1 else 0.0


# -------------------------------------------------------------------------
# 3. Plot aggregated Test Macro-F1 with error bars
# -------------------------------------------------------------------------
try:
    plt.figure(figsize=(7, 4))
    means = [np.mean(test_f1_by_dropout[d]) for d in dropouts]
    errs = [sem(test_f1_by_dropout[d]) for d in dropouts]
    plt.bar(
        [str(d) for d in dropouts],
        means,
        yerr=errs,
        capsize=5,
        color="skyblue",
        label="mean ± SEM",
    )
    plt.ylabel("Macro-F1")
    plt.xlabel("Dropout rate")
    plt.title(
        "SPR_BENCH: Mean Test Macro-F1 vs Dropout (n={})".format(
            len(all_experiment_data)
        )
    )
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "spr_bench_mean_test_F1_vs_dropout.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated bar plot: {e}")
    plt.close()

# -------------------------------------------------------------------------
# 4. Plot averaged learning curves (≤3 rates to keep ≤5 figures total)
# -------------------------------------------------------------------------
max_curve_plots = 3
for idx, d in enumerate(dropouts[:max_curve_plots]):
    try:
        # Align epochs across runs (truncate to shortest)
        curves_tr = train_curves_by_dropout[d]
        curves_val = val_curves_by_dropout[d]
        min_len = min(c.shape[0] for c in curves_tr + curves_val)
        tr_stack = np.stack([c[:min_len] for c in curves_tr], axis=0)
        val_stack = np.stack([c[:min_len] for c in curves_val], axis=0)
        epochs = epochs_by_dropout[d][0][:min_len]  # take first run's epoch vector

        tr_mean, tr_sem = tr_stack.mean(axis=0), tr_stack.std(
            axis=0, ddof=1
        ) / math.sqrt(tr_stack.shape[0])
        val_mean, val_sem = val_stack.mean(axis=0), val_stack.std(
            axis=0, ddof=1
        ) / math.sqrt(val_stack.shape[0])

        plt.figure(figsize=(7, 4))
        plt.plot(epochs, tr_mean, label="Train mean", color="tab:blue")
        plt.fill_between(
            epochs, tr_mean - tr_sem, tr_mean + tr_sem, alpha=0.3, color="tab:blue"
        )
        plt.plot(epochs, val_mean, label="Val mean", color="tab:orange")
        plt.fill_between(
            epochs,
            val_mean - val_sem,
            val_mean + val_sem,
            alpha=0.3,
            color="tab:orange",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.legend()
        plt.title(
            f"SPR_BENCH Macro-F1 Curves (dropout={d})\nMean ± SEM across {tr_stack.shape[0]} runs"
        )
        plt.tight_layout()
        fname = os.path.join(working_dir, f"spr_bench_macro_F1_dropout_{d}_avg.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating averaged curve for dropout={d}: {e}")
        plt.close()

# -------------------------------------------------------------------------
# 5. Print out numeric summary
# -------------------------------------------------------------------------
for d in dropouts:
    print(
        f"Dropout {d}: mean Test-F1={np.mean(test_f1_by_dropout[d]):.4f}  ±SEM={sem(test_f1_by_dropout[d]):.4f}"
    )

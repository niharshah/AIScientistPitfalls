import matplotlib.pyplot as plt
import numpy as np
import os
from math import sqrt

# ------------------------------------------------------------------
# directory handling
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# list of experiment .npy files (relative to AI_SCIENTIST_ROOT)
experiment_data_path_list = [
    "experiments/2025-08-15_16-42-50_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_b6263bb6965a4ed18fb902ebc2cd7d38_proc_2945037/experiment_data.npy",
    "experiments/2025-08-15_16-42-50_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_33fce08015854601941111947f50624d_proc_2945035/experiment_data.npy",
    "experiments/2025-08-15_16-42-50_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_ab688ba1e96b476594dee8a6f576cf3a_proc_2945036/experiment_data.npy",
]

# ------------------------------------------------------------------
# load data
all_experiment_data = []
try:
    ai_root = os.getenv("AI_SCIENTIST_ROOT", ".")
    for p in experiment_data_path_list:
        full_path = os.path.join(ai_root, p)
        exp_data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# ------------------------------------------------------------------
# collect runs shared across experiments (assume same naming scheme)
run_dicts = [d.get("num_epochs", {}) for d in all_experiment_data if "num_epochs" in d]
if not run_dicts:
    print("No run information found in any experiment file.")

shared_run_names = set(run_dicts[0].keys())
for rd in run_dicts[1:]:
    shared_run_names &= set(rd.keys())
shared_run_names = sorted(list(shared_run_names))

# ------------------------------------------------------------------
# aggregate losses and plot mean ± SE
for idx, run_name in enumerate(shared_run_names):
    if idx >= 5:  # plot at most 5 loss figures
        break
    try:
        # collect losses across experiments
        train_mat = []
        val_mat = []
        min_len = np.inf
        for rd in run_dicts:
            tr = np.asarray(rd[run_name]["losses"]["train"])
            va = np.asarray(rd[run_name]["losses"]["val"])
            min_len = min(min_len, len(tr), len(va))
            train_mat.append(tr)
            val_mat.append(va)
        # truncate to shortest
        train_mat = np.vstack([t[: int(min_len)] for t in train_mat])
        val_mat = np.vstack([v[: int(min_len)] for v in val_mat])

        train_mean = train_mat.mean(axis=0)
        val_mean = val_mat.mean(axis=0)
        train_se = train_mat.std(axis=0, ddof=1) / sqrt(train_mat.shape[0])
        val_se = val_mat.std(axis=0, ddof=1) / sqrt(val_mat.shape[0])
        epochs = np.arange(1, len(train_mean) + 1)

        plt.figure()
        plt.plot(epochs, train_mean, label="Train Mean", color="C0")
        plt.fill_between(
            epochs,
            train_mean - train_se,
            train_mean + train_se,
            color="C0",
            alpha=0.3,
            label="Train ±SE",
        )
        plt.plot(epochs, val_mean, label="Val Mean", color="C1")
        plt.fill_between(
            epochs,
            val_mean - val_se,
            val_mean + val_se,
            color="C1",
            alpha=0.3,
            label="Val ±SE",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(
            f"{run_name}: Mean Training & Validation Loss ±SE\nDataset: SPR_BENCH (toy)"
        )
        plt.legend()
        fname = os.path.join(working_dir, f"{run_name}_loss_mean_se_SPR_BENCH.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {run_name}: {e}")
        plt.close()

# ------------------------------------------------------------------
# aggregate final test metrics (bar plot with SE)
try:
    metrics = ["CRWA", "SWA", "CWA"]
    bar_w = 0.2
    x_positions = np.arange(len(shared_run_names))
    fig, ax = plt.subplots(figsize=(10, 4))

    # store numeric values for printing later
    aggregated_numbers = {run: {} for run in shared_run_names}

    for m_idx, m in enumerate(metrics):
        means = []
        ses = []
        for run_name in shared_run_names:
            vals = []
            for rd in run_dicts:
                test_m = rd[run_name].get("metrics", {}).get("test", {})
                if m in test_m:
                    vals.append(test_m[m])
            if not vals:
                vals = [0.0]
            means.append(np.mean(vals))
            ses.append(np.std(vals, ddof=1) / sqrt(len(vals)) if len(vals) > 1 else 0.0)
            aggregated_numbers[run_name][m] = (means[-1], ses[-1])

        bar_x = x_positions + (m_idx - 1) * bar_w
        ax.bar(bar_x, means, width=bar_w, label=m)
        ax.errorbar(bar_x, means, yerr=ses, fmt="none", ecolor="black", capsize=3, lw=1)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(shared_run_names, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Aggregated Test Metrics (Mean ±SE)\nDataset: SPR_BENCH (toy)")
    ax.legend()
    fname = os.path.join(working_dir, "aggregated_test_metrics_SPR_BENCH.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated metrics plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# print numeric summary
print("Aggregated Test Metrics (mean ± SE):")
for run_name in shared_run_names:
    vals = aggregated_numbers.get(run_name, {})
    crwa = vals.get("CRWA", (0, 0))
    swa = vals.get("SWA", (0, 0))
    cwa = vals.get("CWA", (0, 0))
    print(
        f"{run_name}: CRWA={crwa[0]:.4f}±{crwa[1]:.4f}, "
        f"SWA={swa[0]:.4f}±{swa[1]:.4f}, "
        f"CWA={cwa[0]:.4f}±{cwa[1]:.4f}"
    )

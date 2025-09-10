import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
# Basic set-up
# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Paths supplied by the task
experiment_data_path_list = [
    "experiments/2025-08-17_22-28-20_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_8796629ee0ee4696b8354597c67d165a_proc_3442580/experiment_data.npy",
    "experiments/2025-08-17_22-28-20_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_09faf3d23c114b388a04589277926081_proc_3442578/experiment_data.npy",
    "experiments/2025-08-17_22-28-20_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_05cccbfe149f4503a4ec07948432e6fe_proc_3442579/experiment_data.npy",
]

# ------------------------------------------------------------------
# Load every experiment file
# ------------------------------------------------------------------
all_experiment_data = []
root = os.getenv("AI_SCIENTIST_ROOT", ".")  # fall back to current dir
for p in experiment_data_path_list:
    try:
        obj = np.load(os.path.join(root, p), allow_pickle=True).item()
        all_experiment_data.append(obj)
    except Exception as e:
        print(f"Error loading experiment data from {p}: {e}")

if len(all_experiment_data) == 0:
    print("No experiment data found – nothing to plot.")
    quit()

# ------------------------------------------------------------------
# Aggregate across runs (dataset = SPR_BENCH, selection = 'batch_size')
# ------------------------------------------------------------------
runs = []
for exp in all_experiment_data:
    try:
        runs.append(exp["batch_size"]["SPR_BENCH"])
    except Exception as e:
        print(f"Run missing SPR_BENCH: {e}")

if len(runs) == 0:
    print("No SPR_BENCH data present.")
    quit()

# Determine the batch-sizes that are present in every run
bs_sets = [set(int(k.split("_")[-1]) for k in r.keys()) for r in runs]
common_bs = sorted(set.intersection(*bs_sets))
if len(common_bs) == 0:
    print("No common batch sizes across runs.")
    quit()

n_runs = len(runs)

# Helper containers
agg = {}
for bs in common_bs:
    agg[bs] = {"train_loss": [], "val_loss": [], "val_f1": []}

# Collect data per batch size
for r in runs:
    for bs in common_bs:
        logs = r[f"bs_{bs}"]
        agg[bs]["train_loss"].append(np.array(logs["losses"]["train"]))
        agg[bs]["val_loss"].append(np.array(logs["losses"]["val"]))
        agg[bs]["val_f1"].append(np.array(logs["metrics"]["val"]))

# Convert lists to stacked numpy arrays (trim to minimum length)
for bs in common_bs:
    for key in ["train_loss", "val_loss", "val_f1"]:
        # Find shortest epoch length for this metric across runs
        min_len = min(arr.shape[0] for arr in agg[bs][key])
        trimmed = [arr[:min_len] for arr in agg[bs][key]]
        agg[bs][key] = np.stack(trimmed, axis=0)  # shape = (runs, epochs)

# Colors for plotting
color_cycle = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

# ------------------------------------------------------------------
# 1) Mean ± SE Training / Validation Loss
# ------------------------------------------------------------------
try:
    plt.figure()
    for idx, bs in enumerate(common_bs):
        c = color_cycle[idx % len(color_cycle)]
        epochs = np.arange(1, agg[bs]["train_loss"].shape[1] + 1)

        # Train
        mean_train = agg[bs]["train_loss"].mean(axis=0)
        se_train = agg[bs]["train_loss"].std(axis=0, ddof=1) / np.sqrt(n_runs)
        plt.plot(epochs, mean_train, color=c, linestyle="-", label=f"train bs={bs}")
        plt.fill_between(
            epochs, mean_train - se_train, mean_train + se_train, color=c, alpha=0.2
        )

        # Validation
        mean_val = agg[bs]["val_loss"].mean(axis=0)
        se_val = agg[bs]["val_loss"].std(axis=0, ddof=1) / np.sqrt(n_runs)
        plt.plot(epochs, mean_val, color=c, linestyle="--", label=f"val bs={bs}")
        plt.fill_between(
            epochs, mean_val - se_val, mean_val + se_val, color=c, alpha=0.2
        )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"SPR_BENCH: Mean ± SE Training vs Validation Loss (n={n_runs})")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_aggregated_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss curves: {e}")
    plt.close()

# ------------------------------------------------------------------
# 2) Mean ± SE Validation Macro-F1
# ------------------------------------------------------------------
try:
    plt.figure()
    for idx, bs in enumerate(common_bs):
        c = color_cycle[idx % len(color_cycle)]
        epochs = np.arange(1, agg[bs]["val_f1"].shape[1] + 1)
        mean_f1 = agg[bs]["val_f1"].mean(axis=0)
        se_f1 = agg[bs]["val_f1"].std(axis=0, ddof=1) / np.sqrt(n_runs)
        plt.plot(epochs, mean_f1, color=c, label=f"val F1 bs={bs}")
        plt.fill_between(epochs, mean_f1 - se_f1, mean_f1 + se_f1, color=c, alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title(f"SPR_BENCH: Mean ± SE Validation Macro-F1 (n={n_runs})")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_aggregated_macro_f1_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated F1 curves: {e}")
    plt.close()

# ------------------------------------------------------------------
# 3) Final-epoch Macro-F1 Bar Plot with SE
# ------------------------------------------------------------------
try:
    plt.figure()
    means = []
    ses = []
    for bs in common_bs:
        final_vals = agg[bs]["val_f1"][:, -1]  # take last epoch’s F1 for every run
        means.append(final_vals.mean())
        ses.append(final_vals.std(ddof=1) / np.sqrt(n_runs))
    x_pos = np.arange(len(common_bs))
    plt.bar(
        x_pos,
        means,
        yerr=ses,
        capsize=5,
        color=[color_cycle[i % len(color_cycle)] for i in range(len(common_bs))],
    )
    plt.xticks(x_pos, [str(bs) for bs in common_bs])
    plt.xlabel("Batch Size")
    plt.ylabel("Final-Epoch Macro-F1")
    plt.title(f"SPR_BENCH: Final-Epoch Macro-F1 (Mean ± SE, n={n_runs})")
    for x, m in zip(x_pos, means):
        plt.text(x, m + 0.005, f"{m:.2f}", ha="center", va="bottom")
    fname = os.path.join(working_dir, "SPR_BENCH_aggregated_final_f1_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated final F1 bar plot: {e}")
    plt.close()

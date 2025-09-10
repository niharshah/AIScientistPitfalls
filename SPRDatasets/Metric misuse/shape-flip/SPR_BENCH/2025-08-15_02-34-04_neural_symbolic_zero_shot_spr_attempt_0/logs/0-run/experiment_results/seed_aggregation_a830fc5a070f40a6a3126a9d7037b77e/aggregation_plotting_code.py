import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

# ------------------------------------------------------------------
# basic set-up
# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# experiment paths supplied by the prompt
experiment_data_path_list = [
    "experiments/2025-08-15_02-34-04_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_021b0d3d36124e618073a4abfecd8c4b_proc_2822195/experiment_data.npy",
    "experiments/2025-08-15_02-34-04_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_26fab103ac3f40fc9b5b8aca7b2e4a24_proc_2822194/experiment_data.npy",
    "experiments/2025-08-15_02-34-04_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_e25d46ef8667430ca5dc6ef4ddd9414e_proc_2822196/experiment_data.npy",
]

all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# ------------------------------------------------------------------
# aggregate runs across all experiments
# ------------------------------------------------------------------
loss_train_dict = defaultdict(list)  # {run_name: [np.array, ...]}
loss_val_dict = defaultdict(list)
hwa_val_dict = defaultdict(list)  # validation HWA (list of arrays)
hwa_test_dict = defaultdict(list)  # final scalar

for exp in all_experiment_data:
    if "num_epochs" not in exp:
        continue
    for run_name, run in exp["num_epochs"].items():
        # store losses
        loss_train_dict[run_name].append(np.asarray(run["losses"]["train"]))
        loss_val_dict[run_name].append(np.asarray(run["losses"]["val"]))
        # store validation HWA time-series (3rd entry in metrics tuple)
        hwa_vals = np.asarray([m[2] for m in run["metrics"]["val"]])
        hwa_val_dict[run_name].append(hwa_vals)
        # store test HWA (scalar)
        hwa_test_dict[run_name].append(run["metrics"]["test"][2])


def stack_and_trim(list_of_arrays):
    """Stack 1-D arrays after trimming to shortest length"""
    min_len = min(a.shape[0] for a in list_of_arrays)
    trimmed = np.stack([a[:min_len] for a in list_of_arrays], axis=0)
    return trimmed  # shape (n_runs, min_len)


# ------------------------------------------------------------------
# 1) Mean ± stderr loss curves
# ------------------------------------------------------------------
try:
    plt.figure(figsize=(7, 4))
    for run_name in sorted(loss_train_dict.keys()):
        train_mat = stack_and_trim(loss_train_dict[run_name])
        val_mat = stack_and_trim(loss_val_dict[run_name])
        epochs = np.arange(train_mat.shape[1])

        # sample at most 50 points
        step = max(1, len(epochs) // 50)
        epochs_s = epochs[::step]

        # mean & stderr
        train_mean = train_mat.mean(0)[::step]
        train_se = train_mat.std(0, ddof=1) / np.sqrt(train_mat.shape[0])
        train_se = train_se[::step]

        val_mean = val_mat.mean(0)[::step]
        val_se = val_mat.std(0, ddof=1) / np.sqrt(val_mat.shape[0])
        val_se = val_se[::step]

        # plot
        plt.plot(epochs_s, train_mean, ls="--", label=f"{run_name}-train")
        plt.fill_between(
            epochs_s, train_mean - train_se, train_mean + train_se, alpha=0.2
        )

        plt.plot(epochs_s, val_mean, ls="-", label=f"{run_name}-val")
        plt.fill_between(epochs_s, val_mean - val_se, val_mean + val_se, alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        "SPR_BENCH Mean Loss Curves (shaded = SE)\nTrain (dashed) vs Validation (solid)"
    )
    plt.legend(fontsize=8)
    fname = os.path.join(working_dir, "spr_loss_curves_mean_se.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 2) Mean ± stderr Validation HWA curves
# ------------------------------------------------------------------
try:
    plt.figure(figsize=(7, 4))
    for run_name in sorted(hwa_val_dict.keys()):
        hwa_mat = stack_and_trim(hwa_val_dict[run_name])
        epochs = np.arange(hwa_mat.shape[1])
        step = max(1, len(epochs) // 50)
        epochs_s = epochs[::step]

        mean = hwa_mat.mean(0)[::step]
        se = hwa_mat.std(0, ddof=1) / np.sqrt(hwa_mat.shape[0])
        se = se[::step]

        plt.plot(epochs_s, mean, label=run_name)
        plt.fill_between(epochs_s, mean - se, mean + se, alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel("Validation HWA")
    plt.title("SPR_BENCH Mean Validation HWA Across Epochs (shaded = SE)")
    plt.legend(fontsize=8)
    fname = os.path.join(working_dir, "spr_val_hwa_mean_se.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating aggregated HWA plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 3) Final Test HWA Bar Chart with SE
# ------------------------------------------------------------------
try:
    plt.figure(figsize=(7, 4))
    names, means, ses = [], [], []
    for run_name in sorted(hwa_test_dict.keys()):
        vals = np.asarray(hwa_test_dict[run_name])
        names.append(run_name.replace("epochs_", "e"))
        means.append(vals.mean())
        ses.append(vals.std(ddof=1) / np.sqrt(len(vals)))
    x = np.arange(len(names))
    plt.bar(x, means, yerr=ses, capsize=4, color="skyblue")
    plt.xticks(x, names)
    plt.ylabel("Test HWA")
    plt.title("SPR_BENCH Final Test HWA (mean ± SE)")
    fname = os.path.join(working_dir, "spr_test_hwa_bar_mean_se.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating aggregated bar chart: {e}")
    plt.close()

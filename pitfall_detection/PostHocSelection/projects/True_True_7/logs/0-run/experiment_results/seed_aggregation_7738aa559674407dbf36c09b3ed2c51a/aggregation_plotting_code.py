import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
# 0) housekeeping
# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# 1) load every experiment file listed in the prompt
# ------------------------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-15_02-34-04_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_0306c76a120b464fa4da54b8f0e91b77_proc_2815569/experiment_data.npy",
    "experiments/2025-08-15_02-34-04_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_3f7526323e2744f8a769a881143c7fa9_proc_2815571/experiment_data.npy",
    "experiments/2025-08-15_02-34-04_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_dfb4dd9376bf44d48db9288a6a5bb936_proc_2815570/experiment_data.npy",
]

all_experiments = []
for p in experiment_data_path_list:
    try:
        exp = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_experiments.append(exp)
    except Exception as e:
        print(f"Error loading {p}: {e}")

# If nothing loaded we simply stop
if not all_experiments:
    print("No experiment data could be loaded – nothing to plot.")
    quit()

# ------------------------------------------------------------------
# 2) aggregate runs that share the same run_name (e.g. epochs_10)
# ------------------------------------------------------------------
runs_by_name = {}
for exp in all_experiments:
    if "num_epochs" not in exp:
        continue
    for run_name, run_dict in exp["num_epochs"].items():
        runs_by_name.setdefault(run_name, []).append(run_dict)


# ------------------------------------------------------------------
# 3) helper to pad variable-length sequences with np.nan
# ------------------------------------------------------------------
def pad_to_same_length(list_of_lists):
    max_len = max(len(x) for x in list_of_lists)
    out = np.full((len(list_of_lists), max_len), np.nan, dtype=float)
    for i, seq in enumerate(list_of_lists):
        out[i, : len(seq)] = seq
    return out


# ------------------------------------------------------------------
# 4) build epoch-level matrices for losses and HWA
# ------------------------------------------------------------------
train_losses_all, val_losses_all, hwa_vals_all = [], [], []

for run_list in runs_by_name.values():
    for run in run_list:
        train_losses_all.append(run["losses"]["train"])
        val_losses_all.append(run["losses"]["val"])
        hwa_vals_all.append([m[2] for m in run["metrics"]["val"]])

train_mat = pad_to_same_length(train_losses_all)
val_mat = pad_to_same_length(val_losses_all)
hwa_mat = pad_to_same_length(hwa_vals_all)


def mean_sem(mat, axis=0):
    mean = np.nanmean(mat, axis=axis)
    sem = np.nanstd(mat, axis=axis, ddof=1) / np.sqrt(np.sum(~np.isnan(mat), axis=axis))
    return mean, sem


# ------------------------------------------------------------------
# 5) PLOT 1 – mean loss curves with SEM
# ------------------------------------------------------------------
try:
    plt.figure(figsize=(6, 4))
    x = np.arange(train_mat.shape[1])
    train_mean, train_sem = mean_sem(train_mat)
    val_mean, val_sem = mean_sem(val_mat)

    plt.plot(x, train_mean, ls="--", color="tab:blue", label="Train – mean")
    plt.fill_between(
        x,
        train_mean - train_sem,
        train_mean + train_sem,
        color="tab:blue",
        alpha=0.2,
        label="Train – SEM",
    )

    plt.plot(x, val_mean, ls="-", color="tab:orange", label="Val – mean")
    plt.fill_between(
        x,
        val_mean - val_sem,
        val_mean + val_sem,
        color="tab:orange",
        alpha=0.2,
        label="Val – SEM",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH Mean Loss Curves\nTrain (dashed) vs Validation (solid)")
    plt.legend(fontsize=8)
    fname = os.path.join(working_dir, "spr_loss_curves_mean_sem.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 6) PLOT 2 – mean validation-HWA curve with SEM
# ------------------------------------------------------------------
try:
    plt.figure(figsize=(6, 4))
    x = np.arange(hwa_mat.shape[1])
    hwa_mean, hwa_sem = mean_sem(hwa_mat)

    # subsample to at most 50 points
    step = max(1, len(x) // 50)
    plt.plot(x[::step], hwa_mean[::step], color="tab:green", label="Val HWA – mean")
    plt.fill_between(
        x[::step],
        (hwa_mean - hwa_sem)[::step],
        (hwa_mean + hwa_sem)[::step],
        color="tab:green",
        alpha=0.25,
        label="Val HWA – SEM",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Validation HWA")
    plt.title("SPR_BENCH Mean Validation HWA Across Epochs")
    plt.legend(fontsize=8)
    fname = os.path.join(working_dir, "spr_val_hwa_mean_sem.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating aggregated HWA plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 7) PLOT 3 – final test HWA bar chart with SEM by run_name
# ------------------------------------------------------------------
try:
    plt.figure(figsize=(6, 4))
    bar_names, bar_means, bar_sems = [], [], []

    for run_name, run_list in runs_by_name.items():
        finals = [run["metrics"]["test"][2] for run in run_list]
        bar_names.append(run_name.replace("epochs_", "e"))
        bar_means.append(np.mean(finals))
        bar_sems.append(np.std(finals, ddof=1) / np.sqrt(len(finals)))

    x = np.arange(len(bar_names))
    plt.bar(x, bar_means, yerr=bar_sems, capsize=5, color="skyblue")
    plt.xticks(x, bar_names, rotation=45, ha="right")
    plt.ylabel("Test HWA")
    plt.title("SPR_BENCH Final Test HWA (mean ± SEM)")
    fname = os.path.join(working_dir, "spr_test_hwa_bar_mean_sem.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating aggregated bar chart: {e}")
    plt.close()

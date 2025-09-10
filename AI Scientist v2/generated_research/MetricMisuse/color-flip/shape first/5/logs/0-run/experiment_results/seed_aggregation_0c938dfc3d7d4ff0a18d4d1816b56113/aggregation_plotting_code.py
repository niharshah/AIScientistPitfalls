import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# List of experiment data paths relative to $AI_SCIENTIST_ROOT
experiment_data_path_list = [
    "experiments/2025-08-16_00-46-17_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_8b1d16c5a96546e481d660ca48e4700b_proc_3069202/experiment_data.npy",
    "experiments/2025-08-16_00-46-17_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_6822c11b92d8462c9e3bb65826a85a13_proc_3069201/experiment_data.npy",
    "experiments/2025-08-16_00-46-17_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_a54f9e8b65c741868428553a88742bf8_proc_3069204/experiment_data.npy",
]

all_sweeps = []
spr_key = ("supervised_finetuning_epochs", "SPR_BENCH")

# ------------------------------------------------------------------
# Load every run
try:
    root = os.getenv("AI_SCIENTIST_ROOT", "")
    for p in experiment_data_path_list:
        try:
            ed = np.load(os.path.join(root, p), allow_pickle=True).item()
            sweep = ed[spr_key[0]][spr_key[1]]
            all_sweeps.append(sweep)
        except Exception as e:
            print(f"Error loading/reading {p}: {e}")
except Exception as e:
    print(f"Error iterating over experiment paths: {e}")

if len(all_sweeps) == 0:
    print("No experiment data could be loaded – exiting.")
    exit()

# ------------------------------------------------------------------
# Consistency check and aggregation
try:
    epochs_grid = all_sweeps[0]["epochs_grid"]
    n_runs = len(all_sweeps)
    n_settings = len(epochs_grid)

    # Containers: [run, setting, epoch]
    train_curves = []
    val_curves = []
    test_scores = []

    for sweep in all_sweeps:
        if sweep["epochs_grid"] != epochs_grid:
            raise ValueError("Mismatch in epochs_grid across runs.")
        train_curves.append(sweep["metrics"]["train"])
        val_curves.append(sweep["metrics"]["val"])
        test_scores.append(sweep["test_hsca"])

    train_curves = np.array(train_curves, dtype=object)
    val_curves = np.array(val_curves, dtype=object)
    test_scores = np.array(test_scores)  # shape [runs, settings]

    # Convert per-setting variable-length lists to fixed arrays (trim to min length)
    min_len = min([min([len(c) for c in run_curves]) for run_curves in train_curves])
    train_mean = []
    train_sem = []
    val_mean = []
    val_sem = []
    for s in range(n_settings):
        tc = np.stack([run[s][:min_len] for run in train_curves])
        vc = np.stack([run[s][:min_len] for run in val_curves])
        train_mean.append(tc.mean(axis=0))
        train_sem.append(tc.std(axis=0, ddof=1) / np.sqrt(n_runs))
        val_mean.append(vc.mean(axis=0))
        val_sem.append(vc.std(axis=0, ddof=1) / np.sqrt(n_runs))

    train_mean = np.array(train_mean)
    train_sem = np.array(train_sem)
    val_mean = np.array(val_mean)
    val_sem = np.array(val_sem)

    test_mean = test_scores.mean(axis=0)  # shape [settings]
    test_sem = test_scores.std(axis=0, ddof=1) / np.sqrt(n_runs)
except Exception as e:
    print(f"Error during aggregation: {e}")

# ------------------------------------------------------------------
# Plot 1: aggregated train / val HSCA curves with SEM bands
try:
    plt.figure()
    colors = plt.cm.tab10(np.linspace(0, 1, n_settings))
    epoch_axis = np.arange(1, min_len + 1)
    for i, max_ep in enumerate(epochs_grid):
        # Train curve
        plt.plot(epoch_axis, train_mean[i], color=colors[i], label=f"{max_ep}ep train")
        plt.fill_between(
            epoch_axis,
            train_mean[i] - train_sem[i],
            train_mean[i] + train_sem[i],
            color=colors[i],
            alpha=0.2,
        )
        # Val curve
        plt.plot(
            epoch_axis,
            val_mean[i],
            linestyle="--",
            color=colors[i],
            label=f"{max_ep}ep val",
        )
        plt.fill_between(
            epoch_axis,
            val_mean[i] - val_sem[i],
            val_mean[i] + val_sem[i],
            color=colors[i],
            alpha=0.2,
            linestyle="--",
        )

    plt.title(
        "SPR_BENCH – Aggregated HSCA Curves\nSolid: Train, Dashed: Validation (Mean ± SEM)"
    )
    plt.xlabel("Epoch")
    plt.ylabel("HSCA")
    plt.legend(ncol=2, fontsize=8)
    fname = os.path.join(working_dir, "SPR_BENCH_hsca_train_val_aggregated.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating aggregated HSCA curve plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# Plot 2: test HSCA vs max epochs with SEM error bars
try:
    plt.figure()
    x_pos = np.arange(n_settings)
    plt.bar(
        x_pos,
        test_mean,
        yerr=test_sem,
        color="steelblue",
        capsize=5,
        alpha=0.9,
        ecolor="black",
    )
    plt.title("SPR_BENCH – Test HSCA vs Allowed Fine-tuning Epochs (Mean ± SEM)")
    plt.xlabel("Max Fine-tuning Epochs")
    plt.ylabel("Test HSCA")
    plt.xticks(x_pos, [str(e) for e in epochs_grid])
    for x, y in zip(x_pos, test_mean):
        plt.text(x, y + 0.01, f"{y:.3f}", ha="center", va="bottom", fontsize=8)
    fname = os.path.join(working_dir, "SPR_BENCH_test_hsca_bar_aggregated.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating aggregated test HSCA bar plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# Print aggregated evaluation metrics
print("Aggregated Test HSCA (mean ± SEM):")
for max_ep, m, s in zip(epochs_grid, test_mean, test_sem):
    print(f"Max epochs={max_ep:2d} | Test HSCA={m:.4f} ± {s:.4f}")

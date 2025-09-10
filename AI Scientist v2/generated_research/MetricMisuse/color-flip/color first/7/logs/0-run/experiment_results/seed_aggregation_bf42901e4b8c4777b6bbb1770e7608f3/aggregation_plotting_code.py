import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------------
# Initial setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
# List of experiment_data paths (relative to AI_SCIENTIST_ROOT env var)
experiment_data_path_list = [
    "experiments/2025-08-31_14-11-51_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_7d353d62cbb649a0bcb941e6164dff22_proc_1730838/experiment_data.npy",
    "experiments/2025-08-31_14-11-51_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_b9824fcac00641be8326a28518756696_proc_1730839/experiment_data.npy",
    "experiments/2025-08-31_14-11-51_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_164c9f42a8ae4e2f8f602fece4d44d02_proc_1730837/experiment_data.npy",
]

# ---------------------------------------------------------------------
# Load all experiment data
all_runs = []
for exp_path in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), exp_path)
        exp_data = np.load(full_path, allow_pickle=True).item()
        all_runs.append(exp_data)
    except Exception as e:
        print(f"Error loading experiment data from {exp_path}: {e}")

if len(all_runs) == 0:
    print("No experiment files could be loaded; aborting.")
    raise SystemExit

# ---------------------------------------------------------------------
# Extract metrics for SPR_BENCH from each run
epochs_list, train_loss_list = [], []
train_cpx_list, val_cpx_list = [], []
val_cwa_list, val_swa_list = [], []
best_val_cpx_each = []

for idx, exp in enumerate(all_runs):
    try:
        spr = exp.get("epochs_tuning", {}).get("SPR_BENCH", None)
        if spr is None:
            print(f"Run {idx}: SPR_BENCH not found; skipping.")
            continue

        epochs = np.array(spr["epochs"])
        train_loss = np.array(spr["losses"]["train"])

        train_metrics = spr["metrics"]["train"]
        val_metrics = spr["metrics"]["val"]

        train_cpx = np.array([m["cpx"] for m in train_metrics])
        val_cpx = np.array([m["cpx"] for m in val_metrics])
        val_cwa = np.array([m["cwa"] for m in val_metrics])
        val_swa = np.array([m["swa"] for m in val_metrics])

        # Align lengths just in case
        min_len = min(
            len(epochs),
            len(train_loss),
            len(train_cpx),
            len(val_cpx),
            len(val_cwa),
            len(val_swa),
        )
        epochs_list.append(epochs[:min_len])
        train_loss_list.append(train_loss[:min_len])
        train_cpx_list.append(train_cpx[:min_len])
        val_cpx_list.append(val_cpx[:min_len])
        val_cwa_list.append(val_cwa[:min_len])
        val_swa_list.append(val_swa[:min_len])

        # Record best val CpxWA for this run
        best_val_cpx_each.append(float(val_cpx.max()))
        print(
            f"Run {idx}: Best Val CpxWA = {val_cpx.max():.4f} "
            f"@ epoch {epochs[np.argmax(val_cpx)]}"
        )
    except Exception as e:
        print(f"Error processing run {idx}: {e}")

# Sanity-check we have at least one successful run
if len(train_loss_list) == 0:
    print("No valid SPR_BENCH data found across runs; aborting.")
    raise SystemExit


# ---------------------------------------------------------------------
def stack_and_stats(lst):
    arr = np.stack(lst, axis=0)  # shape (n_runs, n_epochs)
    mean = np.nanmean(arr, axis=0)
    sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
    return mean, sem


mean_train_loss, sem_train_loss = stack_and_stats(train_loss_list)
mean_train_cpx, sem_train_cpx = stack_and_stats(train_cpx_list)
mean_val_cpx, sem_val_cpx = stack_and_stats(val_cpx_list)
mean_val_cwa, sem_val_cwa = stack_and_stats(val_cwa_list)
mean_val_swa, sem_val_swa = stack_and_stats(val_swa_list)
epochs = epochs_list[0]  # assume aligned as enforced above

# Overall aggregated best Val CpxWA (per-run bests)
overall_mean_best = np.mean(best_val_cpx_each)
overall_sem_best = np.std(best_val_cpx_each, ddof=1) / np.sqrt(len(best_val_cpx_each))
print(
    f"Aggregated Best Val CpxWA across runs: {overall_mean_best:.4f} ± {overall_sem_best:.4f}"
)

# ---------------------------------------------------------------------
# 1) Mean training loss curve
try:
    plt.figure()
    plt.plot(epochs, mean_train_loss, label="Mean Train Loss")
    plt.fill_between(
        epochs,
        mean_train_loss - sem_train_loss,
        mean_train_loss + sem_train_loss,
        alpha=0.3,
        label="± SEM",
    )
    plt.title("SPR_BENCH: Mean Training Loss per Epoch\nShaded: ±SEM")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_mean_train_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating mean train loss plot: {e}")
    plt.close()

# ---------------------------------------------------------------------
# 2) Mean Train vs. Val CpxWA
try:
    plt.figure()
    plt.plot(epochs, mean_train_cpx, label="Train CpxWA (mean)")
    plt.fill_between(
        epochs,
        mean_train_cpx - sem_train_cpx,
        mean_train_cpx + sem_train_cpx,
        alpha=0.3,
        label="Train ± SEM",
    )
    plt.plot(epochs, mean_val_cpx, label="Val CpxWA (mean)")
    plt.fill_between(
        epochs,
        mean_val_cpx - sem_val_cpx,
        mean_val_cpx + sem_val_cpx,
        alpha=0.3,
        label="Val ± SEM",
    )
    plt.title("SPR_BENCH: Complexity-Weighted Accuracy\nLeft: Train, Right: Validation")
    plt.xlabel("Epoch")
    plt.ylabel("CpxWA")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_mean_cpxwa_train_val_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating mean CpxWA plot: {e}")
    plt.close()

# ---------------------------------------------------------------------
# 3) Mean Validation weighted accuracy comparison
try:
    plt.figure()
    # Use error bars here for clarity
    plt.errorbar(
        epochs, mean_val_cwa, yerr=sem_val_cwa, marker="o", label="Val CWA (mean ± SEM)"
    )
    plt.errorbar(
        epochs, mean_val_swa, yerr=sem_val_swa, marker="^", label="Val SWA (mean ± SEM)"
    )
    plt.errorbar(
        epochs,
        mean_val_cpx,
        yerr=sem_val_cpx,
        marker="s",
        label="Val CpxWA (mean ± SEM)",
    )
    plt.title("SPR_BENCH: Weighted Accuracy Comparison (Validation)")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted Accuracy")
    plt.legend()
    fname = os.path.join(
        working_dir, "SPR_BENCH_mean_val_weighted_accuracy_comparison.png"
    )
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating mean weighted accuracy comparison plot: {e}")
    plt.close()

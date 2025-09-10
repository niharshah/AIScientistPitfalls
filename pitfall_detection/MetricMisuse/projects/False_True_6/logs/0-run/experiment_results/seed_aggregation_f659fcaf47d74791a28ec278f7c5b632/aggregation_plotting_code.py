import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- set up output dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- candidate paths ----------
experiment_data_path_list = [
    "None/experiment_data.npy",
    "None/experiment_data.npy",
    "experiments/2025-08-14_17-37-20_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_094d5cbff8bb4eaeaa7c21d973b5acf2_proc_2705536/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    # if an environment root is provided, try that first
    full_path = (
        os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        if not os.path.isabs(p)
        else p
    )
    try:
        exp_dict = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_dict)
        print(f"Loaded experiment data from {full_path}")
    except Exception as e:
        print(f"Skipping {p}: {e}")

if not all_experiment_data:
    print("No experiment data could be loaded, exiting.")
    exit()

# ------------- aggregate -------------
model_name = "NoSymFeatTransformer"
dataset_name = "SPR_BENCH"

train_losses, val_losses, val_swa, test_loss, test_swa = [], [], [], [], []

for exp in all_experiment_data:
    try:
        rec = exp[model_name][dataset_name]
        train_losses.append(np.asarray(rec["losses"]["train"]))
        val_losses.append(np.asarray(rec["losses"]["val"]))
        val_swa.append(np.asarray(rec["SWA"]["val"]))
        test_loss.append(rec["metrics"]["test"].get("loss", np.nan))
        test_swa.append(rec["metrics"]["test"].get("SWA", np.nan))
    except Exception as e:
        print(f"Missing keys in one run, skipping it: {e}")

n_runs = len(train_losses)
if n_runs == 0:
    print("No valid runs for requested model/dataset.")
    exit()

# make all arrays equal length (truncate to shortest run)
min_len = min(map(len, train_losses))
train_losses = np.stack([tl[:min_len] for tl in train_losses])
val_losses = np.stack([vl[:min_len] for vl in val_losses])
val_swa = np.stack([vs[:min_len] for vs in val_swa])
epochs = np.arange(1, min_len + 1)


def compute_mean_sem(arr):
    mean = arr.mean(axis=0)
    sem = (
        arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        if arr.shape[0] > 1
        else np.zeros_like(mean)
    )
    return mean, sem


train_mean, train_sem = compute_mean_sem(train_losses)
val_mean, val_sem = compute_mean_sem(val_losses)
swa_mean, swa_sem = compute_mean_sem(val_swa)

# --------- PLOT 1: aggregated loss curves -----------
try:
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_mean, label="Train Loss (mean)")
    plt.fill_between(
        epochs,
        train_mean - train_sem,
        train_mean + train_sem,
        alpha=0.3,
        label="Train SEM",
    )
    plt.plot(epochs, val_mean, label="Val Loss (mean)")
    plt.fill_between(
        epochs, val_mean - val_sem, val_mean + val_sem, alpha=0.3, label="Val SEM"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(
        f"{dataset_name}: Training vs Validation Loss\n(Aggregated over {n_runs} run{'s' if n_runs>1 else ''})"
    )
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, f"{dataset_name}_loss_curves_agg.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

# --------- PLOT 2: aggregated SWA -----------
try:
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, swa_mean, marker="o", label="Val SWA (mean)")
    plt.fill_between(
        epochs, swa_mean - swa_sem, swa_mean + swa_sem, alpha=0.3, label="SWA SEM"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Shape-Weighted Accuracy")
    plt.title(
        f"{dataset_name}: Validation SWA per Epoch\n(Aggregated over {n_runs} run{'s' if n_runs>1 else ''})"
    )
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, f"{dataset_name}_swa_curve_agg.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated SWA plot: {e}")
    plt.close()

# ------------- print final aggregated metrics -------------
test_loss = np.asarray(test_loss, dtype=float)
test_swa = np.asarray(test_swa, dtype=float)


def agg_str(arr):
    if len(arr) > 1:
        return f"{np.nanmean(arr):.4f} ± {np.nanstd(arr, ddof=1)/np.sqrt(len(arr)):.4f}"
    else:
        return f"{arr[0]:.4f}"


print(
    f"Aggregated Test Metrics ({n_runs} run{'s' if n_runs>1 else ''}) -> "
    f"Loss: {agg_str(test_loss)}, SWA: {agg_str(test_swa)}"
)

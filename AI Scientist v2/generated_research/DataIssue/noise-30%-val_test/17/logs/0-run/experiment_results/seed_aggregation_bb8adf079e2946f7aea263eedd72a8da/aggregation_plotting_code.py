import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import matthews_corrcoef

# ----------------- house-keeping -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Paths relative to AI_SCIENTIST_ROOT (as provided by the user)
experiment_data_path_list = [
    "experiments/2025-08-17_18-47-59_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_5caae949159d44b8a2b2b12dd15fab41_proc_3327676/experiment_data.npy",
    "experiments/2025-08-17_18-47-59_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_1dd1fe136e394328a2261e3b14e4ef9e_proc_3327678/experiment_data.npy",
    "experiments/2025-08-17_18-47-59_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_657956c1a7b44466b3b31d43fd2de0b4_proc_3327677/experiment_data.npy",
]

all_runs = []
for path in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), path)
        data = np.load(full_path, allow_pickle=True).item()
        all_runs.append(data)
    except Exception as e:
        print(f"Error loading {path}: {e}")

# ----------------- collect metrics -----------------
train_losses, val_losses = [], []
train_mccs, val_mccs = [], []
tp_total = fp_total = fn_total = tn_total = 0

for run in all_runs:
    bench = run.get("SPR_BENCH", {})
    tl = np.asarray(bench.get("losses", {}).get("train", []), dtype=float)
    vl = np.asarray(bench.get("losses", {}).get("val", []), dtype=float)
    tm = np.asarray(bench.get("metrics", {}).get("train", []), dtype=float)
    vm = np.asarray(bench.get("metrics", {}).get("val", []), dtype=float)
    train_losses.append(tl)
    val_losses.append(vl)
    train_mccs.append(tm)
    val_mccs.append(vm)

    preds = np.asarray(bench.get("predictions", []))
    gts = np.asarray(bench.get("ground_truth", []))
    tp_total += np.sum((preds == 1) & (gts == 1))
    fp_total += np.sum((preds == 1) & (gts == 0))
    fn_total += np.sum((preds == 0) & (gts == 1))
    tn_total += np.sum((preds == 0) & (gts == 0))


def pad_to_equal_length(arrays):
    max_len = max(len(a) for a in arrays) if arrays else 0
    out = []
    for a in arrays:
        if len(a) < max_len:
            pad = np.full(max_len - len(a), np.nan)
            out.append(np.concatenate([a, pad]))
        else:
            out.append(a)
    return np.vstack(out) if out else np.empty((0, 0))


train_losses_mat = pad_to_equal_length(train_losses)
val_losses_mat = pad_to_equal_length(val_losses)
train_mccs_mat = pad_to_equal_length(train_mccs)
val_mccs_mat = pad_to_equal_length(val_mccs)


def mean_sem(mat):
    mean = np.nanmean(mat, axis=0)
    sem = np.nanstd(mat, axis=0) / np.sqrt(np.sum(~np.isnan(mat), axis=0))
    return mean, sem


epochs = (
    np.arange(1, train_losses_mat.shape[1] + 1)
    if train_losses_mat.size
    else np.array([])
)

# ----------------- Figure 1: Loss curve (mean ± SEM) -----------------
try:
    plt.figure()
    if epochs.size:
        tl_mean, tl_sem = mean_sem(train_losses_mat)
        vl_mean, vl_sem = mean_sem(val_losses_mat)
        plt.plot(epochs, tl_mean, label="Train Loss (mean)")
        plt.fill_between(
            epochs, tl_mean - tl_sem, tl_mean + tl_sem, alpha=0.3, label="Train SEM"
        )
        plt.plot(epochs, vl_mean, label="Val Loss (mean)")
        plt.fill_between(
            epochs, vl_mean - vl_sem, vl_mean + vl_sem, alpha=0.3, label="Val SEM"
        )
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title(
        "SPR_BENCH Aggregated Training vs Validation Loss\n(shaded = SEM across runs)"
    )
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_aggregated_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss curve: {e}")
    plt.close()

# ----------------- Figure 2: MCC curve (mean ± SEM) -----------------
try:
    plt.figure()
    if epochs.size:
        tm_mean, tm_sem = mean_sem(train_mccs_mat)
        vm_mean, vm_sem = mean_sem(val_mccs_mat)
        plt.plot(epochs, tm_mean, label="Train MCC (mean)")
        plt.fill_between(
            epochs, tm_mean - tm_sem, tm_mean + tm_sem, alpha=0.3, label="Train SEM"
        )
        plt.plot(epochs, vm_mean, label="Val MCC (mean)")
        plt.fill_between(
            epochs, vm_mean - vm_sem, vm_mean + vm_sem, alpha=0.3, label="Val SEM"
        )
    plt.xlabel("Epoch")
    plt.ylabel("MCC")
    plt.title(
        "SPR_BENCH Aggregated Training vs Validation MCC\n(shaded = SEM across runs)"
    )
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_aggregated_mcc_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated MCC curve: {e}")
    plt.close()

# ----------------- Figure 3: Aggregated Confusion Bar -----------------
try:
    plt.figure()
    bars = [tp_total, fp_total, fn_total, tn_total]
    labels = ["TP", "FP", "FN", "TN"]
    plt.bar(labels, bars, color=["g", "r", "r", "g"])
    plt.ylabel("Count")
    plt.title("SPR_BENCH Aggregated Test Confusion Matrix (bar)")
    fname = os.path.join(working_dir, "SPR_BENCH_aggregated_confusion_matrix.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated confusion plot: {e}")
    plt.close()

# ----------------- Print headline metric -----------------
if val_mccs_mat.size:
    final_epoch_val_mcc = val_mccs_mat[:, -1]
    mean_final = np.nanmean(final_epoch_val_mcc)
    sem_final = np.nanstd(final_epoch_val_mcc) / np.sqrt(
        np.sum(~np.isnan(final_epoch_val_mcc))
    )
    print(
        f"Aggregated Final-Epoch Validation MCC: {mean_final:.4f} ± {sem_final:.4f} (mean ± SEM)"
    )

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -------------------------------------------------------------------------
# set up paths
# -------------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# list of experiment result files provided by the system
experiment_data_path_list = [
    "experiments/2025-08-17_18-48-06_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_9891c59e3e0f4298a9167b40758dd094_proc_3327788/experiment_data.npy",
    "experiments/2025-08-17_18-48-06_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_96cb58584cf847579162c4b367b655df_proc_3327790/experiment_data.npy",
    "experiments/2025-08-17_18-48-06_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_d7084f96a1584b27a181610122410395_proc_3327787/experiment_data.npy",
]

# -------------------------------------------------------------------------
# load experiment data
# -------------------------------------------------------------------------
all_experiment_data = []
for path in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), path)
        exp_data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_data)
    except Exception as e:
        print(f"Error loading experiment data from {path}: {e}")


# -------------------------------------------------------------------------
# aggregation helpers
# -------------------------------------------------------------------------
def stack_and_truncate(arrays):
    """Stack 1-D arrays after truncating them to the minimum length."""
    if not arrays:
        return np.array([])
    min_len = min(len(a) for a in arrays if len(a))
    if min_len == 0:
        return np.array([])
    return np.vstack([a[:min_len] for a in arrays]), min_len


dataset_name = "SPR_BENCH"
train_losses, val_losses = [], []
train_f1s, val_f1s = [], []
all_epochs = []
all_best_val_f1 = []
all_preds, all_gts = [], []

# gather arrays
for idx, ed in enumerate(all_experiment_data):
    data = ed.get(dataset_name, {})
    epochs = np.array(data.get("epochs", []))
    train_ls = np.array(data.get("losses", {}).get("train", []))
    val_ls = np.array(data.get("losses", {}).get("val", []))
    train_f1 = np.array(data.get("metrics", {}).get("train", []))
    val_f1 = np.array(data.get("metrics", {}).get("val", []))
    preds = np.array(data.get("predictions", []))
    gts = np.array(data.get("ground_truth", []))

    # keep only non-empty runs
    if len(epochs):
        train_losses.append(train_ls)
        val_losses.append(val_ls)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        all_epochs.append(epochs)
        if len(val_f1):
            best_f1 = float(np.max(val_f1))
            all_best_val_f1.append(best_f1)
            print(f"Run {idx:02d} best Dev Macro-F1 = {best_f1:.4f}")
    if len(preds) and len(gts):
        all_preds.extend(preds.tolist())
        all_gts.extend(gts.tolist())

# -------------------------------------------------------------------------
# compute means and standard errors
# -------------------------------------------------------------------------
train_mat, min_len = stack_and_truncate(train_losses)
val_mat, _ = stack_and_truncate(val_losses)
train_f1_mat, _ = stack_and_truncate(train_f1s)
val_f1_mat, _ = stack_and_truncate(val_f1s)
epoch_axis = np.arange(min_len) if min_len else np.array([])


def mean_and_se(mat):
    if mat.size == 0:
        return np.array([]), np.array([])
    mean = mat.mean(axis=0)
    se = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
    return mean, se


train_loss_mean, train_loss_se = mean_and_se(train_mat)
val_loss_mean, val_loss_se = mean_and_se(val_mat)
train_f1_mean, train_f1_se = mean_and_se(train_f1_mat)
val_f1_mean, val_f1_se = mean_and_se(val_f1_mat)

if all_best_val_f1:
    print(
        f"Aggregated best Dev Macro-F1 (mean ± SE) = "
        f"{np.mean(all_best_val_f1):.4f} ± "
        f"{np.std(all_best_val_f1, ddof=1)/np.sqrt(len(all_best_val_f1)):.4f}"
    )

# -------------------------------------------------------------------------
# 1) aggregated loss curve
# -------------------------------------------------------------------------
try:
    if epoch_axis.size and train_loss_mean.size and val_loss_mean.size:
        plt.figure()
        plt.plot(epoch_axis, train_loss_mean, label="Train Loss (mean)", color="C0")
        plt.fill_between(
            epoch_axis,
            train_loss_mean - train_loss_se,
            train_loss_mean + train_loss_se,
            color="C0",
            alpha=0.3,
            label="Train Loss (±SE)",
        )
        plt.plot(epoch_axis, val_loss_mean, label="Val Loss (mean)", color="C1")
        plt.fill_between(
            epoch_axis,
            val_loss_mean - val_loss_se,
            val_loss_mean + val_loss_se,
            color="C1",
            alpha=0.3,
            label="Val Loss (±SE)",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Aggregated Train/Val Loss\n(Mean ± Standard Error)")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_agg_loss_curve.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss curve: {e}")
    plt.close()

# -------------------------------------------------------------------------
# 2) aggregated macro-F1 curve
# -------------------------------------------------------------------------
try:
    if epoch_axis.size and train_f1_mean.size and val_f1_mean.size:
        plt.figure()
        plt.plot(epoch_axis, train_f1_mean, label="Train Macro-F1 (mean)", color="C0")
        plt.fill_between(
            epoch_axis,
            train_f1_mean - train_f1_se,
            train_f1_mean + train_f1_se,
            color="C0",
            alpha=0.3,
            label="Train Macro-F1 (±SE)",
        )
        plt.plot(epoch_axis, val_f1_mean, label="Val Macro-F1 (mean)", color="C1")
        plt.fill_between(
            epoch_axis,
            val_f1_mean - val_f1_se,
            val_f1_mean + val_f1_se,
            color="C1",
            alpha=0.3,
            label="Val Macro-F1 (±SE)",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH: Aggregated Train/Val Macro-F1\n(Mean ± Standard Error)")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_agg_f1_curve.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated F1 curve: {e}")
    plt.close()

# -------------------------------------------------------------------------
# 3) aggregated confusion matrix (all runs combined)
# -------------------------------------------------------------------------
try:
    if len(all_preds) and len(all_gts):
        cm = confusion_matrix(all_gts, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        plt.figure()
        disp.plot(cmap="Blues", values_format="d")
        plt.title("SPR_BENCH: Confusion Matrix\n(All Runs Combined)")
        fname = os.path.join(working_dir, "SPR_BENCH_agg_confusion_matrix.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated confusion matrix: {e}")
    plt.close()

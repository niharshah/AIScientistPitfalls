import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------- setup ---------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------
# 1) Load all experiment_data.npy files
experiment_data_path_list = [
    "experiments/2025-08-16_01-26-03_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_5938e7ae065e4db6a4eaf452d550a72d_proc_3081932/experiment_data.npy",
    "experiments/2025-08-16_01-26-03_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_6a3035e96fa2453e9eec1ba3dbf2d650_proc_3081931/experiment_data.npy",
    "experiments/2025-08-16_01-26-03_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_0d6948a1925647a2a976b8b4e8b11665_proc_3081930/experiment_data.npy",
]

all_spr_runs = []
for path in experiment_data_path_list:
    try:
        p = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), path)
        data = np.load(p, allow_pickle=True).item()
        if "SPR" in data:
            all_spr_runs.append(data["SPR"])
    except Exception as e:
        print(f"Error loading {path}: {e}")

n_runs = len(all_spr_runs)
if n_runs == 0:
    print("No runs loaded; exiting.")
    quit()


# Helper to stack metric across runs safely
def stack_key(key_path):
    """Return array with shape (n_runs, epochs) for the requested nested key path.

    key_path: tuple of nested keys, e.g. ('losses', 'train')
    """
    arrs = []
    for spr in all_spr_runs:
        cur = spr
        for k in key_path:
            if k not in cur:
                break
            cur = cur[k]
        else:  # only executed if loop did not break
            arrs.append(np.asarray(cur))
            continue
        # if we reach here, missing key
        return None
    min_len = min(a.shape[-1] for a in arrs)
    arrs = [a[:min_len] for a in arrs]
    return np.vstack(arrs)  # shape (runs, epochs)


plots_made = []

# -------------------------------------------------
# 2) Aggregate and plot LOSS curves
try:
    train_losses = stack_key(("losses", "train"))
    val_losses = stack_key(("losses", "val"))
    if train_losses is not None and val_losses is not None:
        epochs = np.arange(1, train_losses.shape[1] + 1)
        train_mean, train_se = train_losses.mean(0), train_losses.std(
            0, ddof=1
        ) / np.sqrt(n_runs)
        val_mean, val_se = val_losses.mean(0), val_losses.std(0, ddof=1) / np.sqrt(
            n_runs
        )

        plt.figure()
        plt.plot(epochs, train_mean, label="Train Loss (mean)")
        plt.fill_between(
            epochs,
            train_mean - train_se,
            train_mean + train_se,
            alpha=0.3,
            label="Train ±1 SE",
        )
        plt.plot(epochs, val_mean, label="Val Loss (mean)")
        plt.fill_between(
            epochs, val_mean - val_se, val_mean + val_se, alpha=0.3, label="Val ±1 SE"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR Aggregate: Training vs Validation Loss (Mean ± SE)")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_loss_curve_aggregate.png")
        plt.savefig(fname)
        plots_made.append(fname)
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
finally:
    plt.close()

# -------------------------------------------------
# 3) Aggregate and plot WEIGHTED ACCURACY curves
try:
    train_acc = stack_key(("metrics", "train"))
    val_acc = stack_key(("metrics", "val"))
    if train_acc is not None and val_acc is not None:
        epochs = np.arange(1, train_acc.shape[1] + 1)
        tr_m, tr_se = train_acc.mean(0), train_acc.std(0, ddof=1) / np.sqrt(n_runs)
        va_m, va_se = val_acc.mean(0), val_acc.std(0, ddof=1) / np.sqrt(n_runs)

        plt.figure()
        plt.plot(epochs, tr_m, label="SWA Train (mean)")
        plt.fill_between(
            epochs, tr_m - tr_se, tr_m + tr_se, alpha=0.3, label="Train ±1 SE"
        )
        plt.plot(epochs, va_m, label="CWA Val (mean)")
        plt.fill_between(
            epochs, va_m - va_se, va_m + va_se, alpha=0.3, label="Val ±1 SE"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.title("SPR Aggregate: Left: SWA (Train), Right: CWA (Val) (Mean ± SE)")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_accuracy_curve_aggregate.png")
        plt.savefig(fname)
        plots_made.append(fname)
except Exception as e:
    print(f"Error creating aggregated accuracy plot: {e}")
finally:
    plt.close()

# -------------------------------------------------
# 4) Aggregate and plot AIS curves
try:
    ais_val = stack_key(("AIS", "val"))
    if ais_val is not None:
        epochs = np.arange(1, ais_val.shape[1] + 1)
        ais_m = ais_val.mean(0)
        ais_se = ais_val.std(0, ddof=1) / np.sqrt(n_runs)

        plt.figure()
        plt.plot(epochs, ais_m, marker="o", label="AIS Val (mean)")
        plt.fill_between(
            epochs, ais_m - ais_se, ais_m + ais_se, alpha=0.3, label="±1 SE"
        )
        plt.xlabel("Epoch")
        plt.ylabel("AIS")
        plt.title(
            "SPR Aggregate: Augmentation Invariance Score (Validation) (Mean ± SE)"
        )
        plt.legend()
        fname = os.path.join(working_dir, "SPR_AIS_curve_aggregate.png")
        plt.savefig(fname)
        plots_made.append(fname)
except Exception as e:
    print(f"Error creating aggregated AIS plot: {e}")
finally:
    plt.close()

# -------------------------------------------------
# 5) Aggregate Confusion Matrix
try:
    # Build summed confusion matrix
    first_preds = all_spr_runs[0].get("predictions", None)
    first_gts = all_spr_runs[0].get("ground_truth", None)
    if first_preds is not None and first_gts is not None:
        num_cls = (
            int(
                max(first_preds).max()
                if hasattr(first_preds, "max")
                else max(first_preds)
            )
            + 1
        )
        cm_total = np.zeros((num_cls, num_cls), dtype=int)

        for spr in all_spr_runs:
            preds = np.asarray(spr["predictions"])
            gts = np.asarray(spr["ground_truth"])
            for t, p in zip(gts, preds):
                cm_total[t, p] += 1

        plt.figure()
        im = plt.imshow(cm_total, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR Aggregate: Confusion Matrix (Validation, summed over runs)")
        # add counts
        for i in range(num_cls):
            for j in range(num_cls):
                plt.text(
                    j,
                    i,
                    cm_total[i, j],
                    ha="center",
                    va="center",
                    color="red",
                    fontsize=8,
                )
        fname = os.path.join(working_dir, "SPR_confusion_matrix_aggregate.png")
        plt.savefig(fname)
        plots_made.append(fname)
except Exception as e:
    print(f"Error creating aggregated confusion matrix: {e}")
finally:
    plt.close()

print("Plots saved:", plots_made)

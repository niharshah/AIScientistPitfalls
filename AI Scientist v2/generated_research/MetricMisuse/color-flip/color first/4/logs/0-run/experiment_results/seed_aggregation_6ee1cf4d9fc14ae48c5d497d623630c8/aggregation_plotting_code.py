import matplotlib.pyplot as plt
import numpy as np
import os
import math

# ---------------------------------------------------------------------------
# basic setup
# ---------------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# load all experiment_data dicts
# ---------------------------------------------------------------------------
try:
    experiment_data_path_list = [
        "experiments/2025-08-31_03-13-24_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_cd42937014c74b0a9f71351344cec1c4_proc_1635407/experiment_data.npy",
        "experiments/2025-08-31_03-13-24_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_d3501db91fa84b8c80f443eea1808db4_proc_1635406/experiment_data.npy",
        "experiments/2025-08-31_03-13-24_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_e8ca8f7f5f2a482f8e6bb84e65b7957d_proc_1635404/experiment_data.npy",
    ]
    all_experiment_data = []
    for experiment_data_path in experiment_data_path_list:
        full_path = os.path.join(
            os.getenv("AI_SCIENTIST_ROOT", ""), experiment_data_path
        )
        exp_dict = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_dict)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# ---------------------------------------------------------------------------
# gather and aggregate metrics across runs
# ---------------------------------------------------------------------------
metrics_by_lr = {}  # {lr_tag: {"loss_train":[], ...}}
if all_experiment_data:
    # discover all learning-rate tags
    lr_tags = sorted(
        {k for d in all_experiment_data for k in d if k.startswith("lr_")},
        key=lambda x: float(x.split("_")[1]),
    )
    for lr in lr_tags:
        metrics_by_lr[lr] = {
            "loss_train": [],
            "loss_val": [],
            "acc": [],
            "cwa": [],
            "swa": [],
            "pcwa": [],
        }
    # fill
    for exp_dict in all_experiment_data:
        for lr in lr_tags:
            if lr not in exp_dict:
                continue  # skip if this run lacks that lr
            rec = exp_dict[lr]["SPR_BENCH"]
            metrics_by_lr[lr]["loss_train"].append(np.asarray(rec["losses"]["train"]))
            metrics_by_lr[lr]["loss_val"].append(np.asarray(rec["losses"]["val"]))
            val_dicts = rec["metrics"]["val"]
            metrics_by_lr[lr]["acc"].append(np.asarray([d["acc"] for d in val_dicts]))
            metrics_by_lr[lr]["cwa"].append(np.asarray([d["cwa"] for d in val_dicts]))
            metrics_by_lr[lr]["swa"].append(np.asarray([d["swa"] for d in val_dicts]))
            metrics_by_lr[lr]["pcwa"].append(np.asarray([d["pcwa"] for d in val_dicts]))


# helper to compute mean and stderr
def mean_se(arrays):
    """arrays: list of 1-D numpy arrays with equal length"""
    stack = np.stack(arrays)  # shape (runs, epochs)
    mean = stack.mean(axis=0)
    se = (
        stack.std(axis=0, ddof=1) / math.sqrt(stack.shape[0])
        if stack.shape[0] > 1
        else np.zeros_like(mean)
    )
    return mean, se


# ---------------------------------------------------------------------------
# plotting utility
# ---------------------------------------------------------------------------
def plot_aggregate(metric_name, ylabel, fname):
    try:
        plt.figure()
        for lr in metrics_by_lr:
            if not metrics_by_lr[lr][metric_name]:
                continue
            mean, se = mean_se(metrics_by_lr[lr][metric_name])
            epochs = np.arange(1, len(mean) + 1)
            label = f"lr={lr.split('_')[1]} (mean)"
            plt.plot(epochs, mean, label=label)
            plt.fill_between(
                epochs, mean - se, mean + se, alpha=0.2, label=f"{label} ±1SE"
            )
        plt.title(
            f"SPR_BENCH {ylabel} over Epochs\n(Mean ± Standard Error across runs)"
        )
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend(frameon=False)
        save_path = os.path.join(working_dir, f"SPR_BENCH_{fname}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating plot {fname}: {e}")
        plt.close()


# ---------------------------------------------------------------------------
# create plots
# ---------------------------------------------------------------------------
if metrics_by_lr:
    plot_aggregate("loss_train", "Training Loss", "train_loss_mean_se")
    plot_aggregate("loss_val", "Validation Loss", "val_loss_mean_se")
    plot_aggregate("acc", "Validation Accuracy", "val_acc_mean_se")
    plot_aggregate("cwa", "Color-Weighted Accuracy", "cwa_mean_se")
    plot_aggregate("swa", "Shape-Weighted Accuracy", "swa_mean_se")
    # plot only first five metrics types, as requested (already 5)

# ---------------------------------------------------------------------------
# print final epoch summary
# ---------------------------------------------------------------------------
for lr in metrics_by_lr:
    for metric in ["acc", "cwa", "swa", "pcwa"]:
        if not metrics_by_lr[lr][metric]:
            continue
        mean_last, se_last = mean_se(metrics_by_lr[lr][metric])
        print(
            f"{lr} | {metric.upper()} final epoch: {mean_last[-1]:.3f} ± {se_last[-1]:.3f}"
        )

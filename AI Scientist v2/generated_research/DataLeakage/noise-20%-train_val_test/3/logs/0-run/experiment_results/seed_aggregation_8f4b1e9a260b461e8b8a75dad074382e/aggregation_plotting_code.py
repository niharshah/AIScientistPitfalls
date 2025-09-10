import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, f1_score

# ------------------ paths & data ------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# PLEASE KEEP THESE EXACT PATHS – THEY COME FROM THE "Experiment Data Path" SECTION
experiment_data_path_list = [
    "experiments/2025-08-17_00-44-36_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_467a8c7584a04880932cf0d4b41b1b62_proc_3158132/experiment_data.npy",
    "experiments/2025-08-17_00-44-36_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_3bd512f8ecfd417fb5f38a0eed49b2f6_proc_3158135/experiment_data.npy",
    "experiments/2025-08-17_00-44-36_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_a961c837e1b84328bbd57426bbcd0ed5_proc_3158133/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        ed = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_experiment_data.append(ed)
    except Exception as e:
        print(f"Error loading {p}: {e}")


# ------------------ helper to aggregate ------------------
def agg_by_epoch(metric_key, runs_data):
    """Return sorted epochs, mean values and stderr for the given key."""
    # Map epoch -> list of measurements
    collector = {}
    for rd in runs_data:
        epochs = rd.get("epochs", [])
        values = rd.get("metrics", {}).get(metric_key, [])
        for e, v in zip(epochs, values):
            if v is None:
                continue
            collector.setdefault(e, []).append(v)
    if not collector:
        return [], [], []
    sorted_epochs = sorted(collector.keys())
    mean_vals, stderr_vals = [], []
    for e in sorted_epochs:
        arr = np.array(collector[e], dtype=float)
        mean_vals.append(arr.mean())
        stderr_vals.append(arr.std(ddof=1) / np.sqrt(len(arr)))
    return sorted_epochs, mean_vals, stderr_vals


# ------------------ iterate over datasets ------------------
datasets = set()
for ed in all_experiment_data:
    datasets.update(ed.get("epochs_tuning", {}).keys())

for ds in datasets:
    # collect run-specific dicts for this dataset
    runs_data = []
    concatenated_preds, concatenated_gts = [], []
    final_epoch_f1s = []
    for ed in all_experiment_data:
        ds_dict = ed.get("epochs_tuning", {}).get(ds, {})
        if ds_dict:
            runs_data.append(ds_dict)
            preds = np.array(ds_dict.get("predictions", []))
            gts = np.array(ds_dict.get("ground_truth", []))
            if preds.size and gts.size:
                concatenated_preds.append(preds)
                concatenated_gts.append(gts)
                final_epoch_f1s.append(f1_score(gts, preds, average="macro"))

    # ---------- plot 1: aggregated loss curves ----------
    try:
        epochs_train, mean_train, stderr_train = agg_by_epoch("train_loss", runs_data)
        epochs_val, mean_val, stderr_val = agg_by_epoch("val_loss", runs_data)
        if epochs_train and epochs_val:
            plt.figure()
            plt.plot(epochs_train, mean_train, label="Train Loss (mean)")
            plt.fill_between(
                epochs_train,
                np.array(mean_train) - np.array(stderr_train),
                np.array(mean_train) + np.array(stderr_train),
                alpha=0.3,
                label="Train Loss (stderr)",
            )
            plt.plot(epochs_val, mean_val, label="Validation Loss (mean)")
            plt.fill_between(
                epochs_val,
                np.array(mean_val) - np.array(stderr_val),
                np.array(mean_val) + np.array(stderr_val),
                alpha=0.3,
                label="Val Loss (stderr)",
            )
            plt.title(
                f"{ds} Loss Curves with Mean ± StdErr\nLeft: Train, Right: Validation"
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{ds}_agg_loss_curves.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {ds}: {e}")
        plt.close()

    # ---------- plot 2: aggregated validation F1 ----------
    try:
        epochs_f1, mean_f1, stderr_f1 = agg_by_epoch("val_f1", runs_data)
        if epochs_f1:
            plt.figure()
            plt.errorbar(
                epochs_f1,
                mean_f1,
                yerr=stderr_f1,
                fmt="-o",
                capsize=3,
                label="Validation Macro-F1 (mean ± stderr)",
            )
            plt.title(f"{ds} Validation Macro-F1 Across Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Macro-F1")
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{ds}_agg_val_f1_curve.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated F1 plot for {ds}: {e}")
        plt.close()

    # ---------- plot 3: aggregate confusion matrix ----------
    try:
        if concatenated_preds and concatenated_gts:
            all_preds = np.concatenate(concatenated_preds)
            all_gts = np.concatenate(concatenated_gts)
            cm = confusion_matrix(all_gts, all_preds)
            plt.figure()
            im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
            plt.title(
                f"{ds} Aggregate Confusion Matrix\nLeft: Ground Truth, Right: Predicted"
            )
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{ds}_aggregate_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated confusion matrix for {ds}: {e}")
        plt.close()

    # ---------- print summary metrics ----------
    if final_epoch_f1s:
        final_epoch_f1s = np.array(final_epoch_f1s)
        print(
            f"{ds} Final-Epoch Macro-F1 (mean ± std): "
            f"{final_epoch_f1s.mean():.4f} ± {final_epoch_f1s.std(ddof=1):.4f}"
        )

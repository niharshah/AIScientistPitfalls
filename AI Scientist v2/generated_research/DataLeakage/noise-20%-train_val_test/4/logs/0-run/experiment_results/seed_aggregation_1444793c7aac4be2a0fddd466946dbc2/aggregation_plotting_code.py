import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, f1_score

# -------------------- setup --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# List of provided experiment_data.npy paths (relative to AI_SCIENTIST_ROOT)
experiment_data_path_list = [
    "experiments/2025-08-17_00-44-46_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_27ff76b6dabb41afae7c6b4c08557054_proc_3154745/experiment_data.npy",
    "experiments/2025-08-17_00-44-46_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_8eb342d1e8a7476cb38aecfcba0d64e0_proc_3154744/experiment_data.npy",
    "experiments/2025-08-17_00-44-46_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_5930aab86e8d4bf8897b72ce48b48c88_proc_3154743/experiment_data.npy",
]

all_experiment_data = []
try:
    root = os.getenv("AI_SCIENTIST_ROOT", ".")
    for p in experiment_data_path_list:
        full_path = os.path.join(root, p)
        exp = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp)
except Exception as e:
    print(f"Error loading experiment data: {e}")


# -------------------- helper for stacking --------------------
def stack_metric(runs, ds, outer_key, inner_key):
    """Collects a metric across runs. Returns list of np.ndarray (one per run)."""
    arrs = []
    for run in runs:
        value = run.get(ds, {}).get(outer_key, {}).get(inner_key, [])
        if isinstance(value, list):
            value = np.asarray(value)
        arrs.append(value)
    # Keep only runs that have data and truncate to common min length
    arrs = [a for a in arrs if a.size > 0]
    if not arrs:
        return None
    min_len = min([len(a) for a in arrs])
    arrs = [a[:min_len] for a in arrs]
    return np.stack(arrs)  # shape (R, E)


def sem(a, axis=0):
    return np.std(a, axis=axis, ddof=1) / np.sqrt(a.shape[axis])


# -------------------- iterate over datasets --------------------
all_dataset_names = set()
for exp in all_experiment_data:
    all_dataset_names.update(exp.keys())

# Limit to at most 5 datasets to respect figure count (plot 3 per ds)
max_datasets_to_plot = 5
for ds_idx, ds_name in enumerate(sorted(all_dataset_names)):
    if ds_idx >= max_datasets_to_plot:
        break

    # ------------- Loss curves with mean ± SEM -------------
    try:
        train_losses = stack_metric(all_experiment_data, ds_name, "losses", "train")
        val_losses = stack_metric(all_experiment_data, ds_name, "losses", "val")
        epochs = None
        # Use epochs from first run that has them
        for run in all_experiment_data:
            ep = run.get(ds_name, {}).get("epochs", [])
            if ep:
                epochs = (
                    np.asarray(ep)[: train_losses.shape[1]]
                    if train_losses is not None
                    else np.asarray(ep)[: val_losses.shape[1]]
                )
                break

        if train_losses is not None and val_losses is not None and epochs is not None:
            plt.figure()
            # Train
            mean_tr = train_losses.mean(axis=0)
            sem_tr = sem(train_losses, axis=0)
            plt.plot(epochs, mean_tr, label="Train Loss (mean)", color="tab:blue")
            plt.fill_between(
                epochs,
                mean_tr - sem_tr,
                mean_tr + sem_tr,
                color="tab:blue",
                alpha=0.3,
                label="Train ± SEM",
            )
            # Validation
            mean_val = val_losses.mean(axis=0)
            sem_val = sem(val_losses, axis=0)
            plt.plot(
                epochs, mean_val, label="Validation Loss (mean)", color="tab:orange"
            )
            plt.fill_between(
                epochs,
                mean_val - sem_val,
                mean_val + sem_val,
                color="tab:orange",
                alpha=0.3,
                label="Val ± SEM",
            )

            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(
                f"{ds_name} Loss Curve (Aggregated over {train_losses.shape[0]} runs)\nLeft: Train, Right: Validation"
            )
            plt.legend()
            fname = os.path.join(
                working_dir, f"{ds_name.lower()}_aggregated_loss_curve.png"
            )
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss curve for {ds_name}: {e}")
        plt.close()

    # ------------- Macro-F1 curves with mean ± SEM -------------
    try:
        train_f1 = stack_metric(all_experiment_data, ds_name, "metrics", "train_f1")
        val_f1 = stack_metric(all_experiment_data, ds_name, "metrics", "val_f1")

        if train_f1 is not None and val_f1 is not None and epochs is not None:
            plt.figure()
            # Train
            mean_tr = train_f1.mean(axis=0)
            sem_tr = sem(train_f1, axis=0)
            plt.plot(epochs, mean_tr, label="Train Macro-F1 (mean)", color="tab:green")
            plt.fill_between(
                epochs,
                mean_tr - sem_tr,
                mean_tr + sem_tr,
                color="tab:green",
                alpha=0.3,
                label="Train ± SEM",
            )
            # Validation
            mean_val = val_f1.mean(axis=0)
            sem_val = sem(val_f1, axis=0)
            plt.plot(
                epochs, mean_val, label="Validation Macro-F1 (mean)", color="tab:red"
            )
            plt.fill_between(
                epochs,
                mean_val - sem_val,
                mean_val + sem_val,
                color="tab:red",
                alpha=0.3,
                label="Val ± SEM",
            )

            plt.xlabel("Epoch")
            plt.ylabel("Macro-F1")
            plt.title(
                f"{ds_name} Macro-F1 Curve (Aggregated over {train_f1.shape[0]} runs)\nLeft: Train, Right: Validation"
            )
            plt.legend()
            fname = os.path.join(
                working_dir, f"{ds_name.lower()}_aggregated_f1_curve.png"
            )
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated F1 curve for {ds_name}: {e}")
        plt.close()

    # ------------- Confusion matrix on concatenated test sets -------------
    try:
        # Concatenate preds and gts across runs
        all_preds, all_gts = [], []
        for run in all_experiment_data:
            preds = run.get(ds_name, {}).get("predictions", [])
            gts = run.get(ds_name, {}).get("ground_truth", [])
            if preds and gts:
                all_preds.extend(preds)
                all_gts.extend(gts)
        if all_preds and all_gts:
            cm = confusion_matrix(all_gts, all_preds)
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title(
                f"{ds_name} Confusion Matrix\nDataset: Test Split (all runs combined)"
            )
            fname = os.path.join(
                working_dir, f"{ds_name.lower()}_combined_confusion_matrix.png"
            )
            plt.savefig(fname)
            plt.close()

            # Also print overall Macro-F1
            print(
                f"{ds_name} combined Test Macro-F1:",
                f1_score(all_gts, all_preds, average="macro"),
            )
    except Exception as e:
        print(f"Error creating combined confusion matrix for {ds_name}: {e}")
        plt.close()

# ------------- Summary barplot of final test Macro-F1 across runs -------------
try:
    for ds_name in sorted(all_dataset_names)[:max_datasets_to_plot]:
        scores = []
        for run in all_experiment_data:
            preds = run.get(ds_name, {}).get("predictions", [])
            gts = run.get(ds_name, {}).get("ground_truth", [])
            if preds and gts:
                scores.append(f1_score(gts, preds, average="macro"))
        if scores:
            scores = np.asarray(scores)
            mean_score = scores.mean()
            sem_score = scores.std(ddof=1) / np.sqrt(len(scores))

            plt.figure()
            x = np.arange(len(scores))
            plt.bar(x, scores, color="tab:purple", alpha=0.7, label="Individual runs")
            plt.errorbar(
                len(scores) + 0.5,
                mean_score,
                yerr=sem_score,
                fmt="o",
                color="black",
                ecolor="black",
                capsize=5,
                label=f"Mean ± SEM ({mean_score:.3f}±{sem_score:.3f})",
            )
            plt.xticks(list(x) + [len(scores) + 0.5], [f"run{i}" for i in x] + ["mean"])
            plt.ylabel("Macro-F1")
            plt.title(f"{ds_name} Final Test Macro-F1 Across Runs")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name.lower()}_test_f1_summary.png")
            plt.savefig(fname)
            plt.close()
except Exception as e:
    print(f"Error creating summary barplot: {e}")
    plt.close()

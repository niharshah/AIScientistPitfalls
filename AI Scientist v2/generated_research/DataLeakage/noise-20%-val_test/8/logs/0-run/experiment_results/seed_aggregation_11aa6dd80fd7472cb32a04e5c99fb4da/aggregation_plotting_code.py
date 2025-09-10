import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
# BASIC SETUP
# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# LOAD ALL EXPERIMENTS
# ------------------------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-17_02-43-50_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_789732ff8980401ba1982918281effe8_proc_3198564/experiment_data.npy",
    "experiments/2025-08-17_02-43-50_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_909a9819078c4981844861b3415c6d26_proc_3198563/experiment_data.npy",
    "experiments/2025-08-17_02-43-50_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_04bbe45aec6349ac92e1161e452c711b_proc_3198566/experiment_data.npy",
]
all_experiment_data = []
for p in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        all_experiment_data.append(np.load(full_path, allow_pickle=True).item())
    except Exception as e:
        print(f"Error loading experiment data ({p}): {e}")

# ------------------------------------------------------------------
# ORGANISE RUNS BY DATASET
# ------------------------------------------------------------------
datasets = {}
for run in all_experiment_data:
    for dname, ddict in run.items():
        datasets.setdefault(dname, []).append(ddict)

# ------------------------------------------------------------------
# AGGREGATED PLOTS
# ------------------------------------------------------------------
for dname, run_list in datasets.items():
    # ------------------- COLLECT SERIES ---------------------------
    val_loss_runs, val_acc_runs = [], []
    preds_runs, gts_runs = [], []

    for r in run_list:
        losses = r.get("losses", {})
        metrics = r.get("metrics", {})
        # Ensure lists/arrays
        if "val" in losses and len(losses["val"]):
            val_loss_runs.append(np.asarray(losses["val"], dtype=float))
        if "val" in metrics and len(metrics["val"]):
            val_acc_runs.append(np.asarray(metrics["val"], dtype=float))
        preds = np.asarray(r.get("predictions", []))
        gts = np.asarray(r.get("ground_truth", []))
        if preds.size and gts.size and preds.shape == gts.shape:
            preds_runs.append(preds)
            gts_runs.append(gts)

    # Align epochs to shortest run length
    def truncate_to_min_len(arrays):
        if not arrays:
            return []
        min_len = min(a.shape[0] for a in arrays)
        return [a[:min_len] for a in arrays]

    val_loss_runs = truncate_to_min_len(val_loss_runs)
    val_acc_runs = truncate_to_min_len(val_acc_runs)

    # ------------------- MEAN ± SE VAL LOSS -----------------------
    try:
        if val_loss_runs:
            data = np.vstack(val_loss_runs)
            mean = data.mean(axis=0)
            se = data.std(axis=0, ddof=1) / np.sqrt(data.shape[0])
            epochs = np.arange(1, len(mean) + 1)
            plt.figure()
            plt.plot(epochs, mean, label="Mean Val Loss", color="tab:blue")
            plt.fill_between(
                epochs, mean - se, mean + se, color="tab:blue", alpha=0.3, label="±1 SE"
            )
            plt.xlabel("Epoch")
            plt.ylabel("Validation Loss")
            plt.title(
                f"{dname} – Mean ± SE Validation Loss\n(Aggregated over {data.shape[0]} runs)"
            )
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_agg_val_loss.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated val loss for {dname}: {e}")
        plt.close()

    # ------------------- MEAN ± SE VAL ACC ------------------------
    try:
        if val_acc_runs:
            data = np.vstack(val_acc_runs)
            mean = data.mean(axis=0)
            se = data.std(axis=0, ddof=1) / np.sqrt(data.shape[0])
            epochs = np.arange(1, len(mean) + 1)
            plt.figure()
            plt.plot(epochs, mean, label="Mean Val Acc", color="tab:green")
            plt.fill_between(
                epochs,
                mean - se,
                mean + se,
                color="tab:green",
                alpha=0.3,
                label="±1 SE",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Validation Accuracy")
            plt.title(
                f"{dname} – Mean ± SE Validation Accuracy\n(Aggregated over {data.shape[0]} runs)"
            )
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_agg_val_accuracy.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated val accuracy for {dname}: {e}")
        plt.close()

    # ------------------- COMBINED CONFUSION MATRIX ---------------
    try:
        if preds_runs and gts_runs:
            from sklearn.metrics import confusion_matrix

            combined_cm = None
            for p, t in zip(preds_runs, gts_runs):
                cm = confusion_matrix(t, p)
                combined_cm = cm if combined_cm is None else combined_cm + cm
            plt.figure()
            im = plt.imshow(combined_cm, cmap="Blues")
            plt.title(
                f"{dname} – Combined Confusion Matrix\n(Aggregated over {len(preds_runs)} runs)"
            )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            for (i, j), v in np.ndenumerate(combined_cm):
                plt.text(j, i, str(v), ha="center", va="center")
            plt.colorbar(im)
            fname = os.path.join(working_dir, f"{dname}_agg_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated confusion matrix for {dname}: {e}")
        plt.close()

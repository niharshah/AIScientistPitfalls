import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------  BASIC SETUP ------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Paths of all experiment_data.npy files (relative to AI_SCIENTIST_ROOT)
experiment_data_path_list = [
    "experiments/2025-08-17_02-43-50_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_7e6e78e4fc04466883e1ac565701e76d_proc_3207743/experiment_data.npy",
    "experiments/2025-08-17_02-43-50_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_53ef2c2e7ac948f5bbd6b7ea6d4ca68f_proc_3207745/experiment_data.npy",
    "experiments/2025-08-17_02-43-50_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_b03bbb1e0388451eb2765d8bc300473b_proc_3207742/experiment_data.npy",
]

all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp_data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# ------------------  AGGREGATE ACROSS RUNS ------------------
aggregated = (
    {}
)  # {dataset: {'val_loss': [...], 'val_acc': [...], 'pred': [...], 'true': [...]}}

for exp in all_experiment_data:
    for dname, ddict in exp.items():
        agg = aggregated.setdefault(
            dname, {"val_loss": [], "val_acc": [], "pred": [], "true": []}
        )

        # losses & metrics
        losses = ddict.get("losses", {})
        metrics = ddict.get("metrics", {})

        agg["val_loss"].append(losses.get("val", []))
        agg["val_acc"].append(metrics.get("val", []))

        # predictions / ground truth (optional)
        if "predictions" in ddict and "ground_truth" in ddict:
            agg["pred"].append(np.asarray(ddict["predictions"]))
            agg["true"].append(np.asarray(ddict["ground_truth"]))

# ------------------  PLOTTING ------------------
for dname, agg in aggregated.items():

    # ---------- Aggregated Validation-Loss ----------
    try:
        if any(len(run) for run in agg["val_loss"]):
            runs = agg["val_loss"]
            max_len = max(len(r) for r in runs)
            mat = np.full((len(runs), max_len), np.nan)
            for i, seq in enumerate(runs):
                mat[i, : len(seq)] = seq

            mean_vals = np.nanmean(mat, axis=0)
            stderr = np.nanstd(mat, axis=0) / np.sqrt(np.sum(~np.isnan(mat), axis=0))
            epochs = np.arange(1, max_len + 1)

            plt.figure()
            plt.plot(epochs, mean_vals, color="blue", label="Mean Validation Loss")
            plt.fill_between(
                epochs,
                mean_vals - stderr,
                mean_vals + stderr,
                color="blue",
                alpha=0.25,
                label="Std. Error",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dname} – Aggregated Validation Loss (N={len(runs)})")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_aggregated_val_loss.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated val-loss plot for {dname}: {e}")
        plt.close()

    # ---------- Aggregated Validation-Accuracy ----------
    try:
        if any(len(run) for run in agg["val_acc"]):
            runs = agg["val_acc"]
            max_len = max(len(r) for r in runs)
            mat = np.full((len(runs), max_len), np.nan)
            for i, seq in enumerate(runs):
                mat[i, : len(seq)] = seq

            mean_vals = np.nanmean(mat, axis=0)
            stderr = np.nanstd(mat, axis=0) / np.sqrt(np.sum(~np.isnan(mat), axis=0))
            epochs = np.arange(1, max_len + 1)

            plt.figure()
            plt.plot(epochs, mean_vals, color="green", label="Mean Validation Accuracy")
            plt.fill_between(
                epochs,
                mean_vals - stderr,
                mean_vals + stderr,
                color="green",
                alpha=0.25,
                label="Std. Error",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"{dname} – Aggregated Validation Accuracy (N={len(runs)})")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_aggregated_val_accuracy.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated val-accuracy plot for {dname}: {e}")
        plt.close()

    # ---------- Aggregated Confusion Matrix ----------
    try:
        if len(agg["pred"]) and len(agg["true"]):
            from sklearn.metrics import confusion_matrix

            y_pred_all = np.concatenate(agg["pred"])
            y_true_all = np.concatenate(agg["true"])

            cm = confusion_matrix(y_true_all, y_pred_all)
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.title(f"{dname} – Confusion Matrix (Aggregated)")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            for (i, j), v in np.ndenumerate(cm):
                plt.text(j, i, str(v), ha="center", va="center")
            plt.colorbar(im)
            fname = os.path.join(
                working_dir, f"{dname}_aggregated_confusion_matrix.png"
            )
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated confusion matrix for {dname}: {e}")
        plt.close()

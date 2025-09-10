import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
# Basic setup
# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data_path_list = [
    "experiments/2025-08-17_02-43-50_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_27f83dee1ff4411daa14d4512db1c994_proc_3203082/experiment_data.npy",
    "experiments/2025-08-17_02-43-50_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_96cd130c50484913821e9fcc3d4dc3c1_proc_3203080/experiment_data.npy",
    "experiments/2025-08-17_02-43-50_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_99b154b5f3c9481f9384e7d24cec6155_proc_3203079/experiment_data.npy",
]

# ------------------------------------------------------------------
# Load every experiment file that exists
# ------------------------------------------------------------------
all_runs = []
for p in experiment_data_path_list:
    try:
        data = np.load(p, allow_pickle=True).item()
        all_runs.append(data)
    except Exception as e:
        print(f"Error loading {p}: {e}")

# ------------------------------------------------------------------
# Aggregate across runs
# ------------------------------------------------------------------
aggregated = {}
for run in all_runs:
    for dname, ddict in run.items():
        agg_d = aggregated.setdefault(
            dname,
            {
                "val_loss_runs": [],
                "val_acc_runs": [],
                "train_loss_runs": [],
                "y_true": [],
                "y_pred": [],
            },
        )
        losses = ddict.get("losses", {})
        metrics = ddict.get("metrics", {})
        # validation loss
        if isinstance(losses, dict):
            val_loss = losses.get("val", [])
            train_loss = losses.get("train", [])
        else:
            val_loss = []
            train_loss = []
        agg_d["val_loss_runs"].append(np.asarray(val_loss, dtype=float))
        agg_d["train_loss_runs"].append(np.asarray(train_loss, dtype=float))
        # validation accuracy  (could be stored several different ways)
        if isinstance(metrics, dict):
            val_acc = metrics.get("val", metrics.get("accuracy", []))
        else:
            val_acc = []
        agg_d["val_acc_runs"].append(np.asarray(val_acc, dtype=float))
        # preds / gt if present
        agg_d["y_true"].extend(ddict.get("ground_truth", []))
        agg_d["y_pred"].extend(ddict.get("predictions", []))


# ------------------------------------------------------------------
# Helper to stack ragged arrays by padding
# ------------------------------------------------------------------
def stack_and_pad(list_of_1d_arrays):
    if not list_of_1d_arrays:
        return np.empty((0, 0))
    max_len = max(len(a) for a in list_of_1d_arrays)
    padded = np.full((len(list_of_1d_arrays), max_len), np.nan)
    for i, arr in enumerate(list_of_1d_arrays):
        padded[i, : len(arr)] = arr
    return padded


# ------------------------------------------------------------------
# Create plots per dataset
# ------------------------------------------------------------------
for dname, ddata in aggregated.items():

    # 1) Validation Loss -------------------------------------------------
    try:
        val_loss_mat = stack_and_pad(ddata["val_loss_runs"])
        if val_loss_mat.size:
            mean_loss = np.nanmean(val_loss_mat, axis=0)
            stderr_loss = np.nanstd(val_loss_mat, axis=0, ddof=1) / np.sqrt(
                val_loss_mat.shape[0]
            )
            epochs = np.arange(1, len(mean_loss) + 1)

            plt.figure()
            plt.plot(epochs, mean_loss, label="Mean")
            plt.fill_between(
                epochs,
                mean_loss - stderr_loss,
                mean_loss + stderr_loss,
                alpha=0.3,
                label="± StdErr",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Validation Loss")
            plt.title(f"{dname} – Mean ± StdErr Validation Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_aggregated_val_loss.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated val loss for {dname}: {e}")
        plt.close()

    # 2) Validation Accuracy --------------------------------------------
    try:
        val_acc_mat = stack_and_pad(ddata["val_acc_runs"])
        if val_acc_mat.size:
            mean_acc = np.nanmean(val_acc_mat, axis=0)
            stderr_acc = np.nanstd(val_acc_mat, axis=0, ddof=1) / np.sqrt(
                val_acc_mat.shape[0]
            )
            epochs = np.arange(1, len(mean_acc) + 1)

            plt.figure()
            plt.plot(epochs, mean_acc, color="green", label="Mean")
            plt.fill_between(
                epochs,
                mean_acc - stderr_acc,
                mean_acc + stderr_acc,
                alpha=0.3,
                color="green",
                label="± StdErr",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Validation Accuracy")
            plt.title(f"{dname} – Mean ± StdErr Validation Accuracy")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_aggregated_val_accuracy.png")
            plt.savefig(fname)
            plt.close()

            # Print final epoch summary
            final_mean = mean_acc[~np.isnan(mean_acc)][-1]
            final_stderr = stderr_acc[~np.isnan(stderr_acc)][-1]
            print(
                f"{dname}: final-epoch val accuracy = {final_mean:.4f} ± {final_stderr:.4f}"
            )
    except Exception as e:
        print(f"Error creating aggregated val accuracy for {dname}: {e}")
        plt.close()

    # 3) Training Loss ---------------------------------------------------
    try:
        train_loss_mat = stack_and_pad(ddata["train_loss_runs"])
        if train_loss_mat.size and not np.all(np.isnan(train_loss_mat)):
            mean_tloss = np.nanmean(train_loss_mat, axis=0)
            stderr_tloss = np.nanstd(train_loss_mat, axis=0, ddof=1) / np.sqrt(
                train_loss_mat.shape[0]
            )
            epochs = np.arange(1, len(mean_tloss) + 1)

            plt.figure()
            plt.plot(epochs, mean_tloss, color="orange", label="Mean")
            plt.fill_between(
                epochs,
                mean_tloss - stderr_tloss,
                mean_tloss + stderr_tloss,
                alpha=0.3,
                color="orange",
                label="± StdErr",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Training Loss")
            plt.title(f"{dname} – Mean ± StdErr Training Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_aggregated_train_loss.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated train loss for {dname}: {e}")
        plt.close()

    # 4) Confusion Matrix ------------------------------------------------
    try:
        y_true = np.asarray(ddata["y_true"])
        y_pred = np.asarray(ddata["y_pred"])
        if y_true.size and y_pred.size:
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(y_true, y_pred)
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

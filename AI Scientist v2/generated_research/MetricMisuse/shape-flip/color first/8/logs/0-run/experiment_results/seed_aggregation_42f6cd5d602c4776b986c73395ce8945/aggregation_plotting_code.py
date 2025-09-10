import matplotlib.pyplot as plt
import numpy as np
import os
import math

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- list all experiment_data paths ----------
experiment_data_path_list = [
    os.path.join(working_dir, "experiment_data.npy"),
    "experiments/2025-08-30_21-49-50_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_3d72e717607241d9b3da9f8a2bb47b25_proc_1512111/experiment_data.npy",
    "None/experiment_data.npy",  # will fail gracefully
]

all_experiments = []
for p in experiment_data_path_list:
    try:
        exp = np.load(p, allow_pickle=True).item()
        all_experiments.append(exp)
    except Exception as e:
        print(f"Error loading {p}: {e}")

# ---------- collect dataset names ----------
dataset_names = set()
for exp in all_experiments:
    dataset_names.update(exp.keys())


# ---------- helper to stack, trim and get mean / se ----------
def stack_and_trim(list_of_arrays):
    if not list_of_arrays:
        return None, None
    min_len = min(len(a) for a in list_of_arrays)
    trimmed = np.stack([a[:min_len] for a in list_of_arrays], axis=0)
    mean = trimmed.mean(axis=0)
    se = (
        trimmed.std(axis=0, ddof=1) / math.sqrt(trimmed.shape[0])
        if trimmed.shape[0] > 1
        else np.zeros_like(mean)
    )
    return mean, se


# ---------- iterate over datasets ----------
for ds_name in dataset_names:
    # collect arrays from each run
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    preds_all, gts_all = [], []

    for exp in all_experiments:
        ds = exp.get(ds_name, {})
        losses = ds.get("losses", {})
        metrics = ds.get("metrics", {})

        if "train" in losses and isinstance(losses["train"], (list, np.ndarray)):
            train_losses.append(np.asarray(losses["train"], dtype=float))
        if "val" in losses and isinstance(losses["val"], (list, np.ndarray)):
            val_losses.append(np.asarray(losses["val"], dtype=float))
        if "train" in metrics and isinstance(metrics["train"], (list, np.ndarray)):
            train_accs.append(np.asarray(metrics["train"], dtype=float))
        if "val" in metrics and isinstance(metrics["val"], (list, np.ndarray)):
            val_accs.append(np.asarray(metrics["val"], dtype=float))

        # predictions / gts for confusion matrix
        if "predictions" in ds and "ground_truth" in ds:
            p = np.asarray(ds["predictions"])
            g = np.asarray(ds["ground_truth"])
            if p.size and g.size and p.shape == g.shape:
                preds_all.append(p)
                gts_all.append(g)

    # ---------- aggregate loss curves ----------
    try:
        mean_train_loss, se_train_loss = stack_and_trim(train_losses)
        mean_val_loss, se_val_loss = stack_and_trim(val_losses)

        if mean_train_loss is not None or mean_val_loss is not None:
            plt.figure()
            epochs = np.arange(
                len(mean_train_loss if mean_train_loss is not None else mean_val_loss)
            )

            if mean_train_loss is not None:
                plt.plot(epochs, mean_train_loss, label="Train Mean")
                plt.fill_between(
                    epochs,
                    mean_train_loss - se_train_loss,
                    mean_train_loss + se_train_loss,
                    alpha=0.3,
                    label="Train ±1SE",
                )
            if mean_val_loss is not None:
                plt.plot(epochs, mean_val_loss, label="Val Mean")
                plt.fill_between(
                    epochs,
                    mean_val_loss - se_val_loss,
                    mean_val_loss + se_val_loss,
                    alpha=0.3,
                    label="Val ±1SE",
                )

            plt.title(f"{ds_name} Mean Loss Curve ± SE\nLeft: Train, Right: Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_mean_loss_curve.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss curve for {ds_name}: {e}")
        plt.close()

    # ---------- aggregate accuracy curves ----------
    try:
        mean_train_acc, se_train_acc = stack_and_trim(train_accs)
        mean_val_acc, se_val_acc = stack_and_trim(val_accs)

        if mean_train_acc is not None or mean_val_acc is not None:
            plt.figure()
            epochs = np.arange(
                len(mean_train_acc if mean_train_acc is not None else mean_val_acc)
            )

            if mean_train_acc is not None:
                plt.plot(epochs, mean_train_acc, label="Train Mean")
                plt.fill_between(
                    epochs,
                    mean_train_acc - se_train_acc,
                    mean_train_acc + se_train_acc,
                    alpha=0.3,
                    label="Train ±1SE",
                )
            if mean_val_acc is not None:
                plt.plot(epochs, mean_val_acc, label="Val Mean")
                plt.fill_between(
                    epochs,
                    mean_val_acc - se_val_acc,
                    mean_val_acc + se_val_acc,
                    alpha=0.3,
                    label="Val ±1SE",
                )

            plt.title(
                f"{ds_name} Mean Accuracy Curve ± SE\nLeft: Train, Right: Validation"
            )
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_mean_accuracy_curve.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated accuracy curve for {ds_name}: {e}")
        plt.close()

    # ---------- aggregated confusion matrix ----------
    try:
        if preds_all and gts_all:
            num_classes = int(max(np.max(p) for p in preds_all + gts_all) + 1)
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for p, g in zip(preds_all, gts_all):
                for gt, pr in zip(g, p):
                    cm[gt, pr] += 1

            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(f"{ds_name} Aggregated Confusion Matrix\nLeft: GT, Right: Preds")
            ticks = np.arange(num_classes)
            plt.xticks(ticks, [f"c{i}" for i in ticks])
            plt.yticks(ticks, [f"c{i}" for i in ticks])
            plt.savefig(
                os.path.join(working_dir, f"{ds_name}_aggregated_confusion_matrix.png")
            )
            plt.close()
        else:
            print(f"Skipping confusion matrix for {ds_name}: missing predictions.")
    except Exception as e:
        print(f"Error creating aggregated confusion matrix for {ds_name}: {e}")
        plt.close()

    # ---------- print summary metric ----------
    if val_accs:
        final_vals = [a[min(len(a) - 1, len(a) - 1)] for a in val_accs]
        mean_final = np.mean(final_vals)
        se_final = (
            np.std(final_vals, ddof=1) / math.sqrt(len(final_vals))
            if len(final_vals) > 1
            else 0.0
        )
        print(f"{ds_name} final validation accuracy: {mean_final:.4f} ± {se_final:.4f}")

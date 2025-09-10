import matplotlib.pyplot as plt
import numpy as np
import os

# --------- setup ---------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------- load data from all runs ---------
experiment_data_path_list = [
    "experiments/2025-08-30_19-33-09_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_d9995f30f62a4cf7bac3b4a616b12129_proc_1471560/experiment_data.npy",
    "experiments/2025-08-30_19-33-09_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_97963e99e71d476c8b81c261e0c717ff_proc_1471559/experiment_data.npy",
    "experiments/2025-08-30_19-33-09_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_89f464be9813416490cbde90035bcb44_proc_1471561/experiment_data.npy",
]

all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        all_experiment_data.append(np.load(full_path, allow_pickle=True).item())
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []


# --------- helper functions ---------
def aggregate_runs(list_of_arrays):
    """
    Stack 1-D arrays from different runs and return:
    mean, sem, and the epoch indices (truncated to min length).
    """
    if not list_of_arrays:
        return None, None, None
    min_len = min(len(a) for a in list_of_arrays)
    if min_len == 0:
        return None, None, None
    data = np.stack([a[:min_len] for a in list_of_arrays], axis=0)
    mean = data.mean(axis=0)
    sem = data.std(axis=0, ddof=1) / np.sqrt(data.shape[0])
    epochs = np.arange(min_len)
    return mean, sem, epochs


# Collect all dataset names present in any run
dataset_names = set()
for exp in all_experiment_data:
    dataset_names.update(exp.keys())

# --------- plotting ---------
for ds_name in dataset_names:
    # Gather per-run data
    train_losses_runs, val_losses_runs, val_cwa2_runs = [], [], []
    preds_runs, gts_runs = [], []
    for exp in all_experiment_data:
        if ds_name not in exp:
            continue
        ds_dict = exp[ds_name]
        losses = ds_dict.get("losses", {})
        metrics = ds_dict.get("metrics", {})
        if losses.get("train"):
            train_losses_runs.append(np.asarray(losses["train"]))
        if losses.get("val"):
            val_losses_runs.append(np.asarray(losses["val"]))
        if metrics.get("val_cwa2"):
            val_cwa2_runs.append(np.asarray(metrics["val_cwa2"]))
        preds_runs.append(np.asarray(ds_dict.get("predictions", [])))
        gts_runs.append(np.asarray(ds_dict.get("ground_truth", [])))

    # ---------- aggregated loss curves ----------
    try:
        mean_train, sem_train, epochs = aggregate_runs(train_losses_runs)
        mean_val, sem_val, _ = aggregate_runs(val_losses_runs)
        if epochs is not None:
            plt.figure()
            if mean_train is not None:
                plt.plot(epochs, mean_train, label="Train Loss (mean)")
                plt.fill_between(
                    epochs,
                    mean_train - sem_train,
                    mean_train + sem_train,
                    alpha=0.2,
                    label="Train SEM",
                )
            if mean_val is not None:
                plt.plot(epochs, mean_val, label="Val Loss (mean)")
                plt.fill_between(
                    epochs,
                    mean_val - sem_val,
                    mean_val + sem_val,
                    alpha=0.2,
                    label="Val SEM",
                )
            plt.title(f"{ds_name} – Aggregated Training vs Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.legend()
            save_path = os.path.join(working_dir, f"{ds_name}_agg_loss_curves.png")
            plt.savefig(save_path)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss curve for {ds_name}: {e}")
        plt.close()

    # ---------- aggregated validation CWA2 ----------
    try:
        mean_cwa2, sem_cwa2, epochs = aggregate_runs(val_cwa2_runs)
        if epochs is not None:
            plt.figure()
            plt.plot(epochs, mean_cwa2, color="tab:orange", label="Val CWA2 (mean)")
            plt.fill_between(
                epochs,
                mean_cwa2 - sem_cwa2,
                mean_cwa2 + sem_cwa2,
                alpha=0.2,
                label="SEM",
            )
            plt.title(f"{ds_name} – Aggregated Validation CWA2 over Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("CWA2")
            plt.legend()
            save_path = os.path.join(working_dir, f"{ds_name}_agg_val_cwa2.png")
            plt.savefig(save_path)
            plt.close()

            # print final aggregated metric
            print(f"{ds_name} final Val CWA2: {mean_cwa2[-1]:.4f} ± {sem_cwa2[-1]:.4f}")
    except Exception as e:
        print(f"Error creating aggregated CWA2 plot for {ds_name}: {e}")
        plt.close()

    # ---------- aggregated confusion matrix ----------
    try:
        # Build cumulative confusion matrix
        agg_cm = None
        for preds, gts in zip(preds_runs, gts_runs):
            if preds.size == 0 or gts.size == 0:
                continue
            num_classes = max(preds.max(), gts.max()) + 1
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            if agg_cm is None:
                agg_cm = cm
            else:
                # Expand if number of classes differs
                if cm.shape != agg_cm.shape:
                    new_n = max(cm.shape[0], agg_cm.shape[0])
                    new_cm = np.zeros((new_n, new_n), dtype=int)
                    new_cm[: agg_cm.shape[0], : agg_cm.shape[1]] += agg_cm
                    new_cm[: cm.shape[0], : cm.shape[1]] += cm
                    agg_cm = new_cm
                else:
                    agg_cm += cm
        if agg_cm is not None:
            plt.figure()
            im = plt.imshow(agg_cm, cmap="Blues")
            plt.colorbar(im)
            plt.title(f"{ds_name} – Aggregated Confusion Matrix\n(GT rows, Pred cols)")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            save_path = os.path.join(working_dir, f"{ds_name}_agg_confusion_matrix.png")
            plt.savefig(save_path)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated confusion matrix for {ds_name}: {e}")
        plt.close()

import matplotlib.pyplot as plt
import numpy as np
import os

# House-keeping
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# Load all experiment_data dicts that were passed by the orchestrator
# ------------------------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-16_02-32-02_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_34957ba5a7784bbea43810148832dbb5_proc_3097193/experiment_data.npy",
    "experiments/2025-08-16_02-32-02_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_9de1d88ec77a440c9319bfd5e574461b_proc_3097195/experiment_data.npy",
    "experiments/2025-08-16_02-32-02_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_5f31e78a744845f086bd621146eb3b56_proc_3097194/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        exp_dict = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_experiment_data.append(exp_dict)
    except Exception as e:
        print(f"Error loading {p}: {e}")

if not all_experiment_data:
    print("No experiment data could be loaded. Exiting.")
    exit()


# ---------------------------------------------------------
# Aggregate over runs for every dataset key that is present
# ---------------------------------------------------------
def pad_to(arr, length):
    """Pad 1-D list/array with np.nan so that len==length."""
    out = np.full(length, np.nan)
    out[: len(arr)] = arr
    return out


for dataset in all_experiment_data[0].keys():
    # Collect metrics over runs
    metric_names = ["train_loss", "val_loss", "SWA", "CWA", "HWA"]
    per_metric_runs = {m: [] for m in metric_names}
    preds_all, gts_all = [], []

    for run_dict in all_experiment_data:
        data = run_dict.get(dataset, {})
        metrics = data.get("metrics", {})
        for m in metric_names:
            per_metric_runs[m].append(metrics.get(m, []))
        preds_all.extend(data.get("predictions", []))
        gts_all.extend(data.get("ground_truth", []))

    # Determine maximum epoch length across runs for each metric group
    max_len_loss = max(
        len(x) for x in per_metric_runs["train_loss"] + per_metric_runs["val_loss"]
    )
    max_len_acc = max(
        len(x)
        for x in per_metric_runs["SWA"]
        + per_metric_runs["CWA"]
        + per_metric_runs["HWA"]
    )
    epochs_loss = np.arange(1, max_len_loss + 1)
    epochs_acc = np.arange(1, max_len_acc + 1)

    # ---------------------------- Plot 1: Loss curves ----------------------------
    try:
        train_matrix = np.vstack(
            [pad_to(r, max_len_loss) for r in per_metric_runs["train_loss"]]
        )
        val_matrix = np.vstack(
            [pad_to(r, max_len_loss) for r in per_metric_runs["val_loss"]]
        )
        train_mean, train_sem = np.nanmean(train_matrix, axis=0), np.nanstd(
            train_matrix, axis=0
        ) / np.sqrt(train_matrix.shape[0])
        val_mean, val_sem = np.nanmean(val_matrix, axis=0), np.nanstd(
            val_matrix, axis=0
        ) / np.sqrt(val_matrix.shape[0])

        plt.figure()
        plt.plot(epochs_loss, train_mean, label="Train Loss (mean)")
        plt.fill_between(
            epochs_loss,
            train_mean - train_sem,
            train_mean + train_sem,
            alpha=0.3,
            label="Train ± SEM",
        )
        plt.plot(epochs_loss, val_mean, label="Val Loss (mean)")
        plt.fill_between(
            epochs_loss,
            val_mean - val_sem,
            val_mean + val_sem,
            alpha=0.3,
            label="Val ± SEM",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(
            f"{dataset} – Aggregated Training vs Validation Loss\nMean ± SEM over {train_matrix.shape[0]} runs"
        )
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset}_aggregated_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dataset}: {e}")
        plt.close()

    # ----------------------- Plot 2: Weighted accuracy curves ---------------------
    try:
        swa_mat = np.vstack([pad_to(r, max_len_acc) for r in per_metric_runs["SWA"]])
        cwa_mat = np.vstack([pad_to(r, max_len_acc) for r in per_metric_runs["CWA"]])
        hwa_mat = np.vstack([pad_to(r, max_len_acc) for r in per_metric_runs["HWA"]])

        def mean_sem(mat):
            return np.nanmean(mat, axis=0), np.nanstd(mat, axis=0) / np.sqrt(
                mat.shape[0]
            )

        swa_mean, swa_sem = mean_sem(swa_mat)
        cwa_mean, cwa_sem = mean_sem(cwa_mat)
        hwa_mean, hwa_sem = mean_sem(hwa_mat)

        plt.figure()
        for m, s, name in [
            (swa_mean, swa_sem, "SWA"),
            (cwa_mean, cwa_sem, "CWA"),
            (hwa_mean, hwa_sem, "HWA"),
        ]:
            plt.plot(epochs_acc, m, label=f"{name} (mean)")
            plt.fill_between(epochs_acc, m - s, m + s, alpha=0.3, label=f"{name} ± SEM")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(
            f"{dataset} – Aggregated Weighted Accuracies\nMean ± SEM over {swa_mat.shape[0]} runs"
        )
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"{dataset}_aggregated_weighted_accuracy.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated weighted accuracy plot for {dataset}: {e}")
        plt.close()

    # --------------------------- Plot 3: Confusion matrix -------------------------
    try:
        if preds_all and gts_all:
            labels = sorted(list(set(gts_all) | set(preds_all)))
            idx = {l: i for i, l in enumerate(labels)}
            cm = np.zeros((len(labels), len(labels)), dtype=int)
            for gt, pr in zip(gts_all, preds_all):
                cm[idx[gt], idx[pr]] += 1

            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
            plt.yticks(range(len(labels)), labels)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(
                f"{dataset} – Aggregated Confusion Matrix\nLeft: GT, Right: Pred (all runs combined)"
            )
            plt.savefig(
                os.path.join(working_dir, f"{dataset}_aggregated_confusion_matrix.png")
            )
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated confusion matrix for {dataset}: {e}")
        plt.close()

    # -------------------- Print final epoch aggregated metrics --------------------
    try:
        final_idx = (
            -1
        )  # last epoch common length may differ; use nanmean across last available values

        def last_valid(mat):  # take last non-nan from each run
            vals = []
            for row in mat:
                valid = row[~np.isnan(row)]
                if valid.size:
                    vals.append(valid[-1])
            return np.array(vals)

        for name, mat in [("SWA", swa_mat), ("CWA", cwa_mat), ("HWA", hwa_mat)]:
            vals = last_valid(mat)
            if vals.size:
                print(
                    f"{dataset} FINAL {name}: {np.mean(vals):.4f} ± {np.std(vals, ddof=1)/np.sqrt(vals.size):.4f}"
                )
    except Exception as e:
        print(f"Error printing final metrics for {dataset}: {e}")

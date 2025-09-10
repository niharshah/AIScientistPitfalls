import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
# basic set-up
# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# load ALL experiment files
# ------------------------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-17_00-44-36_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_a8a07c8aec564c0d945304911fc2a043_proc_3154643/experiment_data.npy",
    "experiments/2025-08-17_00-44-36_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_e8414550cb674b6db0e73a0b2a78f320_proc_3154645/experiment_data.npy",
    "experiments/2025-08-17_00-44-36_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_e92697a89b904015b862f30d7d1e8ed3_proc_3154644/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        exp = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_experiment_data.append(exp)
    except Exception as e:
        print(f"Error loading {p}: {e}")

# Work only if at least one file has been loaded and dataset exists
dataset_key = "SPR_BENCH"
runs = [ed[dataset_key] for ed in all_experiment_data if dataset_key in ed]

if len(runs) == 0:
    print("No valid runs found for dataset", dataset_key)
else:
    # ------------------------------------------------------------------
    # collect per-run arrays
    # ------------------------------------------------------------------
    epochs = np.array(runs[0]["epochs"])
    train_losses = []
    val_losses = []
    val_f1s = []
    all_preds = []
    all_gts = []

    for r in runs:
        train_losses.append(np.array(r["metrics"]["train_loss"]))
        val_losses.append(np.array(r["metrics"]["val_loss"]))
        val_f1s.append(np.array(r["metrics"]["val_f1"]))
        all_preds.append(np.array(r["predictions"]))
        all_gts.append(np.array(r["ground_truth"]))

    train_losses = np.stack(train_losses)  # shape: (n_runs, n_epochs)
    val_losses = np.stack(val_losses)
    val_f1s = np.stack(val_f1s)

    n_runs = train_losses.shape[0]
    sem = lambda x: np.std(x, axis=0, ddof=1) / np.sqrt(n_runs)

    # ------------------------------------------------------------------
    # 1) Mean ± SEM loss curves
    # ------------------------------------------------------------------
    try:
        plt.figure()
        t_mean, t_sem = train_losses.mean(axis=0), sem(train_losses)
        v_mean, v_sem = val_losses.mean(axis=0), sem(val_losses)

        plt.plot(epochs, t_mean, label="Train Loss (mean)", color="tab:blue")
        plt.fill_between(
            epochs, t_mean - t_sem, t_mean + t_sem, alpha=0.3, color="tab:blue"
        )

        plt.plot(epochs, v_mean, label="Validation Loss (mean)", color="tab:orange")
        plt.fill_between(
            epochs, v_mean - v_sem, v_mean + v_sem, alpha=0.3, color="tab:orange"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Train vs Val Loss (Mean ± SEM over runs)")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_mean_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # 2) Mean ± SEM validation F1 curve
    # ------------------------------------------------------------------
    try:
        plt.figure()
        f1_mean, f1_sem = val_f1s.mean(axis=0), sem(val_f1s)

        plt.plot(
            epochs,
            f1_mean,
            marker="o",
            color="green",
            label="Validation Macro-F1 (mean)",
        )
        plt.fill_between(
            epochs, f1_mean - f1_sem, f1_mean + f1_sem, alpha=0.3, color="green"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.title("SPR_BENCH: Validation Macro-F1 (Mean ± SEM over runs)")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_mean_val_f1.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated F1 plot: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # 3) Confusion matrix using concatenated predictions from all runs
    #    (plotted only once because class-wise aggregation is valid)
    # ------------------------------------------------------------------
    try:
        from sklearn.metrics import confusion_matrix

        concat_preds = np.concatenate(all_preds)
        concat_gts = np.concatenate(all_gts)

        cm = confusion_matrix(concat_gts, concat_preds)
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH: Aggregated Confusion Matrix (All Runs)")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_aggregated_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated confusion matrix: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # 4) Print aggregated final-epoch Macro-F1
    # ------------------------------------------------------------------
    final_epoch_f1s = val_f1s[:, -1]  # last value from each run
    print(
        f"Aggregated Final-Epoch Macro-F1: mean={final_epoch_f1s.mean():.4f} ± {sem(final_epoch_f1s):.4f}"
    )

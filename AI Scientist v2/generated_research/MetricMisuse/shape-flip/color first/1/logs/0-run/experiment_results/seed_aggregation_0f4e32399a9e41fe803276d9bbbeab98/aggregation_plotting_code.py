import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
# Mandatory working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# Load every experiment_data.npy that was provided
experiment_data_path_list = [
    "experiments/2025-08-30_17-49-30_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_ec0e78b6aa6a4fb9a7251877d9d4642a_proc_1437073/experiment_data.npy",
    "experiments/2025-08-30_17-49-30_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_b2f19f98da14403c8e2fc9214e5e40c1_proc_1437072/experiment_data.npy",
    "experiments/2025-08-30_17-49-30_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_ce3596fd883648c6bc7f7c8d5628948d_proc_1437074/experiment_data.npy",
]

all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        all_experiment_data.append(np.load(full_path, allow_pickle=True).item())
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

if not all_experiment_data:
    print("No experiment data could be loaded, exiting.")
    exit()


# ------------------------------------------------------------------
# Helper to safely save & close
def _save_and_close(fname: str):
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()


# ------------------------------------------------------------------
# Aggregate by dataset name
dataset_names = list(all_experiment_data[0].keys())
for dset in dataset_names:
    # Collect arrays for each run
    train_losses_runs, val_losses_runs = [], []
    acc_runs, cwa_runs, swa_runs, caa_runs = [], [], [], []
    preds_runs, gts_runs = [], []

    for run_data in all_experiment_data:
        if dset not in run_data:
            continue
        d = run_data[dset]
        train_losses_runs.append(np.asarray(d["losses"]["train"]))
        val_losses_runs.append(np.asarray(d["losses"]["val"]))

        val_metrics = d["metrics"]["val"]
        acc_runs.append(np.asarray([m["acc"] for m in val_metrics]))
        cwa_runs.append(np.asarray([m["cwa"] for m in val_metrics]))
        swa_runs.append(np.asarray([m["swa"] for m in val_metrics]))
        caa_runs.append(np.asarray([m["caa"] for m in val_metrics]))

        if "predictions" in d and "ground_truth" in d:
            preds_runs.append(np.asarray(d["predictions"]))
            gts_runs.append(np.asarray(d["ground_truth"]))

    # Ensure every run has the same number of epochs
    min_epochs = min(arr.shape[0] for arr in train_losses_runs)
    train_losses_runs = [arr[:min_epochs] for arr in train_losses_runs]
    val_losses_runs = [arr[:min_epochs] for arr in val_losses_runs]
    acc_runs, cwa_runs, swa_runs, caa_runs = [
        [arr[:min_epochs] for arr in metric_list]
        for metric_list in (acc_runs, cwa_runs, swa_runs, caa_runs)
    ]
    epochs = np.arange(1, min_epochs + 1)

    # Stack and compute mean & SEM
    def mean_sem(run_list):
        runs = np.stack(run_list, axis=0)
        mean = runs.mean(axis=0)
        sem = runs.std(axis=0, ddof=1) / np.sqrt(runs.shape[0])
        return mean, sem

    train_mean, train_sem = mean_sem(train_losses_runs)
    val_mean, val_sem = mean_sem(val_losses_runs)
    acc_mean, acc_sem = mean_sem(acc_runs)
    cwa_mean, cwa_sem = mean_sem(cwa_runs)
    swa_mean, swa_sem = mean_sem(swa_runs)
    caa_mean, caa_sem = mean_sem(caa_runs)

    # -------------------------------------------------------------- #
    # Plot 1: aggregated loss curves
    try:
        plt.figure()
        plt.plot(epochs, train_mean, label="Train Loss – mean")
        plt.fill_between(
            epochs,
            train_mean - train_sem,
            train_mean + train_sem,
            alpha=0.3,
            label="Train SEM",
        )
        plt.plot(epochs, val_mean, label="Val Loss – mean")
        plt.fill_between(
            epochs, val_mean - val_sem, val_mean + val_sem, alpha=0.3, label="Val SEM"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(
            f"{dset} – Training vs Validation Loss (mean ± SEM)\n(n={len(train_losses_runs)} runs)"
        )
        plt.legend()
        _save_and_close(f"{dset}_agg_loss_curve.png")
    except Exception as e:
        print(f"Error creating aggregated loss curve for {dset}: {e}")
        plt.close()

    # -------------------------------------------------------------- #
    # Plot 2: aggregated validation accuracy
    try:
        plt.figure()
        plt.plot(epochs, acc_mean, marker="o", label="Accuracy – mean")
        plt.fill_between(
            epochs, acc_mean - acc_sem, acc_mean + acc_sem, alpha=0.3, label="SEM"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{dset} – Validation Accuracy (mean ± SEM)")
        plt.legend()
        _save_and_close(f"{dset}_agg_val_accuracy.png")
    except Exception as e:
        print(f"Error creating aggregated accuracy plot for {dset}: {e}")
        plt.close()

    # -------------------------------------------------------------- #
    # Plot 3: aggregated weighted accuracies
    try:
        plt.figure()
        for curve_mean, curve_sem, name in [
            (cwa_mean, cwa_sem, "CWA"),
            (swa_mean, swa_sem, "SWA"),
            (caa_mean, caa_sem, "CAA"),
        ]:
            plt.plot(epochs, curve_mean, label=f"{name} – mean")
            plt.fill_between(
                epochs, curve_mean - curve_sem, curve_mean + curve_sem, alpha=0.3
            )
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(f"{dset} – Weighted Accuracies (mean ± SEM)")
        plt.legend()
        _save_and_close(f"{dset}_agg_weighted_accuracies.png")
    except Exception as e:
        print(f"Error creating aggregated weighted accuracy plot for {dset}: {e}")
        plt.close()

    # -------------------------------------------------------------- #
    # Plot 4: aggregated confusion matrix, if shapes match
    try:
        if preds_runs and all(arr.shape == preds_runs[0].shape for arr in preds_runs):
            cm = np.zeros((2, 2), dtype=int)
            for gts, preds in zip(gts_runs, preds_runs):
                for t, p in zip(gts, preds):
                    cm[t, p] += 1
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
            plt.xticks([0, 1], ["Pred 0", "Pred 1"])
            plt.yticks([0, 1], ["True 0", "True 1"])
            plt.title(f"{dset} – Confusion Matrix (Aggregated)")
            _save_and_close(f"{dset}_agg_confusion_matrix.png")
    except Exception as e:
        print(f"Error creating aggregated confusion matrix for {dset}: {e}")
        plt.close()

    # -------------------------------------------------------------- #
    # Print last-epoch mean metrics
    final_idx = -1
    print(f"{dset} – Final Validation Metrics (mean ± SEM over {len(acc_runs)} runs):")
    print(f"  Accuracy: {acc_mean[final_idx]:.3f} ± {acc_sem[final_idx]:.3f}")
    print(f"  CWA     : {cwa_mean[final_idx]:.3f} ± {cwa_sem[final_idx]:.3f}")
    print(f"  SWA     : {swa_mean[final_idx]:.3f} ± {swa_sem[final_idx]:.3f}")
    print(f"  CAA     : {caa_mean[final_idx]:.3f} ± {caa_sem[final_idx]:.3f}")

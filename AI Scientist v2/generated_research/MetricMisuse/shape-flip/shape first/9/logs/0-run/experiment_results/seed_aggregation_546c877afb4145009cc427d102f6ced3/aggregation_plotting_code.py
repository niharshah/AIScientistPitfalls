import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -----------------------  Collect experiment paths  -----------------------
experiment_data_path_list = [
    "experiments/2025-08-15_01-36-11_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_8c7fa9d5b2124f70a9d2793bff04a86c_proc_2799575/experiment_data.npy",
    "experiments/2025-08-15_01-36-11_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_2bead55726b241379276a50327ebaf48_proc_2799576/experiment_data.npy",
    "experiments/2025-08-15_01-36-11_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_1905f7ed36fb48b88c5b63789020be52_proc_2799578/experiment_data.npy",
]

# -----------------------  Load all experiment data  -----------------------
all_runs = []
for path in experiment_data_path_list:
    try:
        abs_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), path)
        exp = np.load(abs_path, allow_pickle=True).item()
        if "SPR_BENCH" in exp:
            all_runs.append(exp["SPR_BENCH"])
    except Exception as e:
        print(f"Error loading {path}: {e}")

if len(all_runs) == 0:
    print("No runs could be loaded – nothing to plot.")
else:
    # ----------------------------------------------------------------------
    # -----------  Aggregate per-epoch arrays (truncate to min len) --------
    # ----------------------------------------------------------------------
    try:
        train_losses_runs = [np.asarray(r["losses"]["train"]) for r in all_runs]
        val_losses_runs = [np.asarray(r["losses"]["val"]) for r in all_runs]
        val_swa_runs = [np.asarray(r["metrics"]["val"]) for r in all_runs]

        min_len = min(map(len, train_losses_runs))
        train_mat = np.vstack([arr[:min_len] for arr in train_losses_runs])
        val_mat = np.vstack([arr[:min_len] for arr in val_losses_runs])
        swa_mat = np.vstack([arr[:min_len] for arr in val_swa_runs])
        epochs = np.arange(1, min_len + 1)

        # ------------------  Plot aggregated loss curves  -----------------
        try:
            plt.figure(figsize=(6, 4))
            mean_train = train_mat.mean(axis=0)
            se_train = train_mat.std(axis=0) / np.sqrt(train_mat.shape[0])
            mean_val = val_mat.mean(axis=0)
            se_val = val_mat.std(axis=0) / np.sqrt(val_mat.shape[0])

            plt.plot(epochs, mean_train, label="Train – mean", color="tab:blue")
            plt.fill_between(
                epochs,
                mean_train - se_train,
                mean_train + se_train,
                color="tab:blue",
                alpha=0.3,
                label="Train – SE",
            )

            plt.plot(
                epochs, mean_val, linestyle="--", label="Val – mean", color="tab:orange"
            )
            plt.fill_between(
                epochs,
                mean_val - se_val,
                mean_val + se_val,
                color="tab:orange",
                alpha=0.3,
                label="Val – SE",
            )

            plt.title(
                "SPR_BENCH Aggregated Loss Curves\nMean ± Standard Error across runs"
            )
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, "SPR_BENCH_agg_loss_curves.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating aggregated loss curve plot: {e}")
            plt.close()
        # ------------------  Plot aggregated SWA curves  ------------------
        try:
            plt.figure(figsize=(6, 4))
            mean_swa = swa_mat.mean(axis=0)
            se_swa = swa_mat.std(axis=0) / np.sqrt(swa_mat.shape[0])

            plt.plot(
                epochs, mean_swa, marker="o", color="tab:green", label="Val SWA – mean"
            )
            plt.fill_between(
                epochs,
                mean_swa - se_swa,
                mean_swa + se_swa,
                color="tab:green",
                alpha=0.3,
                label="Val SWA – SE",
            )
            plt.title(
                "SPR_BENCH Validation Shape-Weighted-Accuracy\nMean ± SE across runs"
            )
            plt.xlabel("Epoch")
            plt.ylabel("SWA")
            plt.ylim(0, 1.05)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, "SPR_BENCH_agg_val_SWA.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating aggregated SWA plot: {e}")
            plt.close()
    except Exception as e:
        print(f"Error aggregating per-epoch arrays: {e}")

    # ------------------  Aggregate confusion matrix  ----------------------
    try:
        preds_concat = []
        gts_concat = []
        for run in all_runs:
            if "predictions" in run and "ground_truth" in run:
                preds_concat.extend(run["predictions"])
                gts_concat.extend(run["ground_truth"])
        preds_concat = np.asarray(preds_concat)
        gts_concat = np.asarray(gts_concat)
        if preds_concat.size > 0 and preds_concat.shape == gts_concat.shape:
            classes = sorted(set(gts_concat))
            cm = np.zeros((len(classes), len(classes)), dtype=int)
            for t, p in zip(gts_concat, preds_concat):
                cm[t, p] += 1
            plt.figure(figsize=(4, 4))
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.title("SPR_BENCH Confusion Matrix – Aggregated Test Sets")
            plt.xticks(classes)
            plt.yticks(classes)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            for i in classes:
                for j in classes:
                    plt.text(
                        j,
                        i,
                        cm[i, j],
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=8,
                    )
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, "SPR_BENCH_agg_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated confusion matrix: {e}")
        plt.close()

    # --------------  Print summary of final-epoch Test SWA  ---------------
    try:
        test_swa_vals = [
            r["metrics"]["test"]
            for r in all_runs
            if "metrics" in r and "test" in r["metrics"]
        ]
        if len(test_swa_vals) > 0:
            mean_test = np.mean(test_swa_vals)
            std_test = np.std(test_swa_vals)
            print(
                f"Test SWA over {len(test_swa_vals)} runs: {mean_test:.4f} ± {std_test:.4f}"
            )
    except Exception as e:
        print(f"Error summarizing test metrics: {e}")

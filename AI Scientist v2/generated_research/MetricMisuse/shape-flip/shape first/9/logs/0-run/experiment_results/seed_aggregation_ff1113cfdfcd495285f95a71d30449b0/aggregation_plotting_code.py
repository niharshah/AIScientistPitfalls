import matplotlib.pyplot as plt
import numpy as np
import os
from math import sqrt

# ------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------  Collect experiment data paths -------------
experiment_data_path_list = [
    "experiments/2025-08-15_01-36-11_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_9d67ff68721f448a8fa5d85c5784616c_proc_2801223/experiment_data.npy",
    "experiments/2025-08-15_01-36-11_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_88845e35ca6b4d4b94b878cedd57ed8e_proc_2801222/experiment_data.npy",
    "experiments/2025-08-15_01-36-11_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_19874f1d0aa94b6f935b81211a6525d6_proc_2801224/experiment_data.npy",
]

all_experiment_data = []
try:
    for rel_path in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), rel_path)
        if not os.path.isfile(full_path):
            print(f"File not found: {full_path}")
            continue
        all_experiment_data.append(np.load(full_path, allow_pickle=True).item())
    if not all_experiment_data:
        raise RuntimeError("No experiment files found.")
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# Proceed only if we actually loaded runs
if all_experiment_data:
    # --------------------  Aggregate metrics  -------------------------
    train_losses_list, val_losses_list, val_swa_list = [], [], []
    test_swa_values = []
    all_preds, all_gts = [], []

    for exp in all_experiment_data:
        spr = exp.get("SPR_BENCH", {})
        train_losses_list.append(np.asarray(spr["losses"]["train"]))
        val_losses_list.append(np.asarray(spr["losses"]["val"]))
        val_swa_list.append(np.asarray(spr["metrics"]["val"]))
        test_swa_values.append(spr["metrics"]["test"])
        all_preds.append(np.asarray(spr["predictions"]))
        all_gts.append(np.asarray(spr["ground_truth"]))

    # Ensure equal length across runs by truncation to the shortest run
    min_len = min(len(x) for x in train_losses_list)
    train_mat = np.stack([x[:min_len] for x in train_losses_list])
    val_mat = np.stack([x[:min_len] for x in val_losses_list])
    swa_mat = np.stack([x[:min_len] for x in val_swa_list])
    epochs = np.arange(1, min_len + 1)

    def mean_sem(mat):
        mean = mat.mean(axis=0)
        sem = mat.std(axis=0, ddof=1) / sqrt(mat.shape[0])
        return mean, sem

    train_mean, train_sem = mean_sem(train_mat)
    val_mean, val_sem = mean_sem(val_mat)
    swa_mean, swa_sem = mean_sem(swa_mat)

    # --------------------------  Plot 1  ------------------------------
    try:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, train_mean, label="Train (mean)")
        plt.fill_between(
            epochs, train_mean - train_sem, train_mean + train_sem, alpha=0.3
        )
        plt.plot(epochs, val_mean, "--", label="Validation (mean)")
        plt.fill_between(epochs, val_mean - val_sem, val_mean + val_sem, alpha=0.3)
        plt.title("SPR_BENCH Aggregate Loss Curves\n(mean ± SEM across runs)")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_aggregate_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregate loss plot: {e}")
        plt.close()

    # --------------------------  Plot 2  ------------------------------
    try:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, swa_mean, marker="o", label="Validation SWA (mean)")
        plt.fill_between(
            epochs, swa_mean - swa_sem, swa_mean + swa_sem, alpha=0.3, label="± SEM"
        )
        plt.title("SPR_BENCH Aggregate Validation SWA\n(mean ± SEM across runs)")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.ylim(0, 1.05)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_aggregate_val_SWA.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregate SWA plot: {e}")
        plt.close()

    # ----------------------  Confusion Matrix  -----------------------
    try:
        preds_concat = np.concatenate(all_preds)
        gts_concat = np.concatenate(all_gts)
        classes = sorted(set(gts_concat))
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for t, p in zip(gts_concat, preds_concat):
            cm[t, p] += 1

        plt.figure(figsize=(4, 4))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.title("SPR_BENCH Confusion Matrix - Combined Test Sets")
        plt.xticks(classes)
        plt.yticks(classes)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        for i in range(len(classes)):
            for j in range(len(classes)):
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
        plt.savefig(
            os.path.join(working_dir, "SPR_BENCH_confusion_matrix_combined.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating combined confusion matrix: {e}")
        plt.close()

    # ---------------------  Print summary metric  --------------------
    test_swa_arr = np.asarray(test_swa_values)
    print(
        f"Aggregate Test Shape-Weighted-Accuracy: {test_swa_arr.mean():.4f} ± {test_swa_arr.std(ddof=1):.4f} (n={len(test_swa_arr)})"
    )
else:
    print("No data available to aggregate and plot.")

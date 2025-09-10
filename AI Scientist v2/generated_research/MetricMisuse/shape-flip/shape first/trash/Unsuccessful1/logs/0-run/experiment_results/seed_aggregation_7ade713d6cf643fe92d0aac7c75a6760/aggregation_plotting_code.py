import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------- setup --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Paths supplied by the system
experiment_data_path_list = [
    "experiments/2025-08-15_14-47-52_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_bfe906b1f4cc4d848fb4dc3b581a8360_proc_2922424/experiment_data.npy",
    "experiments/2025-08-15_14-47-52_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_cd1c365e73984fb4b1b912dcd3b1329b_proc_2922423/experiment_data.npy",
    "experiments/2025-08-15_14-47-52_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_a8db10a706eb44b2b2f4648723f241fa_proc_2922421/experiment_data.npy",
]

all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp)
except Exception as e:
    print(f"Error loading experiment data: {e}")


# ------------- helper to compute mean & sem -------------
def mean_sem(arr, axis=0):
    arr = np.asarray(arr)
    mean = np.mean(arr, axis=axis)
    sem = np.std(arr, axis=axis, ddof=1) / np.sqrt(arr.shape[axis])
    return mean, sem


# ------------- aggregate over runs for each dataset -----
datasets = set()
for exp in all_experiment_data:
    datasets.update(exp.keys())

for dname in datasets:
    # collect per-run structures
    epochs_list = []
    tr_loss_list, val_loss_list = [], []
    swa_val_list, swa_test_list = [], []
    cm_agg = None
    # ---------- gather ----------
    for exp in all_experiment_data:
        if dname not in exp:
            continue
        spr = exp[dname]
        epochs_list.append(np.array(spr["epochs"]))
        tr_loss_list.append(np.array(spr["losses"]["train"]))
        val_loss_list.append(np.array(spr["losses"]["val"]))
        swa_val_list.append(np.array(spr["metrics"]["val"]))
        swa_test_list.append(float(spr["metrics"]["test"]))
        gts = np.array(spr["ground_truth"])
        preds = np.array(spr["predictions"])
        if gts.size and preds.size:
            cm = np.zeros((2, 2), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            cm_agg = cm if cm_agg is None else cm_agg + cm

    if not tr_loss_list:  # nothing collected
        continue

    # trim to common length
    min_len = min(map(len, tr_loss_list))
    epochs = epochs_list[0][:min_len]
    tr_loss_mat = np.vstack([arr[:min_len] for arr in tr_loss_list])
    val_loss_mat = np.vstack([arr[:min_len] for arr in val_loss_list])
    swa_val_mat = np.vstack([arr[:min_len] for arr in swa_val_list])

    # ------------------ Plot 1: aggregated loss -----------
    try:
        plt.figure()
        m_tr, se_tr = mean_sem(tr_loss_mat)
        m_val, se_val = mean_sem(val_loss_mat)
        plt.plot(epochs, m_tr, "--", label="train mean")
        plt.fill_between(
            epochs, m_tr - se_tr, m_tr + se_tr, alpha=0.3, label="train ±SEM"
        )
        plt.plot(epochs, m_val, "-", label="validation mean")
        plt.fill_between(
            epochs, m_val - se_val, m_val + se_val, alpha=0.3, label="val ±SEM"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dname}: Aggregated Training vs Validation Loss (mean ± SEM)")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname}_loss_curves_mean_sem.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dname}: {e}")
        plt.close()

    # ------------------ Plot 2: aggregated val SWA --------
    try:
        plt.figure()
        m_swa, se_swa = mean_sem(swa_val_mat)
        plt.plot(epochs, m_swa, marker="o", color="green", label="val SWA mean")
        plt.fill_between(
            epochs,
            m_swa - se_swa,
            m_swa + se_swa,
            alpha=0.3,
            color="green",
            label="±SEM",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy (SWA)")
        plt.title(f"{dname}: Validation SWA Across Epochs (mean ± SEM)")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname}_val_SWA_curve_mean_sem.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating SWA curve plot for {dname}: {e}")
        plt.close()

    # ------------------ Plot 3: aggregated confusion ------
    try:
        if cm_agg is not None:
            plt.figure()
            plt.imshow(cm_agg, cmap="Blues")
            plt.colorbar()
            for i in range(2):
                for j in range(2):
                    plt.text(
                        j, i, cm_agg[i, j], ha="center", va="center", color="black"
                    )
            plt.xticks([0, 1], ["invalid", "valid"])
            plt.yticks([0, 1], ["invalid", "valid"])
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title(
                f"{dname}: Aggregated Test Confusion Matrix\nLeft: Ground Truth, Right: Predicted"
            )
            fname = os.path.join(
                working_dir, f"{dname}_test_confusion_matrix_aggregated.png"
            )
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot for {dname}: {e}")
        plt.close()

    # ------------------ Plot 4: final test SWA bar --------
    try:
        swa_test_arr = np.array(swa_test_list)
        mean_test, sem_test = np.mean(swa_test_arr), np.std(
            swa_test_arr, ddof=1
        ) / np.sqrt(len(swa_test_arr))
        plt.figure()
        plt.bar(
            ["SWA"],
            [mean_test],
            yerr=[sem_test],
            color="steelblue",
            capsize=8,
            label="mean ± SEM",
        )
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title(f"{dname}: Final Test SWA (mean ± SEM across runs)")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname}_test_SWA_bar_mean_sem.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated test SWA bar plot for {dname}: {e}")
        plt.close()

    # ------------------ print metric ----------------------
    mean_test, sem_test = np.mean(swa_test_list), np.std(
        swa_test_list, ddof=1
    ) / np.sqrt(len(swa_test_list))
    print(
        f"{dname} - Final test SWA: {mean_test:.4f} ± {sem_test:.4f} (mean ± SEM across {len(swa_test_list)} runs)"
    )

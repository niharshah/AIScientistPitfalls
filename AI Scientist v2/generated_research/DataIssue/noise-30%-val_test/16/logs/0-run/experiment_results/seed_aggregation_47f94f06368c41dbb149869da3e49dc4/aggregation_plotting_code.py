import matplotlib.pyplot as plt
import numpy as np
import os
import math

# ------------------------------------------------------------------ setup ---
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- add here every experiment_data.npy path that was provided -------
experiment_data_path_list = [
    "experiments/2025-08-17_18-47-55_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_3b0e5fc9f3d0490db9fe94540ade17f9_proc_3327621/experiment_data.npy",
    "experiments/2025-08-17_18-47-55_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_0756859685e545839334dda4058b0025_proc_3327623/experiment_data.npy",
    "experiments/2025-08-17_18-47-55_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_efa10bfdb5d14392acfe951e5d3c67d4_proc_3327624/experiment_data.npy",
]

# ----------------------------------------------------------------- loading --
all_experiment_data = []
for p in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp_d = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_d)
    except Exception as e:
        print(f"Error loading {p}: {e}")

if len(all_experiment_data) == 0:
    raise SystemExit("No experiment data could be loaded.")

# -------------------------------------------------------------- aggregation --
# discover shared dataset names
dataset_names = set.intersection(*[set(ed.keys()) for ed in all_experiment_data])

for dset in dataset_names:
    # Gather per-run tensors
    loss_train_runs, loss_val_runs = [], []
    mcc_train_runs, mcc_val_runs = [], []
    epochs_ref = None
    cm_counts = []  # TP,TN,FP,FN per run

    for run_idx, ed in enumerate(all_experiment_data):
        try:
            data = ed[dset]
            epochs = np.asarray(data["epochs"])
            if epochs_ref is None:
                epochs_ref = epochs
            # only keep runs that have identical epoch length
            if len(epochs) != len(epochs_ref):
                print(f"Skipping run {run_idx} for {dset}: mismatched epochs.")
                continue
            loss_train_runs.append(np.asarray(data["losses"]["train"]))
            loss_val_runs.append(np.asarray(data["losses"]["val"]))
            mcc_train_runs.append(np.asarray(data["metrics"]["train_MCC"]))
            mcc_val_runs.append(np.asarray(data["metrics"]["val_MCC"]))

            # confusion matrix
            preds = np.asarray(data["predictions"])
            gts = np.asarray(data["ground_truth"])
            tp = int(((preds == 1) & (gts == 1)).sum())
            tn = int(((preds == 0) & (gts == 0)).sum())
            fp = int(((preds == 1) & (gts == 0)).sum())
            fn = int(((preds == 0) & (gts == 1)).sum())
            cm_counts.append([tp, tn, fp, fn])
        except Exception as e:
            print(f"Run {run_idx} missing keys for {dset}: {e}")

    # convert to arrays
    def stack_and_stats(lst):
        arr = np.stack(lst, axis=0)  # shape (runs, epochs)
        mean = arr.mean(axis=0)
        sem = (
            arr.std(axis=0, ddof=1) / math.sqrt(arr.shape[0])
            if arr.shape[0] > 1
            else np.zeros_like(mean)
        )
        return mean, sem

    if len(loss_train_runs) == 0:
        print(f"No complete runs for {dset}, skipping plotting.")
        continue

    m_train, s_train = stack_and_stats(loss_train_runs)
    m_val, s_val = stack_and_stats(loss_val_runs)
    m_mcc_tr, s_mcc_tr = stack_and_stats(mcc_train_runs)
    m_mcc_val, s_mcc_val = stack_and_stats(mcc_val_runs)

    # ----------------------------------------------------------- plot losses
    try:
        plt.figure()
        plt.plot(epochs_ref, m_train, label="Train Loss mean")
        plt.fill_between(
            epochs_ref,
            m_train - s_train,
            m_train + s_train,
            alpha=0.25,
            label="Train Loss ± SEM",
        )
        plt.plot(epochs_ref, m_val, label="Val Loss mean")
        plt.fill_between(
            epochs_ref, m_val - s_val, m_val + s_val, alpha=0.25, label="Val Loss ± SEM"
        )
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title(
            f"{dset} – Aggregated Loss Curve\n(Mean ± SEM over {len(loss_train_runs)} runs)"
        )
        plt.legend()
        save_path = os.path.join(working_dir, f"{dset}_agg_loss_curve.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss for {dset}: {e}")
        plt.close()

    # ----------------------------------------------------------- plot MCC
    try:
        plt.figure()
        plt.plot(epochs_ref, m_mcc_tr, label="Train MCC mean")
        plt.fill_between(
            epochs_ref,
            m_mcc_tr - s_mcc_tr,
            m_mcc_tr + s_mcc_tr,
            alpha=0.25,
            label="Train MCC ± SEM",
        )
        plt.plot(epochs_ref, m_mcc_val, label="Val MCC mean")
        plt.fill_between(
            epochs_ref,
            m_mcc_val - s_mcc_val,
            m_mcc_val + s_mcc_val,
            alpha=0.25,
            label="Val MCC ± SEM",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Matthews Corr. Coef.")
        plt.title(
            f"{dset} – Aggregated MCC Curve\n(Mean ± SEM over {len(mcc_train_runs)} runs)"
        )
        plt.legend()
        save_path = os.path.join(working_dir, f"{dset}_agg_MCC_curve.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated MCC for {dset}: {e}")
        plt.close()

    # --------------------------------------- aggregated confusion-matrix bar
    try:
        cm_arr = np.asarray(cm_counts)  # shape (runs, 4)
        cm_mean = cm_arr.mean(axis=0)
        cm_sem = (
            cm_arr.std(axis=0, ddof=1) / math.sqrt(cm_arr.shape[0])
            if cm_arr.shape[0] > 1
            else np.zeros_like(cm_mean)
        )
        labels = ["TP", "TN", "FP", "FN"]
        plt.figure()
        plt.bar(
            labels,
            cm_mean,
            yerr=cm_sem,
            capsize=5,
            color=["green", "blue", "orange", "red"],
        )
        for idx, val in enumerate(cm_mean):
            plt.text(idx, val + max(cm_sem[idx], 1e-3), f"{val:.1f}", ha="center")
        plt.ylabel("Count (mean)")
        plt.title(
            f"{dset} – Aggregated Confusion Matrix\n(Mean ± SEM over {cm_arr.shape[0]} runs)"
        )
        save_path = os.path.join(working_dir, f"{dset}_agg_confusion_matrix.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated confusion matrix for {dset}: {e}")
        plt.close()

    # ----------------------------------------- console summary (final epoch)
    final_val_mcc_mean = m_mcc_val[-1]
    final_val_mcc_sem = s_mcc_val[-1]
    print(
        f"{dset}: final-epoch Val MCC = {final_val_mcc_mean:.4f} ± {final_val_mcc_sem:.4f} (mean ± SEM, N={len(mcc_val_runs)})"
    )

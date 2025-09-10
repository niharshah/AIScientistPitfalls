import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------- paths ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load experiment data ----------------
experiment_data_path_list = [
    "experiments/2025-08-14_15-47-52_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_5c384894bdaf4f0d90b7546f36df57e5_proc_2676157/experiment_data.npy",
    "experiments/2025-08-14_15-47-52_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_3b84f54b6a494119bcd445d07871d050_proc_2676156/experiment_data.npy",
    "experiments/2025-08-14_15-47-52_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_31dee2ef2a7443d49440d2db9c06cc65_proc_2676159/experiment_data.npy",
]

all_runs = []
for rel_path in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), rel_path)
        data = np.load(full_path, allow_pickle=True).item()
        run_rec = data["epochs_tuning"]["SPR_BENCH"]
        all_runs.append(run_rec)
    except Exception as e:
        print(f"Error loading {rel_path}: {e}")

if len(all_runs) == 0:
    print("No runs loaded – nothing to plot.")
else:
    # ----------- aggregate per-epoch arrays -----------
    train_loss_list, val_loss_list = [], []
    train_rcwa_list, val_rcwa_list = [], []
    test_acc_list = []

    for rec in all_runs:
        tl = np.asarray(rec["losses"]["train"])
        vl = np.asarray(rec["losses"]["val"])
        tr = np.asarray(rec["metrics"]["train_rcwa"])
        vr = np.asarray(rec["metrics"]["val_rcwa"])
        preds = np.asarray(rec["predictions"])
        gts = np.asarray(rec["ground_truth"])
        acc = (preds == gts).mean() if len(preds) else np.nan

        train_loss_list.append(tl)
        val_loss_list.append(vl)
        train_rcwa_list.append(tr)
        val_rcwa_list.append(vr)
        test_acc_list.append(acc)

    # Trim to common length (shortest run)
    min_len = min(map(len, train_loss_list))
    train_loss_mat = np.stack([x[:min_len] for x in train_loss_list], axis=0)
    val_loss_mat = np.stack([x[:min_len] for x in val_loss_list], axis=0)
    train_rcwa_mat = np.stack([x[:min_len] for x in train_rcwa_list], axis=0)
    val_rcwa_mat = np.stack([x[:min_len] for x in val_rcwa_list], axis=0)
    epochs = np.arange(1, min_len + 1)

    # Mean and stderr
    def mean_stderr(mat):
        mean = np.nanmean(mat, axis=0)
        stderr = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(mat.shape[0])
        return mean, stderr

    tr_loss_mean, tr_loss_se = mean_stderr(train_loss_mat)
    va_loss_mean, va_loss_se = mean_stderr(val_loss_mat)
    tr_rcwa_mean, tr_rcwa_se = mean_stderr(train_rcwa_mat)
    va_rcwa_mean, va_rcwa_se = mean_stderr(val_rcwa_mat)

    # ----------- plot 1: aggregated loss curves -----------
    try:
        plt.figure()
        plt.plot(epochs, tr_loss_mean, label="Train (mean)")
        plt.fill_between(
            epochs,
            tr_loss_mean - tr_loss_se,
            tr_loss_mean + tr_loss_se,
            alpha=0.3,
            label="Train ± SE",
        )
        plt.plot(epochs, va_loss_mean, label="Validation (mean)")
        plt.fill_between(
            epochs,
            va_loss_mean - va_loss_se,
            va_loss_mean + va_loss_se,
            alpha=0.3,
            label="Val ± SE",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(
            "SPR_BENCH: Aggregated Train vs Validation Loss\n(Mean ± Standard Error)"
        )
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_agg_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot: {e}")
        plt.close()

    # ----------- plot 2: aggregated RCWA curves -----------
    try:
        plt.figure()
        plt.plot(epochs, tr_rcwa_mean, label="Train RCWA (mean)")
        plt.fill_between(
            epochs,
            tr_rcwa_mean - tr_rcwa_se,
            tr_rcwa_mean + tr_rcwa_se,
            alpha=0.3,
            label="Train ± SE",
        )
        plt.plot(epochs, va_rcwa_mean, label="Validation RCWA (mean)")
        plt.fill_between(
            epochs,
            va_rcwa_mean - va_rcwa_se,
            va_rcwa_mean + va_rcwa_se,
            alpha=0.3,
            label="Val ± SE",
        )
        plt.xlabel("Epoch")
        plt.ylabel("RCWA")
        plt.title(
            "SPR_BENCH: Aggregated Train vs Validation RCWA\n(Mean ± Standard Error)"
        )
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_agg_rcwa_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated RCWA plot: {e}")
        plt.close()

    # ----------- plot 3: aggregated test accuracy -----------
    try:
        test_acc_arr = np.asarray(test_acc_list, dtype=float)
        mean_acc = np.nanmean(test_acc_arr)
        se_acc = np.nanstd(test_acc_arr, ddof=1) / np.sqrt(len(test_acc_arr))

        plt.figure()
        plt.bar(["Accuracy"], [mean_acc], yerr=[se_acc], capsize=10)
        plt.ylim(0, 1)
        plt.title("SPR_BENCH: Aggregated Test Accuracy\n(Mean ± Standard Error)")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_agg_test_accuracy.png"))
        plt.close()

        print(f"Aggregated Test Accuracy: mean={mean_acc:.4f}, SE={se_acc:.4f}")
    except Exception as e:
        print(f"Error creating aggregated accuracy plot: {e}")
        plt.close()

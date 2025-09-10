import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------- paths & loading --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --- list every experiment_data.npy given by the instructions ---
experiment_data_path_list = [
    "experiments/2025-08-14_21-45-52_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_b5b16e62600c42fab915e7701e5dc8b8_proc_2755151/experiment_data.npy",
    "experiments/2025-08-14_21-45-52_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_bd43d1a17f674e64a06a197ba24baace_proc_2755149/experiment_data.npy",
    "experiments/2025-08-14_21-45-52_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_5343b363d71c4eef82e5909b046c9be2_proc_2755148/experiment_data.npy",
]

all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        if os.path.isfile(full_path):
            all_experiment_data.append(np.load(full_path, allow_pickle=True).item())
        else:
            print(f"File not found: {full_path}")
except Exception as e:
    print(f"Error loading experiment data: {e}")

# -------------------- collect per-run arrays --------------------
train_curves, val_acc_curves, val_ura_curves = [], [], []
test_accs, test_uras = [], []

for exp in all_experiment_data:
    for run_key, run_data in exp.get("EPOCHS", {}).items():
        m = run_data.get("metrics", {})
        if {"train_acc", "val_acc", "val_ura"} <= m.keys():
            train_curves.append(np.asarray(m["train_acc"], dtype=float))
            val_acc_curves.append(np.asarray(m["val_acc"], dtype=float))
            val_ura_curves.append(np.asarray(m["val_ura"], dtype=float))
        # final-epoch / test metrics (may be missing)
        test_accs.append(run_data.get("test_acc", np.nan))
        test_uras.append(run_data.get("test_ura", np.nan))

num_runs = len(train_curves)

# nothing to do if no runs found
if num_runs == 0:
    print("No runs discovered – no plots will be produced.")
else:
    # -------------------- helper: pad to same length --------------------
    def pad_to_max(arr_list):
        max_len = max(len(a) for a in arr_list)
        padded = np.full((len(arr_list), max_len), np.nan)
        for i, a in enumerate(arr_list):
            padded[i, : len(a)] = a
        return padded

    train_mat = pad_to_max(train_curves)
    val_acc_mat = pad_to_max(val_acc_curves)
    val_ura_mat = pad_to_max(val_ura_curves)
    epochs = np.arange(1, train_mat.shape[1] + 1)

    # -------------------- figure 1: aggregated epoch curves --------------------
    try:
        plt.figure(figsize=(7, 4))
        for mat, label, color in [
            (train_mat, "Train Acc", "tab:blue"),
            (val_acc_mat, "Val Acc", "tab:orange"),
            (val_ura_mat, "Val URA", "tab:green"),
        ]:
            mean = np.nanmean(mat, axis=0)
            sem = np.nanstd(mat, axis=0) / np.sqrt(np.sum(~np.isnan(mat), axis=0))
            plt.plot(epochs, mean, label=f"{label} (mean)", color=color)
            plt.fill_between(
                epochs,
                mean - sem,
                mean + sem,
                alpha=0.25,
                color=color,
                label=f"{label} ± SEM",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.title(f"SPR_BENCH – Aggregated Accuracy/URA Curves (N={num_runs})")
        plt.legend(ncol=2, fontsize=8)
        fname = os.path.join(working_dir, "SPR_BENCH_aggregated_acc_curves.png")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
        print(f"Saved: {fname}")
    except Exception as e:
        print(f"Error creating aggregated curves: {e}")
        plt.close()

    # -------------------- figure 2: aggregated test metrics --------------------
    try:
        metrics = np.array([test_accs, test_uras], dtype=float)  # shape (2, n)
        mean_vals = np.nanmean(metrics, axis=1)
        sem_vals = np.nanstd(metrics, axis=1) / np.sqrt(
            np.sum(~np.isnan(metrics), axis=1)
        )

        x = np.arange(2)
        plt.figure(figsize=(5, 4))
        plt.bar(
            x,
            mean_vals,
            yerr=sem_vals,
            capsize=5,
            color=["tab:purple", "tab:red"],
            tick_label=["Test Acc", "Test URA"],
        )
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title(f"SPR_BENCH – Aggregated Test Metrics (N={num_runs})")
        for i, v in enumerate(mean_vals):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        fname = os.path.join(working_dir, "SPR_BENCH_aggregated_test_metrics.png")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
        print(f"Saved: {fname}")
    except Exception as e:
        print(f"Error creating aggregated bar plot: {e}")
        plt.close()

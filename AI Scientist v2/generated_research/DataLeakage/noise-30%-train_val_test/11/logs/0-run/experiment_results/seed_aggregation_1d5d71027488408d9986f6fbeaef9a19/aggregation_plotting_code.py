import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------------
# IO set-up
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
# Collect experiment files
experiment_data_path_list = [
    "experiments/2025-08-17_23-44-10_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_40b6080bd9f845e2af6638fc580d2577_proc_3462733/experiment_data.npy",
    "experiments/2025-08-17_23-44-10_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_a2e2cc5d474548998c61a0880dd81265_proc_3462734/experiment_data.npy",
    "experiments/2025-08-17_23-44-10_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_834353a4c8df41f8bc31dfaa8be751b4_proc_3462732/experiment_data.npy",
]

all_experiment_data = []
for path in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), path)
        exp = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp)
    except Exception as e:
        print(f"Error loading {path}: {e}")

if len(all_experiment_data) == 0:
    print("No experiment files could be loaded – nothing to plot.")
    exit()

# ---------------------------------------------------------------------
# Aggregate over runs (assuming the same dataset keys in each run)
dataset_names = all_experiment_data[0].keys()

for dname in dataset_names:
    # Collect per-run arrays -------------------------------------------------
    train_loss_runs, val_loss_runs = [], []
    train_f1_runs, val_f1_runs = [], []
    test_f1_runs = []  # shape = [n_runs, n_schedules]
    epochs_ref = None
    for run in all_experiment_data:
        try:
            d = run[dname]
            if epochs_ref is None:
                epochs_ref = np.asarray(d["epochs"])
            train_loss_runs.append(np.asarray(d["losses"]["train"]))
            val_loss_runs.append(np.asarray(d["losses"]["val"]))
            train_f1_runs.append(np.asarray(d["metrics"]["train_macro_f1"]))
            val_f1_runs.append(np.asarray(d["metrics"]["val_macro_f1"]))
            test_f1_runs.append(np.asarray(d["metrics"]["test_macro_f1"]))
        except KeyError as e:
            print(f"Missing key {e} in run – skipping.")
    if len(train_loss_runs) == 0:
        print(f"No usable runs found for dataset {dname}")
        continue

    # Convert to arrays: shape = [runs, epochs]
    train_loss_mat = np.stack(train_loss_runs)
    val_loss_mat = np.stack(val_loss_runs)
    train_f1_mat = np.stack(train_f1_runs)
    val_f1_mat = np.stack(val_f1_runs)
    test_f1_mat = np.stack(test_f1_runs)  # [runs, n_schedules]

    n_runs = train_loss_mat.shape[0]

    # Mean and SEM -----------------------------------------------------------
    def mean_sem(mat, axis=0):
        mean = mat.mean(axis=axis)
        sem = mat.std(axis=axis, ddof=1) / np.sqrt(mat.shape[axis])
        return mean, sem

    tr_loss_mean, tr_loss_sem = mean_sem(train_loss_mat)
    va_loss_mean, va_loss_sem = mean_sem(val_loss_mat)
    tr_f1_mean, tr_f1_sem = mean_sem(train_f1_mat)
    va_f1_mean, va_f1_sem = mean_sem(val_f1_mat)
    test_f1_mean = test_f1_mat.mean(axis=0)
    test_f1_sem = test_f1_mat.std(axis=0, ddof=1) / np.sqrt(n_runs)

    # ---------------------------------------------------------------------
    # 1) Loss curves with SEM shading
    try:
        plt.figure()
        plt.plot(epochs_ref, tr_loss_mean, label="Train Loss (mean)", color="tab:blue")
        plt.fill_between(
            epochs_ref,
            tr_loss_mean - tr_loss_sem,
            tr_loss_mean + tr_loss_sem,
            color="tab:blue",
            alpha=0.3,
            label="Train ± SEM",
        )
        plt.plot(epochs_ref, va_loss_mean, label="Val Loss (mean)", color="tab:orange")
        plt.fill_between(
            epochs_ref,
            va_loss_mean - va_loss_sem,
            va_loss_mean + va_loss_sem,
            color="tab:orange",
            alpha=0.3,
            label="Val ± SEM",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dname} Loss Curves (Mean ± SEM)\nAggregated over {n_runs} runs")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname}_agg_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dname}: {e}")
        plt.close()

    # ---------------------------------------------------------------------
    # 2) Macro-F1 curves with SEM shading
    try:
        plt.figure()
        plt.plot(
            epochs_ref, tr_f1_mean, label="Train Macro-F1 (mean)", color="tab:green"
        )
        plt.fill_between(
            epochs_ref,
            tr_f1_mean - tr_f1_sem,
            tr_f1_mean + tr_f1_sem,
            color="tab:green",
            alpha=0.3,
            label="Train ± SEM",
        )
        plt.plot(epochs_ref, va_f1_mean, label="Val Macro-F1 (mean)", color="tab:red")
        plt.fill_between(
            epochs_ref,
            va_f1_mean - va_f1_sem,
            va_f1_mean + va_f1_sem,
            color="tab:red",
            alpha=0.3,
            label="Val ± SEM",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title(
            f"{dname} Macro-F1 Curves (Mean ± SEM)\nAggregated over {n_runs} runs"
        )
        plt.legend()
        fname = os.path.join(working_dir, f"{dname}_agg_f1_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated F1 plot for {dname}: {e}")
        plt.close()

    # ---------------------------------------------------------------------
    # 3) Test performance bar chart with error bars
    try:
        plt.figure()
        schedules = ["5 epochs", "10 epochs", "15 epochs"]
        x = np.arange(len(schedules))
        plt.bar(
            x,
            test_f1_mean,
            yerr=test_f1_sem,
            capsize=5,
            color="skyblue",
            alpha=0.8,
            label="Mean ± SEM",
        )
        plt.xticks(x, schedules)
        plt.ylim(0, 1)
        for i, v in enumerate(test_f1_mean):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
        plt.ylabel("Macro-F1")
        plt.title(f"{dname} Test Macro-F1\nAggregated over {n_runs} runs")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname}_agg_test_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated test bar for {dname}: {e}")
        plt.close()

    # ---------------------------------------------------------------------
    # Print summary statistics
    print(f"{dname}:")
    print(
        "  Mean Test Macro-F1 ± SEM:",
        {
            s: f"{m:.3f}±{e:.3f}"
            for s, m, e in zip(["5e", "10e", "15e"], test_f1_mean, test_f1_sem)
        },
    )

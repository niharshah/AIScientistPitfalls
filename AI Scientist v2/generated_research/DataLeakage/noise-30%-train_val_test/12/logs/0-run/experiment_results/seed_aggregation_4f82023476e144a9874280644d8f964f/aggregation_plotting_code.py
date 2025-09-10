import matplotlib.pyplot as plt
import numpy as np
import os
from math import sqrt

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# Load every experiment_data.npy that was listed in the instructions
# ------------------------------------------------------------------
try:
    experiment_data_path_list = [
        "experiments/2025-08-17_23-44-14_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_3fb72856d0ef4cf499f47544df4cf1c7_proc_3458402/experiment_data.npy",
        "experiments/2025-08-17_23-44-14_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_03523b99bca44b79a63d807c9573d529_proc_3458404/experiment_data.npy",
        "experiments/2025-08-17_23-44-14_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_fda6956bb09d4da5b6e1132ec9817237_proc_3458408/experiment_data.npy",
    ]
    all_experiment_data = []
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp_dict = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_dict)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# ---------------------------------------------------------------
# Aggregate curves (mean and SEM) and create the requested plots
# ---------------------------------------------------------------
if all_experiment_data:
    # assume every run contains exactly one dataset key and that
    # they are identical across runs
    dataset_name = list(all_experiment_data[0].keys())[0]

    # containers for per-run arrays
    train_loss_runs, val_loss_runs = [], []
    train_f1_runs, val_f1_runs = [], []
    epochs = None

    for run in all_experiment_data:
        d = run[dataset_name]
        if epochs is None:
            epochs = np.asarray(d["epochs"])
        train_loss_runs.append(np.asarray(d["losses"]["train"]))
        val_loss_runs.append(np.asarray(d["losses"]["val"]))
        train_f1_runs.append(np.asarray(d["metrics"]["train_f1"]))
        val_f1_runs.append(np.asarray(d["metrics"]["val_f1"]))

    # stack and compute mean / SEM
    train_loss_stack = np.vstack(train_loss_runs)
    val_loss_stack = np.vstack(val_loss_runs)
    train_f1_stack = np.vstack(train_f1_runs)
    val_f1_stack = np.vstack(val_f1_runs)

    n_runs = train_loss_stack.shape[0]

    def mean_sem(stack):
        mean = stack.mean(axis=0)
        sem = stack.std(axis=0, ddof=1) / sqrt(n_runs)
        return mean, sem

    train_loss_mean, train_loss_sem = mean_sem(train_loss_stack)
    val_loss_mean, val_loss_sem = mean_sem(val_loss_stack)
    train_f1_mean, train_f1_sem = mean_sem(train_f1_stack)
    val_f1_mean, val_f1_sem = mean_sem(val_f1_stack)

    # ----------------------- PLOT A: LOSS w/ SEM -----------------------
    try:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, train_loss_mean, label="Train Loss (mean)")
        plt.fill_between(
            epochs,
            train_loss_mean - train_loss_sem,
            train_loss_mean + train_loss_sem,
            alpha=0.3,
        )
        plt.plot(epochs, val_loss_mean, label="Val Loss (mean)")
        plt.fill_between(
            epochs,
            val_loss_mean - val_loss_sem,
            val_loss_mean + val_loss_sem,
            alpha=0.3,
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dataset_name} Loss Curves\nMean ± SEM across {n_runs} runs")
        plt.legend()
        plt.tight_layout()
        fname = f"{dataset_name}_agg_loss_curve.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss curve: {e}")
        plt.close()

    # ----------------------- PLOT B: F1 w/ SEM -----------------------
    try:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, train_f1_mean, label="Train Macro-F1 (mean)")
        plt.fill_between(
            epochs,
            train_f1_mean - train_f1_sem,
            train_f1_mean + train_f1_sem,
            alpha=0.3,
        )
        plt.plot(epochs, val_f1_mean, label="Val Macro-F1 (mean)")
        plt.fill_between(
            epochs, val_f1_mean - val_f1_sem, val_f1_mean + val_f1_sem, alpha=0.3
        )
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.title(f"{dataset_name} Macro-F1 Curves\nMean ± SEM across {n_runs} runs")
        plt.legend()
        plt.tight_layout()
        fname = f"{dataset_name}_agg_f1_curve.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated F1 curve: {e}")
        plt.close()

    # ----------------------- PLOT C: Final Val F1 Distribution -----------------------
    try:
        final_val_f1 = val_f1_stack[:, -1]
        mean_final = final_val_f1.mean()
        sem_final = final_val_f1.std(ddof=1) / sqrt(n_runs)

        plt.figure(figsize=(5, 4))
        x = np.arange(n_runs)
        plt.scatter(x, final_val_f1, label="Individual runs")
        plt.hlines(
            mean_final,
            -0.5,
            n_runs - 0.5,
            colors="red",
            label=f"Mean = {mean_final:.3f}",
        )
        plt.fill_between(
            [-0.5, n_runs - 0.5],
            mean_final - sem_final,
            mean_final + sem_final,
            color="red",
            alpha=0.2,
            label="Mean ± SEM",
        )
        plt.xticks(x)
        plt.ylabel("Final Val Macro-F1")
        plt.title(f"{dataset_name} Final Val Macro-F1 across runs")
        plt.legend()
        plt.tight_layout()
        fname = f"{dataset_name}_final_val_f1_scatter.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating final-F1 scatter: {e}")
        plt.close()

    # ----------------------- PRINT METRIC -----------------------
    print(f"Mean final validation Macro-F1 ± SEM: {mean_final:.4f} ± {sem_final:.4f}")
else:
    print("No experiment data could be aggregated.")

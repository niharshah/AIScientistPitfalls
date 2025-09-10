import matplotlib.pyplot as plt
import numpy as np
import os

# -----------------------------------------------------------
# Paths and data loading
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# List of experiment_data.npy files provided by the user
experiment_data_path_list = [
    "experiments/2025-08-14_19-19-19_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_e4dbffa4bca24f36b6766d9b99d46ed8_proc_2715930/experiment_data.npy",
    "experiments/2025-08-14_19-19-19_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_db1513de4f99486e8d358255eec02a94_proc_2715929/experiment_data.npy",
    "experiments/2025-08-14_19-19-19_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_a1324be6ff2649c4ae388e08f238fb22_proc_2715931/experiment_data.npy",
]

all_experiment_data = []
for exp_path in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), exp_path)
        exp_dict = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_dict)
    except Exception as e:
        print(f"Error loading {exp_path}: {e}")

if not all_experiment_data:
    print("No experiment data could be loaded.  Exiting.")
    quit()


# -----------------------------------------------------------
# Helper to compute mean & sem safely
def mean_sem(stack):
    mean = np.mean(stack, axis=0)
    sem = (
        np.std(stack, axis=0, ddof=1) / np.sqrt(stack.shape[0])
        if stack.shape[0] > 1
        else np.zeros_like(mean)
    )
    return mean, sem


# -----------------------------------------------------------
# Aggregate plots for every dataset encountered
datasets = set()
for exp in all_experiment_data:
    datasets.update(exp.keys())

for dset in datasets:
    # ------------------------------------------------------------------
    # Aggregate Loss Curves (Train / Val)
    try:
        train_losses, val_losses = [], []
        for exp in all_experiment_data:
            if dset in exp and "losses" in exp[dset]:
                if "train" in exp[dset]["losses"]:
                    train_losses.append(np.array(exp[dset]["losses"]["train"]))
                if "val" in exp[dset]["losses"]:
                    val_losses.append(np.array(exp[dset]["losses"]["val"]))

        if train_losses and val_losses:
            # assume epochs (column 0) are identical across runs, take from first
            epochs = train_losses[0][:, 0]
            train_stack = np.stack([tl[:, 1] for tl in train_losses], axis=0)
            val_stack = np.stack([vl[:, 1] for vl in val_losses], axis=0)

            train_mean, train_sem = mean_sem(train_stack)
            val_mean, val_sem = mean_sem(val_stack)

            plt.figure()
            plt.plot(epochs, train_mean, label="Train Mean", color="tab:blue")
            plt.fill_between(
                epochs,
                train_mean - train_sem,
                train_mean + train_sem,
                color="tab:blue",
                alpha=0.3,
                label="Train ± SEM",
            )
            plt.plot(epochs, val_mean, label="Val Mean", color="tab:orange")
            plt.fill_between(
                epochs,
                val_mean - val_sem,
                val_mean + val_sem,
                color="tab:orange",
                alpha=0.3,
                label="Val ± SEM",
            )

            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(
                f"{dset} Loss Curves with Standard Error\nLeft: Train, Right: Validation"
            )
            plt.legend()
            fname = os.path.join(working_dir, f"{dset}_loss_curve_agg.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss curve for {dset}: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # Aggregate Metric Curves (HWA or generic 'metrics')
    try:
        train_metrics, val_metrics = [], []
        for exp in all_experiment_data:
            if dset in exp and "metrics" in exp[dset]:
                if "train" in exp[dset]["metrics"]:
                    train_metrics.append(np.array(exp[dset]["metrics"]["train"]))
                if "val" in exp[dset]["metrics"]:
                    val_metrics.append(np.array(exp[dset]["metrics"]["val"]))

        if train_metrics and val_metrics:
            epochs = train_metrics[0][:, 0]
            train_stack = np.stack([tm[:, 1] for tm in train_metrics], axis=0)
            val_stack = np.stack([vm[:, 1] for vm in val_metrics], axis=0)

            train_mean, train_sem = mean_sem(train_stack)
            val_mean, val_sem = mean_sem(val_stack)

            plt.figure()
            plt.plot(epochs, train_mean, label="Train Mean", color="tab:green")
            plt.fill_between(
                epochs,
                train_mean - train_sem,
                train_mean + train_sem,
                color="tab:green",
                alpha=0.3,
                label="Train ± SEM",
            )
            plt.plot(epochs, val_mean, label="Val Mean", color="tab:red")
            plt.fill_between(
                epochs,
                val_mean - val_sem,
                val_mean + val_sem,
                color="tab:red",
                alpha=0.3,
                label="Val ± SEM",
            )

            plt.xlabel("Epoch")
            plt.ylabel("Harmonic-Weighted Accuracy")
            plt.title(
                f"{dset} HWA Curves with Standard Error\nLeft: Train, Right: Validation"
            )
            plt.legend()
            fname = os.path.join(working_dir, f"{dset}_hwa_curve_agg.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated HWA curve for {dset}: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # Aggregate final test accuracy across runs (print only)
    try:
        final_accs = []
        for exp in all_experiment_data:
            if dset in exp:
                y_true = np.array(exp[dset].get("ground_truth", []))
                y_pred = np.array(exp[dset].get("predictions", []))
                if len(y_true):
                    final_accs.append((y_true == y_pred).mean())
        if final_accs:
            final_accs = np.array(final_accs)
            mean_acc = final_accs.mean()
            sem_acc = (
                final_accs.std(ddof=1) / np.sqrt(len(final_accs))
                if len(final_accs) > 1
                else 0.0
            )
            print(
                f"{dset} – Mean Test Accuracy over {len(final_accs)} runs: "
                f"{mean_acc:.4f} ± {sem_acc:.4f}"
            )
    except Exception as e:
        print(f"Error computing aggregated accuracy for {dset}: {e}")

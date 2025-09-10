import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
# directory preparation
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# paths of the different experiment_data.npy files (given by user)
experiment_data_path_list = [
    "experiments/2025-08-16_00-46-17_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_fb6897937f97456c82349c0c5eef43b8_proc_3065838/experiment_data.npy",
    "experiments/2025-08-16_00-46-17_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_faf3a123d2614806af34cbb29cbfc101_proc_3065840/experiment_data.npy",
    "experiments/2025-08-16_00-46-17_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_6b98d960d33c44309c0c879944f013ac_proc_3065841/experiment_data.npy",
]

# ------------------------------------------------------------------
# load all runs
all_experiment_data = []
for p in experiment_data_path_list:
    try:
        arr = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_experiment_data.append(arr)
    except Exception as e:
        print(f"Error loading {p}: {e}")

if not all_experiment_data:
    print("No experiment data could be loaded; exiting.")
    exit()

# ------------------------------------------------------------------
# gather all dataset names that appear in at least one run
dataset_names = set()
for run in all_experiment_data:
    dataset_names.update(run.keys())


# ------------------------------------------------------------------
def stack_metric(runs, ds, key_chain):
    """Return stacked np.array of shape (n_runs, n_epochs) or None."""
    series_list = []
    for run in runs:
        cur = run.get(ds, {})
        for k in key_chain:
            cur = cur.get(k, [])
        series_list.append(np.array(cur, dtype=float))

    # remove empty series
    series_list = [s for s in series_list if s.size > 0]
    if not series_list:
        return None

    # align by shortest length so every epoch has same #runs
    min_len = min(map(len, series_list))
    if min_len == 0:
        return None
    series_list = [s[:min_len] for s in series_list]
    return np.vstack(series_list)  # shape (n_runs, min_len)


# ------------------------------------------------------------------
for ds in dataset_names:
    # --------------------------------------------------------------
    # aggregated HSCA (train)
    try:
        train_mat = stack_metric(all_experiment_data, ds, ["metrics", "train"])
        if train_mat is not None:
            epochs = np.arange(1, train_mat.shape[1] + 1)
            mean = train_mat.mean(axis=0)
            sem = train_mat.std(axis=0, ddof=1) / np.sqrt(train_mat.shape[0])

            plt.figure()
            plt.plot(epochs, mean, label="Train HSCA – mean", color="tab:blue")
            plt.fill_between(
                epochs,
                mean - sem,
                mean + sem,
                alpha=0.3,
                color="tab:blue",
                label="±1 SEM",
            )
            plt.title(f"{ds} – Aggregated Train HSCA (n={train_mat.shape[0]} runs)")
            plt.xlabel("Epoch")
            plt.ylabel("HSCA")
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{ds}_agg_train_HSCA.png")
            plt.savefig(fname)
            plt.close()
            # console summary
            print(f"{ds} final Train HSCA: {mean[-1]:.4f} ± {sem[-1]:.4f}")
        else:
            print(f"{ds}: No train HSCA found; skipping plot.")
    except Exception as e:
        print(f"Error creating aggregated train HSCA plot for {ds}: {e}")
        plt.close()

    # --------------------------------------------------------------
    # aggregated HSCA (validation/test)
    try:
        val_mat = stack_metric(all_experiment_data, ds, ["metrics", "val"])
        if val_mat is not None:
            epochs = np.arange(1, val_mat.shape[1] + 1)
            mean = val_mat.mean(axis=0)
            sem = val_mat.std(axis=0, ddof=1) / np.sqrt(val_mat.shape[0])

            plt.figure()
            plt.plot(epochs, mean, label="Val/Test HSCA – mean", color="tab:green")
            plt.fill_between(
                epochs,
                mean - sem,
                mean + sem,
                alpha=0.3,
                color="tab:green",
                label="±1 SEM",
            )
            plt.title(f"{ds} – Aggregated Val/Test HSCA (n={val_mat.shape[0]} runs)")
            plt.xlabel("Epoch")
            plt.ylabel("HSCA")
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{ds}_agg_val_HSCA.png")
            plt.savefig(fname)
            plt.close()
            # console summary
            print(f"{ds} final Val/Test HSCA: {mean[-1]:.4f} ± {sem[-1]:.4f}")
        else:
            print(f"{ds}: No validation HSCA found; skipping plot.")
    except Exception as e:
        print(f"Error creating aggregated val HSCA plot for {ds}: {e}")
        plt.close()

    # --------------------------------------------------------------
    # aggregated training loss
    try:
        loss_mat = stack_metric(all_experiment_data, ds, ["losses", "train"])
        if loss_mat is not None:
            epochs = np.arange(1, loss_mat.shape[1] + 1)
            mean = loss_mat.mean(axis=0)
            sem = loss_mat.std(axis=0, ddof=1) / np.sqrt(loss_mat.shape[0])

            plt.figure()
            plt.plot(epochs, mean, label="Train Loss – mean", color="tab:red")
            plt.fill_between(
                epochs,
                mean - sem,
                mean + sem,
                alpha=0.3,
                color="tab:red",
                label="±1 SEM",
            )
            plt.title(f"{ds} – Aggregated Train Loss (n={loss_mat.shape[0]} runs)")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{ds}_agg_train_loss.png")
            plt.savefig(fname)
            plt.close()
        else:
            print(f"{ds}: No train loss found; skipping plot.")
    except Exception as e:
        print(f"Error creating aggregated train loss plot for {ds}: {e}")
        plt.close()

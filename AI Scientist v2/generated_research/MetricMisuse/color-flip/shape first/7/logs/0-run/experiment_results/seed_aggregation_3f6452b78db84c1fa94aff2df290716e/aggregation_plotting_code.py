import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------
# basic setup
# ---------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# paths provided by the platform
experiment_data_path_list = [
    "experiments/2025-08-16_02-30-16_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_51edb58f0eeb48b2adfce152111207f5_proc_3095886/experiment_data.npy",
    "experiments/2025-08-16_02-30-16_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_9501006aac1447e78d54f88b6fb613be_proc_3095883/experiment_data.npy",
    "experiments/2025-08-16_02-30-16_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_7f9d2dee3be844c6824b1533be6549a0_proc_3095884/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_experiment_data.append(data)
    except Exception as e:
        print(f"Error loading {p}: {e}")


# ---------------------------------------------------------
# helper to stack & truncate runs to the same epoch length
# ---------------------------------------------------------
def stack_across_runs(run_list, column_idx):
    """
    run_list: list of 2-D arrays with shape (epochs, 2+) containing
              first column = epoch, other columns = values.
    column_idx: int – column from which the values are taken.
    returns epochs (1-D), stacked_values (runs, epochs)
    """
    if not run_list:
        return None, None
    # keep only runs that actually contain data
    run_list = [r for r in run_list if r.size]
    if not run_list:
        return None, None
    min_len = min(len(r) for r in run_list)
    epochs = run_list[0][:min_len, 0]
    stacked = np.stack([r[:min_len, column_idx] for r in run_list], axis=0)
    return epochs, stacked


# ---------------------------------------------------------
# iterate over datasets and create aggregate plots
# ---------------------------------------------------------
datasets = set()
for run in all_experiment_data:
    datasets.update(run.keys())

for ds in datasets:
    runs_losses_tr, runs_losses_val, runs_metrics_val = [], [], []
    for run in all_experiment_data:
        ds_data = run.get(ds, {})
        # convert to numpy arrays or empty array
        runs_losses_tr.append(np.array(ds_data.get("losses", {}).get("train", [])))
        runs_losses_val.append(np.array(ds_data.get("losses", {}).get("val", [])))
        runs_metrics_val.append(np.array(ds_data.get("metrics", {}).get("val", [])))

    # -------------------------------------------------
    # 1) aggregate loss curves
    # -------------------------------------------------
    try:
        epochs_tr, stack_tr = stack_across_runs(runs_losses_tr, 1)
        epochs_val, stack_val = stack_across_runs(runs_losses_val, 1)

        if stack_tr is not None or stack_val is not None:
            plt.figure()

            if stack_tr is not None:
                mu_tr = stack_tr.mean(0)
                se_tr = stack_tr.std(0, ddof=1) / np.sqrt(stack_tr.shape[0])
                plt.plot(epochs_tr, mu_tr, label="Train Loss (mean)")
                plt.fill_between(
                    epochs_tr,
                    mu_tr - se_tr,
                    mu_tr + se_tr,
                    alpha=0.3,
                    label="Train Loss ±SE",
                )

            if stack_val is not None:
                mu_val = stack_val.mean(0)
                se_val = stack_val.std(0, ddof=1) / np.sqrt(stack_val.shape[0])
                plt.plot(epochs_val, mu_val, label="Val Loss (mean)")
                plt.fill_between(
                    epochs_val,
                    mu_val - se_val,
                    mu_val + se_val,
                    alpha=0.3,
                    label="Val Loss ±SE",
                )

            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{ds} Aggregate Loss Curves (N={len(all_experiment_data)} Runs)")
            plt.legend()
            fname = f"{ds}_aggregate_loss_curves.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error creating aggregate loss for {ds}: {e}")
        plt.close()

    # -------------------------------------------------
    # 2) aggregate validation metrics
    # -------------------------------------------------
    try:
        epochs_met, stack_swa = stack_across_runs(runs_metrics_val, 1)
        _, stack_cwa = stack_across_runs(runs_metrics_val, 2)
        _, stack_hwa = stack_across_runs(runs_metrics_val, 3)

        if stack_swa is not None:
            plt.figure()
            # SWA
            mu_swa = stack_swa.mean(0)
            se_swa = stack_swa.std(0, ddof=1) / np.sqrt(stack_swa.shape[0])
            plt.plot(epochs_met, mu_swa, label="SWA (mean)")
            plt.fill_between(
                epochs_met,
                mu_swa - se_swa,
                mu_swa + se_swa,
                alpha=0.25,
                label="SWA ±SE",
            )
            # CWA
            if stack_cwa is not None:
                mu_cwa = stack_cwa.mean(0)
                se_cwa = stack_cwa.std(0, ddof=1) / np.sqrt(stack_cwa.shape[0])
                plt.plot(epochs_met, mu_cwa, label="CWA (mean)")
                plt.fill_between(
                    epochs_met,
                    mu_cwa - se_cwa,
                    mu_cwa + se_cwa,
                    alpha=0.25,
                    label="CWA ±SE",
                )
            # HWA
            if stack_hwa is not None:
                mu_hwa = stack_hwa.mean(0)
                se_hwa = stack_hwa.std(0, ddof=1) / np.sqrt(stack_hwa.shape[0])
                plt.plot(epochs_met, mu_hwa, label="HWA (mean)")
                plt.fill_between(
                    epochs_met,
                    mu_hwa - se_hwa,
                    mu_hwa + se_hwa,
                    alpha=0.25,
                    label="HWA ±SE",
                )

            plt.xlabel("Epoch")
            plt.ylabel("Weighted Accuracy")
            plt.title(
                f"{ds} Aggregate Validation Metrics (N={len(all_experiment_data)} Runs)"
            )
            plt.legend()
            fname = f"{ds}_aggregate_val_metrics.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()

            # Print final HWA statistics if available
            if stack_hwa is not None:
                final_hwa_vals = stack_hwa[:, -1]
                mean_final = final_hwa_vals.mean()
                se_final = final_hwa_vals.std(ddof=1) / np.sqrt(len(final_hwa_vals))
                print(f"{ds} Final HWA: {mean_final:.4f} ± {se_final:.4f} (SE)")

    except Exception as e:
        print(f"Error creating aggregate metrics for {ds}: {e}")
        plt.close()

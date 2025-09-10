import matplotlib.pyplot as plt
import numpy as np
import os
import math

# mandatory working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------- Load all experiments ---------------------------
experiment_data_path_list = [
    "experiments/2025-08-17_23-44-10_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_723c819693a84042b0bb82e30dff271a_proc_3470354/experiment_data.npy",
    "experiments/2025-08-17_23-44-10_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_db91c43a7fd24ea9b884b0d2b9d25b92_proc_3470356/experiment_data.npy",
    "experiments/2025-08-17_23-44-10_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_6861a0d5657f4b899c6741abbd4cef15_proc_3470353/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp)
    except Exception as e:
        print(f"Error loading {p}: {e}")


# ------------- Helper to aggregate --------------------------------------
def stack_metric(runs, ds_key, metric_path):
    """
    metric_path: list of nested keys to reach the 1-D metric array
    returns np.ndarray shape (n_runs, n_common_epochs)
    """
    arrays = []
    for run in runs:
        try:
            arr = run[ds_key]
            for k in metric_path:
                arr = arr[k]
            arrays.append(np.asarray(arr, dtype=float))
        except KeyError:
            pass
    if not arrays:
        return None
    min_len = min(len(a) for a in arrays)
    stacked = np.vstack([a[:min_len] for a in arrays])
    return stacked


ds_name = "SPR_BENCH_reasoning"
if not all_experiment_data or ds_name not in all_experiment_data[0]:
    print("No experiment data with the expected key was found.")
else:
    # ------------ Figure 1 : aggregated Macro-F1 ------------------------
    try:
        train_f1 = stack_metric(
            all_experiment_data, ds_name, ["metrics", "train_macro_f1"]
        )
        val_f1 = stack_metric(all_experiment_data, ds_name, ["metrics", "val_macro_f1"])
        if train_f1 is not None and val_f1 is not None:
            epochs = np.arange(train_f1.shape[1])
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
            fig.suptitle(
                f"{ds_name} Mean Macro-F1 ± SEM over Epochs\nLeft: Train  Right: Validation",
                fontsize=14,
            )
            # Train subplot
            mean_t = train_f1.mean(axis=0)
            sem_t = train_f1.std(axis=0, ddof=1) / np.sqrt(train_f1.shape[0])
            axes[0].plot(epochs, mean_t, label="mean train")
            axes[0].fill_between(
                epochs,
                mean_t - sem_t,
                mean_t + sem_t,
                color="blue",
                alpha=0.2,
                label="±SEM",
            )
            # Val subplot
            mean_v = val_f1.mean(axis=0)
            sem_v = val_f1.std(axis=0, ddof=1) / np.sqrt(val_f1.shape[0])
            axes[1].plot(epochs, mean_v, color="orange", label="mean val")
            axes[1].fill_between(
                epochs,
                mean_v - sem_v,
                mean_v + sem_v,
                color="orange",
                alpha=0.2,
                label="±SEM",
            )
            for ax, ttl in zip(axes, ["Train Macro-F1", "Validation Macro-F1"]):
                ax.set_title(ttl)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Macro-F1")
                ax.set_ylim(0, 1)
                ax.legend()
            out_path = os.path.join(
                working_dir, f"{ds_name.lower()}_macro_f1_mean_sem.png"
            )
            plt.savefig(out_path)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated Macro-F1 plot: {e}")
        plt.close()

    # ------------ Figure 2 : aggregated Loss ----------------------------
    try:
        train_loss = stack_metric(all_experiment_data, ds_name, ["losses", "train"])
        val_loss = stack_metric(all_experiment_data, ds_name, ["losses", "val"])
        if train_loss is not None and val_loss is not None:
            epochs = np.arange(train_loss.shape[1])
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
            fig.suptitle(
                f"{ds_name} Mean Cross-Entropy Loss ± SEM over Epochs\nLeft: Train  Right: Validation",
                fontsize=14,
            )
            mean_t = train_loss.mean(axis=0)
            sem_t = train_loss.std(axis=0, ddof=1) / np.sqrt(train_loss.shape[0])
            axes[0].plot(epochs, mean_t, label="mean train")
            axes[0].fill_between(
                epochs,
                mean_t - sem_t,
                mean_t + sem_t,
                color="blue",
                alpha=0.2,
                label="±SEM",
            )
            mean_v = val_loss.mean(axis=0)
            sem_v = val_loss.std(axis=0, ddof=1) / np.sqrt(val_loss.shape[0])
            axes[1].plot(epochs, mean_v, color="orange", label="mean val")
            axes[1].fill_between(
                epochs,
                mean_v - sem_v,
                mean_v + sem_v,
                color="orange",
                alpha=0.2,
                label="±SEM",
            )
            for ax, ttl in zip(axes, ["Train Loss", "Validation Loss"]):
                ax.set_title(ttl)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.legend()
            out_path = os.path.join(working_dir, f"{ds_name.lower()}_loss_mean_sem.png")
            plt.savefig(out_path)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated Loss plot: {e}")
        plt.close()

    # ------------- Console summary of final test metrics -----------------
    try:
        test_loss_list = []
        test_f1_list = []
        for run in all_experiment_data:
            try:
                rec = run[ds_name]
                test_loss_list.append(float(rec.get("test_loss", np.nan)))
                test_f1_list.append(float(rec.get("test_macro_f1", np.nan)))
            except KeyError:
                pass
        if test_loss_list:
            tl = np.array(test_loss_list)
            tf = np.array(test_f1_list)
            print(f"\nAggregated Test Metrics for {ds_name}:")
            print(f"  Loss       : {tl.mean():.4f} ± {tl.std(ddof=1):.4f}")
            print(f"  Macro-F1   : {tf.mean():.4f} ± {tf.std(ddof=1):.4f}")
    except Exception as e:
        print(f"Error computing aggregated test metrics: {e}")

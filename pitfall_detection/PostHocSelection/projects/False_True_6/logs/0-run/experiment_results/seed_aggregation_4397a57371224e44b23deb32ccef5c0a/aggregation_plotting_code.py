import matplotlib.pyplot as plt
import numpy as np
import os
from math import sqrt

# --------------------------------------------------------------------------
# mandatory working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------------------------------------------------------
# paths supplied by the task
experiment_data_path_list = [
    "experiments/2025-08-14_17-37-20_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_93d52ce36f784cc0ad1f60657a0aa157_proc_2700561/experiment_data.npy",
    "experiments/2025-08-14_17-37-20_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_48de19f26e044d96ae1bebc814b8227b_proc_2700562/experiment_data.npy",
    "experiments/2025-08-14_17-37-20_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_6fad2165297240b59574a1e6fef4ba6b_proc_2700563/experiment_data.npy",
]

# --------------------------------------------------------------------------
# load all experiment dicts
all_exp_data = []
for p in experiment_data_path_list:
    try:
        full_p = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        d = np.load(full_p, allow_pickle=True).item()
        all_exp_data.append(d)
    except Exception as e:
        print(f"Error loading {p}: {e}")

if not all_exp_data:
    print("No experiment data loaded – nothing to plot.")
else:
    # assume identical structure across runs
    top_key = "EPOCH_TUNING"
    datasets = list(all_exp_data[0][top_key].keys())

    for ds_name in datasets:
        # --------------------------------------------------------------
        # collect per-run series for this dataset
        train_losses, val_losses, val_accs = [], [], []
        test_metrics_list = []

        for run_d in all_exp_data:
            try:
                rec = run_d[top_key][ds_name]
                train_losses.append(np.asarray(rec["losses"]["train"], dtype=float))
                val_losses.append(np.asarray(rec["losses"]["val"], dtype=float))
                val_accs.append(
                    np.asarray([m["acc"] for m in rec["metrics"]["val"]], dtype=float)
                )
                test_metrics_list.append(rec["metrics"]["test"])
            except Exception as e:
                print(f"Run skipped for {ds_name} due to: {e}")

        # ensure at least one successful run
        if not train_losses:
            continue

        # stack and compute statistics (truncate to min length if unequal)
        min_len = min(map(len, train_losses))
        train_mat = np.stack([tl[:min_len] for tl in train_losses])
        val_mat = np.stack([vl[:min_len] for vl in val_losses])
        acc_mat = np.stack([va[:min_len] for va in val_accs])

        epochs = np.arange(1, min_len + 1)
        n_runs = train_mat.shape[0]
        sem = lambda x: x.std(0) / sqrt(n_runs)

        # ---------------------------------------------------------- plot 1
        try:
            plt.figure()
            plt.plot(epochs, train_mat.mean(0), label="Train Mean", color="C0")
            plt.fill_between(
                epochs,
                train_mat.mean(0) - sem(train_mat),
                train_mat.mean(0) + sem(train_mat),
                color="C0",
                alpha=0.3,
                label="Train ±SEM",
            )
            plt.plot(epochs, val_mat.mean(0), label="Val Mean", color="C1")
            plt.fill_between(
                epochs,
                val_mat.mean(0) - sem(val_mat),
                val_mat.mean(0) + sem(val_mat),
                color="C1",
                alpha=0.3,
                label="Val ±SEM",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(
                f"{ds_name} Aggregated Loss Curves\nMean ± SEM over {n_runs} runs"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_agg_loss_curve.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating aggregated loss curve for {ds_name}: {e}")
            plt.close()

        # ---------------------------------------------------------- plot 2
        try:
            plt.figure()
            plt.plot(epochs, acc_mat.mean(0), color="C2", label="Val Acc Mean")
            plt.fill_between(
                epochs,
                acc_mat.mean(0) - sem(acc_mat),
                acc_mat.mean(0) + sem(acc_mat),
                color="C2",
                alpha=0.3,
                label="Val Acc ±SEM",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.ylim(0, 1)
            plt.title(
                f"{ds_name} Aggregated Validation Accuracy\nMean ± SEM over {n_runs} runs"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_agg_val_accuracy.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating aggregated accuracy plot for {ds_name}: {e}")
            plt.close()

        # ---------------------------------------------------------- plot 3 : test metrics
        try:
            # gather into array
            metric_names = list(test_metrics_list[0].keys())
            metric_vals = np.array(
                [[tm[k] for k in metric_names] for tm in test_metrics_list], dtype=float
            )
            means = metric_vals.mean(0)
            errors = metric_vals.std(0) / sqrt(n_runs)

            x = np.arange(len(metric_names))
            plt.figure()
            plt.bar(x, means, yerr=errors, capsize=5, color="skyblue")
            plt.xticks(x, metric_names)
            plt.ylim(0, 1)
            plt.title(
                f"{ds_name} Aggregated Test Metrics\nMean ± SEM over {n_runs} runs"
            )
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_agg_test_metrics.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating aggregated test metric plot for {ds_name}: {e}")
            plt.close()

        # ---------------------------------------------------------- print aggregated test metrics
        print(f"\n{ds_name} aggregated test metrics over {n_runs} runs:")
        for k, m, e in zip(metric_names, means, errors):
            print(f"  {k:>10s}: {m:.4f} ± {e:.4f}")

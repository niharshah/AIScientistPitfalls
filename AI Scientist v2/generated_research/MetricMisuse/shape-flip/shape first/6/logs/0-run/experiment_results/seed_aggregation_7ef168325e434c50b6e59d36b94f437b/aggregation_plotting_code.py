import matplotlib.pyplot as plt
import numpy as np
import os
from math import sqrt

# ---------- set up working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- list of experiment_data paths (relative to AI_SCIENTIST_ROOT) ----------
experiment_data_path_list = [
    "experiments/2025-08-14_17-37-20_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_b263c8a563fd4c06b7d448a539cea2f2_proc_2703062/experiment_data.npy",
    "experiments/2025-08-14_17-37-20_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_f567ea8876254708904edc2ebc8fe7e5_proc_2703059/experiment_data.npy",
    "experiments/2025-08-14_17-37-20_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_2a8d1200e365415a93691c344707d711_proc_2703061/experiment_data.npy",
]

# ---------- load all experiment_data ----------
all_experiment_data = []
try:
    root = os.getenv("AI_SCIENTIST_ROOT", "")
    for p in experiment_data_path_list:
        full_path = os.path.join(root, p)
        if os.path.isfile(full_path):
            all_experiment_data.append(np.load(full_path, allow_pickle=True).item())
        else:
            print(f"File not found: {full_path}")
except Exception as e:
    print(f"Error loading experiment data: {e}")

if not all_experiment_data:
    print("No experiment data loaded – nothing to plot.")
else:
    # discover dataset names from the first run
    dataset_names = list(all_experiment_data[0].get("EPOCH_TUNING", {}).keys())

    for ds in dataset_names:
        # containers for aggregation
        train_losses, val_losses, val_accs, test_metrics_list = [], [], [], []
        max_epochs = 0

        # collect data from every run
        for run in all_experiment_data:
            rec = run.get("EPOCH_TUNING", {}).get(ds, None)
            if rec is None:
                continue
            tr_loss = np.array(rec["losses"]["train"])
            vl_loss = np.array(rec["losses"]["val"])
            epochs_here = len(tr_loss)
            max_epochs = max(max_epochs, epochs_here)

            # keep only common epoch length across metrics of this run
            min_len = min(len(tr_loss), len(vl_loss))
            tr_loss = tr_loss[:min_len]
            vl_loss = vl_loss[:min_len]

            train_losses.append(tr_loss)
            val_losses.append(vl_loss)

            # validation accuracy list of dicts -> list of floats
            v_acc = np.array([m["acc"] for m in rec["metrics"]["val"]][:min_len])
            val_accs.append(v_acc)

            test_metrics_list.append(rec["metrics"]["test"])

        n_runs = len(train_losses)
        if n_runs == 0:
            continue  # nothing available for this dataset

        # pad runs to same length if necessary (right pad with nan then nanmean will ignore)
        def pad_to(arr_list, target_len):
            return [
                np.pad(a, (0, target_len - len(a)), constant_values=np.nan)
                for a in arr_list
            ]

        train_losses = np.vstack(pad_to(train_losses, max_epochs))
        val_losses = np.vstack(pad_to(val_losses, max_epochs))
        val_accs = np.vstack(pad_to(val_accs, max_epochs))

        epochs = np.arange(1, max_epochs + 1)

        # ---------- aggregated train / val loss ----------
        try:
            plt.figure()
            mean_tr = np.nanmean(train_losses, axis=0)
            se_tr = np.nanstd(train_losses, axis=0, ddof=1) / sqrt(n_runs)

            mean_vl = np.nanmean(val_losses, axis=0)
            se_vl = np.nanstd(val_losses, axis=0, ddof=1) / sqrt(n_runs)

            plt.plot(epochs, mean_tr, label="Train Mean", color="tab:blue")
            plt.fill_between(
                epochs,
                mean_tr - se_tr,
                mean_tr + se_tr,
                color="tab:blue",
                alpha=0.2,
                label="Train ±SE",
            )

            plt.plot(epochs, mean_vl, label="Val Mean", color="tab:orange")
            plt.fill_between(
                epochs,
                mean_vl - se_vl,
                mean_vl + se_vl,
                color="tab:orange",
                alpha=0.2,
                label="Val ±SE",
            )

            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{ds} Aggregated Loss Curves\nShaded: ±SE over {n_runs} runs")
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{ds}_aggregated_loss_curve.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating aggregated loss curve for {ds}: {e}")
            plt.close()

        # ---------- aggregated validation accuracy ----------
        try:
            plt.figure()
            mean_acc = np.nanmean(val_accs, axis=0)
            se_acc = np.nanstd(val_accs, axis=0, ddof=1) / sqrt(n_runs)

            plt.plot(epochs, mean_acc, color="tab:green", label="Val Acc Mean")
            plt.fill_between(
                epochs,
                mean_acc - se_acc,
                mean_acc + se_acc,
                color="tab:green",
                alpha=0.2,
                label="Val Acc ±SE",
            )
            plt.ylim(0, 1)
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(
                f"{ds} Aggregated Validation Accuracy\nShaded: ±SE over {n_runs} runs"
            )
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{ds}_aggregated_val_accuracy.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating aggregated val accuracy for {ds}: {e}")
            plt.close()

        # ---------- aggregated test metrics bar chart ----------
        try:
            # collect test metrics into array
            metric_keys = sorted(test_metrics_list[0].keys())
            metric_matrix = np.array(
                [[tm[k] for k in metric_keys] for tm in test_metrics_list]
            )
            mean_test = metric_matrix.mean(axis=0)
            se_test = metric_matrix.std(axis=0, ddof=1) / sqrt(n_runs)

            plt.figure()
            x = np.arange(len(metric_keys))
            plt.bar(x, mean_test, yerr=se_test, color="skyblue", capsize=5)
            plt.xticks(x, metric_keys)
            plt.ylim(0, 1)
            plt.title(
                f"{ds} Aggregated Test Metrics\nBars: mean, Error: ±SE ({n_runs} runs)"
            )
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{ds}_aggregated_test_metrics.png")
            plt.savefig(fname)
            plt.close()

            # also print the values
            print(f"\n{ds} – Aggregated Test Metrics over {n_runs} runs:")
            for k, m, s in zip(metric_keys, mean_test, se_test):
                print(f"  {k}: {m:.4f} ± {s:.4f}")
        except Exception as e:
            print(f"Error creating aggregated test metrics for {ds}: {e}")
            plt.close()

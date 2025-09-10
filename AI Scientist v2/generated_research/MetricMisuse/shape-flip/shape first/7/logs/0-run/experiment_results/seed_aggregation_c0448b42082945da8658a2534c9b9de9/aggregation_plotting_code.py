import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# Load all experiment_data dicts
experiment_data_path_list = [
    "experiments/2025-08-14_19-19-19_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_b0bda82c2c9b49329c58b249c02ee241_proc_2720208/experiment_data.npy",
    "experiments/2025-08-14_19-19-19_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_52da10742dbf469080e60a3a6e79ead2_proc_2720209/experiment_data.npy",
    "experiments/2025-08-14_19-19-19_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_179f4e37de9c4831982efe5e472636d2_proc_2720210/experiment_data.npy",
]

all_experiment_data = []
root = os.getenv("AI_SCIENTIST_ROOT", "")
for p in experiment_data_path_list:
    try:
        exp = np.load(os.path.join(root, p), allow_pickle=True).item()
        all_experiment_data.append(exp)
    except Exception as e:
        print(f"Error loading {p}: {e}")

# ------------------------------------------------------------------
# Aggregate per dataset
datasets = defaultdict(list)
for exp in all_experiment_data:
    for dset_name, dset_val in exp.items():
        datasets[dset_name].append(dset_val)

for dset_name, runs in datasets.items():
    # ------------  gather per-epoch arrays ------------
    # Use first run as template for epoch vector
    try:
        epochs = np.array([e for e, _ in runs[0]["losses"]["train"]])
    except Exception as e:
        print(f"Error extracting epochs for {dset_name}: {e}")
        continue

    def collect(field):
        """Return 2-D array shape (num_runs, num_epochs) for given field path."""
        arr_list = []
        for r in runs:
            try:
                arr = np.array([v for _, v in r[field]["train"]])  # train list
            except Exception:
                arr = None
            arr_list.append(arr)
        return np.stack(arr_list)

    # Losses
    tr_loss = np.stack([np.array([v for _, v in r["losses"]["train"]]) for r in runs])
    val_loss = np.stack([np.array([v for _, v in r["losses"]["val"]]) for r in runs])
    # Metrics (SWA)
    tr_swa = np.stack([np.array([v for _, v in r["metrics"]["train"]]) for r in runs])
    val_swa = np.stack([np.array([v for _, v in r["metrics"]["val"]]) for r in runs])

    # ------------  mean & stderr ------------
    def mean_stderr(x):
        return x.mean(axis=0), x.std(axis=0) / np.sqrt(x.shape[0])

    tr_loss_m, tr_loss_se = mean_stderr(tr_loss)
    val_loss_m, val_loss_se = mean_stderr(val_loss)
    tr_swa_m, tr_swa_se = mean_stderr(tr_swa)
    val_swa_m, val_swa_se = mean_stderr(val_swa)

    # ------------  Plot 1 : Loss -------------
    try:
        plt.figure()
        plt.plot(epochs, tr_loss_m, color="tab:blue", label="Train (mean)")
        plt.fill_between(
            epochs,
            tr_loss_m - tr_loss_se,
            tr_loss_m + tr_loss_se,
            color="tab:blue",
            alpha=0.3,
            label="Train (stderr)",
        )
        plt.plot(epochs, val_loss_m, color="tab:orange", label="Val (mean)")
        plt.fill_between(
            epochs,
            val_loss_m - val_loss_se,
            val_loss_m + val_loss_se,
            color="tab:orange",
            alpha=0.3,
            label="Val (stderr)",
        )
        plt.title(f"{dset_name} – Cross-Entropy Loss vs Epoch (mean ± stderr)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        fname = os.path.join(
            working_dir, f"{dset_name.lower()}_aggregated_loss_curves.png"
        )
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dset_name}: {e}")
        plt.close()

    # ------------  Plot 2 : SWA -------------
    try:
        plt.figure()
        plt.plot(epochs, tr_swa_m, color="tab:green", label="Train (mean)")
        plt.fill_between(
            epochs,
            tr_swa_m - tr_swa_se,
            tr_swa_m + tr_swa_se,
            color="tab:green",
            alpha=0.3,
            label="Train (stderr)",
        )
        plt.plot(epochs, val_swa_m, color="tab:red", label="Val (mean)")
        plt.fill_between(
            epochs,
            val_swa_m - val_swa_se,
            val_swa_m + val_swa_se,
            color="tab:red",
            alpha=0.3,
            label="Val (stderr)",
        )
        plt.title(f"{dset_name} – Shape-Weighted Accuracy vs Epoch (mean ± stderr)")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        plt.legend()
        fname = os.path.join(
            working_dir, f"{dset_name.lower()}_aggregated_swa_curves.png"
        )
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated SWA plot for {dset_name}: {e}")
        plt.close()

    # ------------  Plot 3 : Test accuracy bar chart -------------
    try:
        test_accs = []
        for r in runs:
            gts = r.get("ground_truth", [])
            preds = r.get("predictions", [])
            acc = (
                (sum(int(g == p) for g, p in zip(gts, preds)) / len(gts))
                if gts
                else np.nan
            )
            test_accs.append(acc)
        test_accs = np.array(test_accs)
        mean_acc = np.nanmean(test_accs)
        se_acc = np.nanstd(test_accs) / np.sqrt(len(test_accs))
        plt.figure()
        plt.bar([dset_name], [mean_acc], yerr=[se_acc], capsize=5)
        plt.title(f"{dset_name} – Test Accuracy (mean ± stderr)")
        plt.ylabel("Accuracy")
        plt.text(
            0, mean_acc + 0.01, f"{mean_acc:.2f}±{se_acc:.2f}", ha="center", va="bottom"
        )
        fname = os.path.join(
            working_dir, f"{dset_name.lower()}_aggregated_test_accuracy.png"
        )
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated test accuracy for {dset_name}: {e}")
        plt.close()

    # ------------  Print summary -------------
    best_val_swa_each_run = [
        max(r["metrics"]["val"], key=lambda t: t[1])[1] for r in runs
    ]
    best_val_swa_m = np.mean(best_val_swa_each_run)
    best_val_swa_se = np.std(best_val_swa_each_run) / np.sqrt(
        len(best_val_swa_each_run)
    )
    print(f"{dset_name} – Best Val SWA: {best_val_swa_m:.4f} ± {best_val_swa_se:.4f}")
    print(f"{dset_name} – Test Acc   : {mean_acc:.4f} ± {se_acc:.4f}")

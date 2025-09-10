import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- Gather data from all runs ----------
experiment_data_path_list = [
    "experiments/2025-08-15_14-47-52_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_66cc9825fbed4eb6b3dc7bec93708776_proc_2916117/experiment_data.npy",
    "experiments/2025-08-15_14-47-52_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_d14fadf4ef3745b6927149063f5dfa66_proc_2916116/experiment_data.npy",
    "experiments/2025-08-15_14-47-52_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_7bde6a3f3e834dc1ab964683b937b557_proc_2916118/experiment_data.npy",
]

all_runs_data = []
for p in experiment_data_path_list:
    try:
        root = os.getenv("AI_SCIENTIST_ROOT", "")
        exp_dict = np.load(os.path.join(root, p), allow_pickle=True).item()
        if "SPR_BENCH" in exp_dict:
            all_runs_data.append(exp_dict["SPR_BENCH"])
        else:
            print(f"'SPR_BENCH' not found in {p}")
    except Exception as e:
        print(f"Error loading {p}: {e}")

n_runs = len(all_runs_data)
if n_runs == 0:
    print("No data loaded; aborting plots.")
else:
    # ------------- align epoch lengths -------------
    min_epochs = min(len(d["losses"]["train"]) for d in all_runs_data)
    epochs = np.arange(1, min_epochs + 1)

    train_losses = np.stack([d["losses"]["train"][:min_epochs] for d in all_runs_data])
    val_losses = np.stack([d["losses"]["val"][:min_epochs] for d in all_runs_data])

    # Validation metrics
    val_swa = np.stack(
        [[m["SWA"] for m in d["metrics"]["val"][:min_epochs]] for d in all_runs_data]
    )
    val_cwa = np.stack(
        [[m["CWA"] for m in d["metrics"]["val"][:min_epochs]] for d in all_runs_data]
    )
    val_hwa = np.stack(
        [[m["HWA"] for m in d["metrics"]["val"][:min_epochs]] for d in all_runs_data]
    )

    # Test metrics
    test_metrics_keys = list(all_runs_data[0]["metrics"]["test"].keys())
    test_values = np.array(
        [
            [run["metrics"]["test"][k] for k in test_metrics_keys]
            for run in all_runs_data
        ]
    )

    # Helper: mean & sem
    def mean_sem(arr, axis=0):
        mean = arr.mean(axis=axis)
        sem = arr.std(axis=axis, ddof=1) / np.sqrt(arr.shape[axis])
        return mean, sem

    # ---------------- Plot 1: Loss curves ----------------
    try:
        plt.figure()
        tr_mean, tr_sem = mean_sem(train_losses)
        va_mean, va_sem = mean_sem(val_losses)

        plt.plot(epochs, tr_mean, label="Train Loss (mean)", color="tab:blue")
        plt.fill_between(
            epochs,
            tr_mean - tr_sem,
            tr_mean + tr_sem,
            color="tab:blue",
            alpha=0.3,
            label="Train SEM",
        )
        plt.plot(epochs, va_mean, label="Validation Loss (mean)", color="tab:orange")
        plt.fill_between(
            epochs,
            va_mean - va_sem,
            va_mean + va_sem,
            color="tab:orange",
            alpha=0.3,
            label="Val SEM",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Aggregated Loss Curves\nMean ± SEM across runs")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_agg_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot: {e}")
        plt.close()

    # ---------------- Plot 2: Validation metrics ----------------
    try:
        plt.figure()
        for metric_arr, name, color in [
            (val_swa, "SWA", "tab:green"),
            (val_cwa, "CWA", "tab:red"),
            (val_hwa, "HWA", "tab:purple"),
        ]:
            mean, sem = mean_sem(metric_arr)
            plt.plot(epochs, mean, label=f"{name} (mean)", color=color)
            plt.fill_between(
                epochs,
                mean - sem,
                mean + sem,
                color=color,
                alpha=0.3,
                label=f"{name} SEM",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.title("SPR_BENCH Aggregated Validation Metrics\nMean ± SEM across runs")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_agg_val_metrics.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated validation metric plot: {e}")
        plt.close()

    # ---------------- Plot 3: Test metrics ----------------
    try:
        plt.figure()
        test_mean, test_sem = mean_sem(test_values, axis=0)
        x = np.arange(len(test_metrics_keys))
        plt.bar(
            x,
            test_mean,
            yerr=test_sem,
            capsize=5,
            color=["tab:blue", "tab:orange", "tab:green"],
        )
        plt.xticks(x, test_metrics_keys)
        plt.ylim(0, 1)
        plt.ylabel("Weighted Accuracy")
        plt.title("SPR_BENCH Test Metrics\nMean ± SEM across runs")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_agg_test_metrics_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated test metric bar plot: {e}")
        plt.close()

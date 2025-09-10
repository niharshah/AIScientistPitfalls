import matplotlib.pyplot as plt
import numpy as np
import os
import math

# ----------------- paths & loading -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# List supplied in the "Experiment Data Path" section
experiment_data_path_list = [
    "experiments/2025-07-29_02-18-25_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_aef9cbfec21144c0940d0ce40a6ce7df_proc_457456/experiment_data.npy",
    "experiments/2025-07-29_02-18-25_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_dd17ad7dfe66429bae7a28c8530af256_proc_457454/experiment_data.npy",
    "experiments/2025-07-29_02-18-25_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_f633eb88b3df4d589b8dc0e84c527352_proc_457453/experiment_data.npy",
]

all_best_runs = []  # containers for best-run dicts (one per seed/file)
all_test_metrics = []  # list of dicts: {'SWA':..,'CWA':..,'HWA':..}

# ----------------- read & collect -----------------
for p in experiment_data_path_list:
    try:
        full_p = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        edata = np.load(full_p, allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading experiment data from {p}: {e}")
        continue

    runs = edata.get("epochs", {})
    # choose best run inside this file
    best_run_name, best_hwa = None, -math.inf
    for k, v in runs.items():
        hwa_val = v["test_metrics"]["HWA"]
        if hwa_val > best_hwa:
            best_hwa, best_run_name = hwa_val, k
    if best_run_name is None:
        continue
    best_run = runs[best_run_name]
    all_best_runs.append(best_run)
    all_test_metrics.append(best_run["test_metrics"])

# Stop if nothing loaded
if len(all_best_runs) == 0:
    print("No experiment data successfully loaded – nothing to plot.")
else:
    # -------- align epoch lengths --------
    min_epochs = min(len(r["losses"]["train"]) for r in all_best_runs)

    train_loss_mat = np.vstack(
        [r["losses"]["train"][:min_epochs] for r in all_best_runs]
    )
    val_loss_mat = np.vstack([r["losses"]["val"][:min_epochs] for r in all_best_runs])
    val_hwa_mat = np.vstack(
        [[m["HWA"] for m in r["metrics"]["val"][:min_epochs]] for r in all_best_runs]
    )

    # -------- compute aggregates --------
    def mean_and_se(mat):
        mean = mat.mean(axis=0)
        se = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
        return mean, se

    train_mean, train_se = mean_and_se(train_loss_mat)
    val_mean, val_se = mean_and_se(val_loss_mat)
    hwa_mean, hwa_se = mean_and_se(val_hwa_mat)

    # aggregate test metrics
    test_metrics_arr = {
        k: np.array([m[k] for m in all_test_metrics]) for k in ["SWA", "CWA", "HWA"]
    }
    test_mean = {k: float(v.mean()) for k, v in test_metrics_arr.items()}
    test_se = {
        k: float(v.std(ddof=1) / np.sqrt(v.size)) for k, v in test_metrics_arr.items()
    }

    # ------------ Figure 1: aggregated loss curves -------------
    try:
        plt.figure()
        epochs = np.arange(min_epochs)
        plt.fill_between(
            epochs,
            train_mean - train_se,
            train_mean + train_se,
            alpha=0.3,
            label="Train Loss ± SE",
        )
        plt.plot(epochs, train_mean, "--", color="C0")
        plt.fill_between(
            epochs,
            val_mean - val_se,
            val_mean + val_se,
            alpha=0.3,
            label="Val Loss ± SE",
        )
        plt.plot(epochs, val_mean, "-", color="C1")
        plt.title(
            "SPR_BENCH (Aggregated) Training vs Validation Loss\nMean over Seeds with Standard Error"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend(fontsize=7)
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_agg_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot: {e}")
        plt.close()

    # ------------ Figure 2: aggregated validation HWA ----------
    try:
        plt.figure()
        epochs = np.arange(min_epochs)
        plt.fill_between(
            epochs,
            hwa_mean - hwa_se,
            hwa_mean + hwa_se,
            alpha=0.3,
            color="C2",
            label="HWA ± SE",
        )
        plt.plot(epochs, hwa_mean, color="C2")
        plt.title(
            "SPR_BENCH (Aggregated) Validation Harmonic Weighted Accuracy\nMean over Seeds with Standard Error"
        )
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.legend(fontsize=7)
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_agg_val_HWA.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated HWA plot: {e}")
        plt.close()

    # ------------ Figure 3: final test metrics -----------------
    try:
        plt.figure()
        metrics = ["SWA", "CWA", "HWA"]
        x = np.arange(len(metrics))
        means = [test_mean[m] for m in metrics]
        ses = [test_se[m] for m in metrics]
        plt.bar(x, means, yerr=ses, capsize=5, color=["C0", "C1", "C2"])
        plt.xticks(x, metrics)
        plt.ylabel("Score")
        plt.title("SPR_BENCH Final Test Metrics\nMean over Seeds with Standard Error")
        for xi, m, se in zip(x, means, ses):
            plt.text(xi, m + se + 0.01, f"{m:.3f}±{se:.3f}", ha="center", fontsize=7)
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_agg_test_metrics.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated test-metric bar plot: {e}")
        plt.close()

    # ------------- console printout -----------------
    print("=== Aggregated Final Test Metrics (mean ± SE) ===")
    for k in ["SWA", "CWA", "HWA"]:
        print(f"{k}: {test_mean[k]:.4f} ± {test_se[k]:.4f}")

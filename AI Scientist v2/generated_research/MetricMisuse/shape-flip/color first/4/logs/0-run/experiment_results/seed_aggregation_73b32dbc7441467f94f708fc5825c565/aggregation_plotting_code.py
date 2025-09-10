import matplotlib.pyplot as plt
import numpy as np
import os

# ── paths ──────────────────────────────────────────────────────
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ── load all experiment_data dicts ─────────────────────────────
experiment_data_path_list = [
    "experiments/2025-08-30_19-33-09_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_dc379799bfa343bf9efb506dd06138d3_proc_1483402/experiment_data.npy",
    "experiments/2025-08-30_19-33-09_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_6521535a7d224a9097a92606eb4061b9_proc_1483403/experiment_data.npy",
    "experiments/2025-08-30_19-33-09_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_9aeda3761c6e44918c615750f3128f6c_proc_1483404/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        exp_dict = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_experiment_data.append(exp_dict)
    except Exception as e:
        print(f"Error loading {p}: {e}")

if not all_experiment_data:
    print("No experiment data could be loaded; aborting plotting.")
    exit()

# ── discover all run_keys available across experiments ─────────
run_keys = set()
for exp in all_experiment_data:
    run_keys.update(exp.keys())


# ── helper to stack aligned sequences ──────────────────────────
def stack_and_trim(list_of_lists):
    """Return np.array shape (n_runs, min_len) after trimming every list."""
    if not list_of_lists:
        return np.array([])
    min_len = min(len(seq) for seq in list_of_lists if len(seq) > 0)
    if min_len == 0:
        return np.array([])
    trimmed = [np.asarray(seq[:min_len]) for seq in list_of_lists]
    return np.vstack(trimmed)


# ── iterate over every run_key and create plots ────────────────
for run_key in run_keys:
    # gather per-run series ---------------------------------------------------
    train_losses, val_losses = [], []
    val_metrics_runs, test_metrics_runs = [], []

    for exp in all_experiment_data:
        run = exp.get(run_key, {})
        train_losses.append(run.get("losses", {}).get("train", []))
        val_losses.append(run.get("losses", {}).get("val", []))
        val_metrics_runs.append(run.get("metrics", {}).get("val", []))
        test_metrics_runs.append(run.get("metrics", {}).get("test", {}))

    # ── losses (train/val) ---------------------------------------------------
    try:
        tr_stack = stack_and_trim(train_losses)
        val_stack = stack_and_trim(val_losses)
        if tr_stack.size == 0 or val_stack.size == 0:
            raise ValueError("Empty loss arrays; skipping loss plot.")

        epochs = np.arange(1, tr_stack.shape[1] + 1)
        tr_mean, tr_se = tr_stack.mean(0), tr_stack.std(0, ddof=1) / np.sqrt(
            tr_stack.shape[0]
        )
        val_mean, val_se = val_stack.mean(0), val_stack.std(0, ddof=1) / np.sqrt(
            val_stack.shape[0]
        )

        plt.figure()
        plt.plot(epochs, tr_mean, label="Train Mean")
        plt.fill_between(
            epochs, tr_mean - tr_se, tr_mean + tr_se, alpha=0.3, label="Train SE"
        )
        plt.plot(epochs, val_mean, label="Val Mean")
        plt.fill_between(
            epochs, val_mean - val_se, val_mean + val_se, alpha=0.3, label="Val SE"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"Training vs Validation Loss (mean±SE) – {run_key}")
        plt.legend()
        fname = os.path.join(working_dir, f"{run_key}_agg_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {run_key}: {e}")
        plt.close()

    # ── validation metrics ---------------------------------------------------
    try:
        # find metric names present in at least one run
        metric_names = set()
        for run_val in val_metrics_runs:
            if run_val:
                metric_names.update(run_val[0].keys())
        if not metric_names:
            raise ValueError("No validation metrics found; skipping metric plot.")

        plt.figure()
        for m in sorted(metric_names):
            # collect per-run arrays for metric m
            metric_series = []
            for run_val in val_metrics_runs:
                metric_series.append(
                    [epoch_dict.get(m, np.nan) for epoch_dict in run_val]
                )
            metric_series_stack = stack_and_trim(metric_series)
            if metric_series_stack.size == 0:
                continue
            mean = np.nanmean(metric_series_stack, axis=0)
            se = np.nanstd(metric_series_stack, axis=0, ddof=1) / np.sqrt(
                metric_series_stack.shape[0]
            )
            epochs = np.arange(1, mean.size + 1)
            plt.plot(epochs, mean, label=f"{m} Mean")
            plt.fill_between(epochs, mean - se, mean + se, alpha=0.2, label=f"{m} SE")

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Validation Metrics (mean±SE) – {run_key}")
        plt.legend()
        fname = os.path.join(working_dir, f"{run_key}_agg_val_metrics.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated val metric plot for {run_key}: {e}")
        plt.close()

    # ── final test metrics bar plot -----------------------------------------
    try:
        metric_names = set()
        for tm in test_metrics_runs:
            metric_names.update(tm.keys())
        if not metric_names:
            raise ValueError("No test metrics; skipping test plot.")

        means, ses = [], []
        for m in sorted(metric_names):
            values = [tm.get(m, np.nan) for tm in test_metrics_runs if m in tm]
            if not values:
                means.append(np.nan)
                ses.append(np.nan)
                continue
            arr = np.asarray(values, dtype=float)
            means.append(np.nanmean(arr))
            ses.append(np.nanstd(arr, ddof=1) / np.sqrt(arr.size))

        x = np.arange(len(metric_names))
        plt.figure()
        plt.bar(x, means, yerr=ses, capsize=5, color="tab:blue", alpha=0.7)
        plt.ylim(0, 1)
        plt.xticks(x, sorted(metric_names))
        plt.ylabel("Accuracy")
        plt.title(f"Test Metrics (mean±SE) – {run_key}")
        plt.legend(["Mean ± SE"])
        fname = os.path.join(working_dir, f"{run_key}_agg_test_metrics.png")
        plt.savefig(fname)
        plt.close()

        # print numerical summary
        print(f"\nAggregated Test Metrics for {run_key}:")
        for name, mean, se in zip(sorted(metric_names), means, ses):
            if np.isnan(mean):
                continue
            print(f"  {name}: {mean:.3f} ± {se:.3f}")
    except Exception as e:
        print(f"Error creating aggregated test metric plot for {run_key}: {e}")
        plt.close()

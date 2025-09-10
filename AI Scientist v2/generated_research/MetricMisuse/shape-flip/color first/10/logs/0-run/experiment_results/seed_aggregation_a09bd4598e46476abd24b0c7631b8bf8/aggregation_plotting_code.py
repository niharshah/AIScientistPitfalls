import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------- 0. Setup ------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -----------------------------------------------------------------
# 1. Load all experiment_data dicts that really exist on disk
# -----------------------------------------------------------------
try:
    experiment_data_path_list = [
        "experiments/2025-08-30_23-24-25_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_b6b2d69d037d4880b5db745003827895_proc_1551994/experiment_data.npy",
        "experiments/2025-08-30_23-24-25_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_e73fdde05f6141df8367bf5236717e1e_proc_1551996/experiment_data.npy",
        "experiments/2025-08-30_23-24-25_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_21d1a213ca594508abb5295315d06410_proc_1551995/experiment_data.npy",
    ]
    all_experiment_data = []
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        if os.path.isfile(full_path):
            try:
                ed = np.load(full_path, allow_pickle=True).item()
                all_experiment_data.append(ed)
            except Exception as ie:
                print(f"Could not load {full_path}: {ie}")
        else:
            print(f"Missing file: {full_path}")
    if len(all_experiment_data) == 0:
        raise RuntimeError("No experiment_data files could be loaded.")
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# -----------------------------------------------------------------
# 2. Aggregate metrics for SPR_BENCH
# -----------------------------------------------------------------
spr_runs = [ed["SPR_BENCH"] for ed in all_experiment_data if "SPR_BENCH" in ed]

if len(spr_runs) == 0:
    print("No SPR_BENCH data available to aggregate.")
else:
    # Helper to stack lists of scalars into 2-D array [runs, epochs]
    def stack_metric(key1, key2):
        arrs = []
        for run in spr_runs:
            try:
                arrs.append(np.array(run[key1][key2], dtype=float))
            except Exception:
                return None
        # Pad shorter runs with NaN so we can stack safely
        max_len = max(len(a) for a in arrs)
        padded = [
            np.pad(a, (0, max_len - len(a)), constant_values=np.nan) for a in arrs
        ]
        return np.vstack(padded)

    train_losses = stack_metric("losses", "train")
    val_losses = stack_metric("losses", "val")

    val_cwa = stack_metric("metrics", "val")
    if val_cwa is not None:
        # metrics['val'] is list of dicts; rebuild array
        tmp = []
        for run in spr_runs:
            tmp.append([m["cwa"] for m in run["metrics"]["val"]])
        val_cwa = stack_metric("metrics", "val")  # placeholder will be overwritten
        val_cwa = np.vstack(tmp)
    # same for swa and cpxwa
    tmp_swa, tmp_cpx = [], []
    for run in spr_runs:
        if "metrics" in run:
            tmp_swa.append([m["swa"] for m in run["metrics"]["val"]])
            tmp_cpx.append([m["cpxwa"] for m in run["metrics"]["val"]])
    val_swa = np.vstack(tmp_swa) if tmp_swa else None
    val_cpx = np.vstack(tmp_cpx) if tmp_cpx else None

    # Test metrics
    test_metrics = {}
    for m in ["cwa", "swa", "cpxwa"]:
        vals = []
        for run in spr_runs:
            try:
                vals.append(float(run["metrics"]["test"][m]))
            except Exception:
                continue
        test_metrics[m] = np.array(vals) if vals else None

    # Number of runs
    n_runs = len(spr_runs)

    # -----------------------------------------------------------------
    # 3. Aggregated Loss Curves
    # -----------------------------------------------------------------
    try:
        if train_losses is not None and val_losses is not None:
            epochs = np.arange(train_losses.shape[1]) + 1
            train_mean = np.nanmean(train_losses, axis=0)
            val_mean = np.nanmean(val_losses, axis=0)
            train_sem = np.nanstd(train_losses, axis=0, ddof=1) / np.sqrt(n_runs)
            val_sem = np.nanstd(val_losses, axis=0, ddof=1) / np.sqrt(n_runs)

            plt.figure()
            plt.plot(epochs, train_mean, label="Train Loss (mean)")
            plt.fill_between(
                epochs,
                train_mean - train_sem,
                train_mean + train_sem,
                alpha=0.3,
                label="Train ±1 SEM",
            )
            plt.plot(epochs, val_mean, label="Val Loss (mean)")
            plt.fill_between(
                epochs,
                val_mean - val_sem,
                val_mean + val_sem,
                alpha=0.3,
                label="Val ±1 SEM",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title("SPR_BENCH Aggregated Loss Curves\nMean ±1 SEM across runs")
            plt.legend()
            fname = os.path.join(working_dir, "spr_bench_agg_loss_curves.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss curves: {e}")
        plt.close()

    # -----------------------------------------------------------------
    # 4. Aggregated Validation Metric Curves
    # -----------------------------------------------------------------
    try:
        if val_cwa is not None and val_swa is not None and val_cpx is not None:
            max_ep = val_cwa.shape[1]
            epochs = np.arange(max_ep) + 1

            # helper
            def mean_sem(arr):
                return np.nanmean(arr, axis=0), np.nanstd(
                    arr, axis=0, ddof=1
                ) / np.sqrt(n_runs)

            cwa_m, cwa_s = mean_sem(val_cwa)
            swa_m, swa_s = mean_sem(val_swa)
            cpx_m, cpx_s = mean_sem(val_cpx)

            plt.figure()
            for m, s, lbl, col in [
                (cwa_m, cwa_s, "CWA", "tab:blue"),
                (swa_m, swa_s, "SWA", "tab:orange"),
                (cpx_m, cpx_s, "CpxWA", "tab:green"),
            ]:
                plt.plot(epochs, m, label=f"{lbl} (mean)", color=col)
                plt.fill_between(
                    epochs, m - s, m + s, alpha=0.25, color=col, label=f"{lbl} ±1 SEM"
                )
            plt.xlabel("Epoch")
            plt.ylabel("Weighted Accuracy")
            plt.title(
                "SPR_BENCH Aggregated Validation Metrics\nMean ±1 SEM across runs"
            )
            plt.legend(ncol=2, fontsize="small")
            fname = os.path.join(working_dir, "spr_bench_agg_val_metrics.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated validation curves: {e}")
        plt.close()

    # -----------------------------------------------------------------
    # 5. Aggregated Test Metric Bar Chart
    # -----------------------------------------------------------------
    try:
        avail_keys = [k for k, v in test_metrics.items() if v is not None and len(v)]
        if avail_keys:
            means = [np.mean(test_metrics[k]) for k in avail_keys]
            sems = [
                np.std(test_metrics[k], ddof=1) / np.sqrt(len(test_metrics[k]))
                for k in avail_keys
            ]
            plt.figure()
            x = np.arange(len(avail_keys))
            plt.bar(
                x,
                means,
                yerr=sems,
                capsize=5,
                color=["tab:blue", "tab:orange", "tab:green"][: len(avail_keys)],
            )
            plt.xticks(x, avail_keys)
            plt.ylabel("Score")
            plt.ylim(0, 1.05)
            plt.title("SPR_BENCH Aggregated Test Metrics\nMean ±1 SEM across runs")
            for xi, m in zip(x, means):
                plt.text(xi, m + 0.02, f"{m:.2f}", ha="center")
            fname = os.path.join(working_dir, "spr_bench_agg_test_metrics.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated test metric bar chart: {e}")
        plt.close()

    # -----------------------------------------------------------------
    # 6. Print numeric summary
    # -----------------------------------------------------------------
    try:
        summary = "Aggregated TEST metrics (mean ±1 SEM) -> "
        parts = []
        for k in ["cwa", "swa", "cpxwa"]:
            if test_metrics.get(k) is not None and len(test_metrics[k]):
                mean = np.mean(test_metrics[k])
                sem = np.std(test_metrics[k], ddof=1) / np.sqrt(len(test_metrics[k]))
                parts.append(f"{k.upper()}: {mean:.3f}±{sem:.3f}")
        summary += ", ".join(parts)
        print(summary)
    except Exception as e:
        print(f"Error printing summary: {e}")

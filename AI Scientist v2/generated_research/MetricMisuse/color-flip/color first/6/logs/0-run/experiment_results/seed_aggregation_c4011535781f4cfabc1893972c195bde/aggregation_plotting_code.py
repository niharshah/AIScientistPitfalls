import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# Paths supplied in "Experiment Data Path" section
experiment_data_path_list = [
    "experiments/2025-08-31_03-29-17_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_e78776b6595b4902b1f020d9aaa15f6a_proc_1695463/experiment_data.npy",
    "experiments/2025-08-31_03-29-17_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_ba21b6fbe17b4603bcdce328bd947429_proc_1695464/experiment_data.npy",
    "experiments/2025-08-31_03-29-17_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_9eeb3c9d590940ce9c488b40ebe4c2bf_proc_1695462/experiment_data.npy",
]

# ------------------------------------------------------------------
# Helper containers
train_loss_by_epoch = defaultdict(list)
val_loss_by_epoch = defaultdict(list)
metric_by_epoch = defaultdict(lambda: defaultdict(list))  # epoch -> metric_name -> list
final_metrics = defaultdict(list)  # metric_name -> list

loaded_runs = 0
for p in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp = np.load(full_path, allow_pickle=True).item()
        data = exp.get("dual_channel", {})
        losses = data.get("losses", {})
        metrics_val = data.get("metrics", {}).get("val", [])
        # Losses
        for ep, l in losses.get("train", []):
            train_loss_by_epoch[ep].append(l)
        for ep, l in losses.get("val", []):
            val_loss_by_epoch[ep].append(l)
        # Validation metrics
        for ep, mdict in metrics_val:
            for mname, mval in mdict.items():
                metric_by_epoch[ep][mname].append(mval)
        # Final metrics (last val entry if available)
        if metrics_val:
            _, last_dict = metrics_val[-1]
            for mname, mval in last_dict.items():
                final_metrics[mname].append(mval)
        loaded_runs += 1
    except Exception as e:
        print(f"Error loading {p}: {e}")

if loaded_runs == 0:
    print("No experiment data to visualize.")
    exit()


# ------------------------------------------------------------------
def mean_sem(values):
    arr = np.asarray(values, dtype=float)
    return arr.mean(), arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0


# ------------------------------------------------------------------
# 1) Aggregate loss curve -------------------------------------------------
try:
    if train_loss_by_epoch and val_loss_by_epoch:
        epochs = sorted(set(train_loss_by_epoch.keys()) | set(val_loss_by_epoch.keys()))
        tr_mean, tr_sem, va_mean, va_sem = [], [], [], []
        for ep in epochs:
            m, s = mean_sem(train_loss_by_epoch.get(ep, []))
            tr_mean.append(m)
            tr_sem.append(s)
            m, s = mean_sem(val_loss_by_epoch.get(ep, []))
            va_mean.append(m)
            va_sem.append(s)
        plt.figure()
        plt.plot(epochs, tr_mean, label="Train Mean", color="steelblue")
        plt.fill_between(
            epochs,
            np.array(tr_mean) - np.array(tr_sem),
            np.array(tr_mean) + np.array(tr_sem),
            color="steelblue",
            alpha=0.3,
            label="Train SEM",
        )
        plt.plot(epochs, va_mean, label="Val Mean", color="darkorange")
        plt.fill_between(
            epochs,
            np.array(va_mean) - np.array(va_sem),
            np.array(va_mean) + np.array(va_sem),
            color="darkorange",
            alpha=0.3,
            label="Val SEM",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Aggregated Loss Curve\nMean ± SEM across runs")
        plt.legend()
        fname = "dual_channel_loss_curve_aggregated_SPR_BENCH.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
except Exception as e:
    print(f"Error creating aggregated loss curve: {e}")
    plt.close()

# 2) Aggregate validation metric curves ----------------------------------
try:
    if metric_by_epoch:
        epochs = sorted(metric_by_epoch.keys())
        metric_names = set()
        for ep in epochs:
            metric_names.update(metric_by_epoch[ep].keys())
        colors = {"CWA": "tab:green", "SWA": "tab:red", "PCWA": "tab:purple"}
        plt.figure()
        for mname in sorted(metric_names):
            means, sems = [], []
            for ep in epochs:
                m, s = mean_sem(metric_by_epoch[ep].get(mname, []))
                means.append(m)
                sems.append(s)
            plt.plot(
                epochs, means, label=f"{mname} Mean", color=colors.get(mname, None)
            )
            plt.fill_between(
                epochs,
                np.array(means) - np.array(sems),
                np.array(means) + np.array(sems),
                alpha=0.25,
                color=colors.get(mname, None),
            )
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR_BENCH Aggregated Validation Metrics\nMean ± SEM across runs")
        plt.legend()
        fname = "dual_channel_metric_curves_aggregated_SPR_BENCH.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
except Exception as e:
    print(f"Error creating aggregated metric curves: {e}")
    plt.close()

# 3) Final metric bar chart with error bars ------------------------------
try:
    if final_metrics:
        metric_names = sorted(final_metrics.keys())
        means = [np.mean(final_metrics[m]) for m in metric_names]
        sems = [
            (
                np.std(final_metrics[m], ddof=1) / np.sqrt(len(final_metrics[m]))
                if len(final_metrics[m]) > 1
                else 0.0
            )
            for m in metric_names
        ]
        plt.figure()
        bar_pos = np.arange(len(metric_names))
        plt.bar(
            bar_pos,
            means,
            yerr=sems,
            capsize=5,
            color=["steelblue", "salmon", "seagreen"][: len(metric_names)],
        )
        plt.xticks(bar_pos, metric_names)
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title("SPR_BENCH Final Validation Metrics\nMean ± SEM across runs")
        fname = "dual_channel_final_val_metrics_aggregated_SPR_BENCH.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
        # Print to console
        print("Aggregated Final Metrics (mean ± sem):")
        for n, m, s in zip(metric_names, means, sems):
            print(f"  {n}: {m:.4f} ± {s:.4f}")
except Exception as e:
    print(f"Error creating aggregated final metric bar chart: {e}")
    plt.close()

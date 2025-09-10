import matplotlib.pyplot as plt
import numpy as np
import os

# -------- setup --------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load all experiment_data dicts --------
experiment_data_path_list = [
    "None/experiment_data.npy",
    "experiments/2025-08-31_02-26-44_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_ab8cdb33a86d420f96e59dfa244aac7f_proc_1608753/experiment_data.npy",
    "experiments/2025-08-31_02-26-44_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_0f4c21b5f0b648b482824b0041b28b16_proc_1608752/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        ed = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(ed)
    except Exception as e:
        print(f"Error loading {p}: {e}")

if not all_experiment_data:
    print("No experiment data available across any provided path.")
    quit()

# -------- build aggregated containers --------
# structure: agg[dname][tag][run_index] = dict with losses/metrics
agg = {}
for run_idx, run_data in enumerate(all_experiment_data):
    for tag, tag_dict in run_data.items():
        for dname, ds_dict in tag_dict.items():
            if dname not in agg:
                agg[dname] = {}
            if tag not in agg[dname]:
                agg[dname][tag] = []
            agg[dname][tag].append(ds_dict)  # keep full structure per run


def stack_and_trim(list_of_lists):
    """Stack list of 1D lists (different lengths allowed) into 2D array [runs, epochs] trimmed to min length."""
    min_len = min(len(x) for x in list_of_lists)
    arr = np.array([x[:min_len] for x in list_of_lists], dtype=float)
    return arr, np.arange(1, min_len + 1)


# --------------- plotting ---------------
for dname, tag_dict in agg.items():

    # ===== LOSS CURVES =====
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for tag, runs in tag_dict.items():
            # gather per-run arrays
            train_runs = [r["losses"]["train"] for r in runs if "losses" in r]
            val_runs = [r["losses"]["val"] for r in runs if "losses" in r]
            if not train_runs or not val_runs:
                continue
            train_arr, epochs = stack_and_trim(train_runs)
            val_arr, _ = stack_and_trim(val_runs)

            # compute statistics
            train_mean = train_arr.mean(axis=0)
            val_mean = val_arr.mean(axis=0)
            if train_arr.shape[0] > 1:
                train_sem = train_arr.std(axis=0, ddof=1) / np.sqrt(train_arr.shape[0])
                val_sem = val_arr.std(axis=0, ddof=1) / np.sqrt(val_arr.shape[0])
                axes[0].fill_between(
                    epochs, train_mean - train_sem, train_mean + train_sem, alpha=0.2
                )
                axes[1].fill_between(
                    epochs, val_mean - val_sem, val_mean + val_sem, alpha=0.2
                )
            axes[0].plot(epochs, train_mean, label=tag)
            axes[1].plot(epochs, val_mean, label=tag)

        for ax, title in zip(axes, ["Train Loss", "Validation Loss"]):
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Cross-Entropy")
            ax.legend()
        fig.suptitle(f"{dname} Loss Curves (Mean ± SEM across runs)")
        fname = os.path.join(working_dir, f"{dname}_loss_mean_sem.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss curves for {dname}: {e}")
        plt.close()

    # ===== VALIDATION METRICS =====
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        metric_keys = ["cwa", "swa", "cva"]
        metric_titles = [
            "Color-Weighted Acc.",
            "Shape-Weighted Acc.",
            "Composite Variety Acc.",
        ]
        for tag, runs in tag_dict.items():
            val_metrics_runs = {k: [] for k in metric_keys}
            for r in runs:
                if "metrics" not in r or "val" not in r["metrics"]:
                    continue
                for k in metric_keys:
                    vals = [m[k] for m in r["metrics"]["val"]]
                    val_metrics_runs[k].append(vals)
            if not all(val_metrics_runs[k] for k in metric_keys):
                continue
            # make statistics curve for each metric
            for idx, k in enumerate(metric_keys):
                arr, epochs = stack_and_trim(val_metrics_runs[k])
                mean = arr.mean(axis=0)
                axes[idx].plot(epochs, mean, label=tag)
                if arr.shape[0] > 1:
                    sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
                    axes[idx].fill_between(epochs, mean - sem, mean + sem, alpha=0.2)

        for ax, ttl in zip(axes, metric_titles):
            ax.set_title(ttl)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.legend()
        fig.suptitle(f"{dname} Validation Metrics (Mean ± SEM across runs)")
        fname = os.path.join(working_dir, f"{dname}_val_metrics_mean_sem.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated validation metrics for {dname}: {e}")
        plt.close()

    # ===== TEST METRIC BAR CHART =====
    try:
        metric_keys = ["cwa", "swa", "cva"]
        metric_titles = ["CWA", "SWA", "CVA"]

        # collect per-tag arrays
        bar_data = {tag: {k: [] for k in metric_keys} for tag in tag_dict}
        for tag, runs in tag_dict.items():
            for r in runs:
                tm = r.get("metrics", {}).get("test", {})
                for k in metric_keys:
                    if k in tm:
                        bar_data[tag][k].append(tm[k])

        # keep only tags with at least one recorded metric
        bar_data = {t: v for t, v in bar_data.items() if any(v[k] for k in metric_keys)}
        if bar_data:
            tags_sorted = sorted(bar_data.keys())
            indices = np.arange(len(tags_sorted))
            width = 0.25

            plt.figure(figsize=(10, 5))
            for i, k in enumerate(metric_keys):
                means = [
                    np.mean(bar_data[t][k]) if bar_data[t][k] else np.nan
                    for t in tags_sorted
                ]
                sems = [
                    (
                        np.std(bar_data[t][k], ddof=1) / np.sqrt(len(bar_data[t][k]))
                        if len(bar_data[t][k]) > 1
                        else 0
                    )
                    for t in tags_sorted
                ]
                plt.bar(
                    indices + (i - 1) * width,
                    means,
                    width,
                    yerr=sems,
                    label=metric_titles[i],
                    capsize=4,
                )

            plt.xticks(indices, tags_sorted, rotation=45, ha="right")
            plt.ylabel("Accuracy")
            plt.title(f"{dname} Test Metrics (Mean ± SEM across runs)")
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{dname}_test_metrics_mean_sem.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated test metrics for {dname}: {e}")
        plt.close()

# -------- print aggregated test metrics --------
print("\nAggregated test-set performance (mean ± sem):")
for dname, tag_dict in agg.items():
    for tag, runs in tag_dict.items():
        vals = {k: [] for k in ["cwa", "swa", "cva"]}
        for r in runs:
            tm = r.get("metrics", {}).get("test", {})
            for k in vals:
                if k in tm:
                    vals[k].append(tm[k])
        if any(vals[k] for k in vals):
            out = []
            for k in vals:
                if vals[k]:
                    mean = np.mean(vals[k])
                    sem = (
                        np.std(vals[k], ddof=1) / np.sqrt(len(vals[k]))
                        if len(vals[k]) > 1
                        else 0
                    )
                    out.append(f"{k.upper()}={mean:.4f}±{sem:.4f}")
            print(f"{dname} | {tag}: " + ", ".join(out))

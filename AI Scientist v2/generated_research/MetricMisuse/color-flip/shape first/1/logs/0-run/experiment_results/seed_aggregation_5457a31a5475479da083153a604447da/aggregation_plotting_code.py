import matplotlib.pyplot as plt
import numpy as np
import os

# mandatory working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------- load all experiment files -----------------
experiment_data_path_list = [
    "experiments/2025-08-15_22-24-43_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_c53ede312b8b4af7942f0a4209d14bd5_proc_2983499/experiment_data.npy",
    "experiments/2025-08-15_22-24-43_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_7405fcb2acf44d2f98ecf2659de8a4fc_proc_2983502/experiment_data.npy",
    "experiments/2025-08-15_22-24-43_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_b49cc689da4745b1a3b001353789e01d_proc_2983501/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp)
    except Exception as e:
        print(f"Error loading {p}: {e}")

num_runs = len(all_experiment_data)
if num_runs == 0:
    print("No experiment data could be loaded – aborting plotting.")
    exit()


# ---------------- helper --------------------
def safe_get(dic, *keys):
    for k in keys:
        dic = dic.get(k, {})
    return dic if dic else None


def aggregate_across_runs(key_chain):
    """Return list of np.arrays (one per run) following the key_chain"""
    series_list = []
    for exp in all_experiment_data:
        val = safe_get(exp, *key_chain)
        if isinstance(val, (list, tuple)):
            val = np.asarray(val, dtype=np.float32)
        if val is not None and val.size:
            series_list.append(val)
    return series_list  # may be []


# ---------------- iterate over datasets ------------------
# collect union of dataset names
dataset_names = set()
for exp in all_experiment_data:
    dataset_names.update(exp.keys())

for dname in dataset_names:
    # ------------------ aggregated loss curves ---------------
    try:
        tags = ["pretrain", "train", "val"]
        plotted_any = False
        plt.figure()
        for tag in tags:
            series_per_run = aggregate_across_runs([dname, "losses", tag])
            if len(series_per_run) >= 1:
                # align length
                min_len = min(len(s) for s in series_per_run)
                trimmed = np.stack([s[:min_len] for s in series_per_run], axis=0)
                mean = trimmed.mean(axis=0)
                se = trimmed.std(axis=0, ddof=1) / np.sqrt(trimmed.shape[0])
                epochs = np.arange(1, min_len + 1)
                plt.plot(epochs, mean, label=f"{tag} mean")
                plt.fill_between(epochs, mean - se, mean + se, alpha=0.25)
                plotted_any = True
        if plotted_any:
            plt.title(
                f"{dname} Aggregated Loss Curves (mean ± SE)\nAcross {num_runs} runs"
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_aggregate_loss_curves.png")
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dname}: {e}")
        plt.close()

    # ------------------ aggregated metric curves -------------
    try:
        metrics = ["SWA", "CWA", "SCHM"]
        plotted_any = False
        plt.figure()
        for metric in metrics:
            series_per_run = aggregate_across_runs([dname, "metrics", metric])
            if len(series_per_run) >= 1:
                min_len = min(len(s) for s in series_per_run)
                trimmed = np.stack([s[:min_len] for s in series_per_run], axis=0)
                mean = trimmed.mean(axis=0)
                se = trimmed.std(axis=0, ddof=1) / np.sqrt(trimmed.shape[0])
                epochs = np.arange(1, min_len + 1)
                plt.plot(epochs, mean, label=f"{metric} mean")
                plt.fill_between(epochs, mean - se, mean + se, alpha=0.25)
                plotted_any = True
        if plotted_any:
            plt.title(
                f"{dname} Aggregated Validation Metrics (mean ± SE)\nAcross {num_runs} runs"
            )
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_aggregate_metric_curves.png")
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated metric plot for {dname}: {e}")
        plt.close()

    # --------------- print final aggregated metrics ----------
    final_stats = {}
    for metric in ["SWA", "CWA", "SCHM"]:
        final_vals = []
        for exp in all_experiment_data:
            vals = safe_get(exp, dname, "metrics", metric)
            if vals:
                final_vals.append(vals[-1])
        if final_vals:
            arr = np.asarray(final_vals, dtype=np.float32)
            final_stats[metric] = (arr.mean(), arr.std(ddof=1))
    if final_stats:
        print(f"{dname} final epoch metrics (mean ± std over {num_runs} runs):")
        for m, (mu, sd) in final_stats.items():
            print(f"  {m}: {mu:.4f} ± {sd:.4f}")

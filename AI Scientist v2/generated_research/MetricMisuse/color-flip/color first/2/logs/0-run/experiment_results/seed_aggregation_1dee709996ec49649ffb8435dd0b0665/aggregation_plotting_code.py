import matplotlib.pyplot as plt
import numpy as np
import os

# -----------------------------------------------------------
# basic set-up
# -----------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -----------------------------------------------------------
# paths to all experiment_data.npy files (provided)
# -----------------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-31_02-26-55_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_8ca1511a69474a2ca556f4c4734bcb9d_proc_1599545/experiment_data.npy",
    "experiments/2025-08-31_02-26-55_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_93e33804bbe14ad3a4a4d6a05dde8736_proc_1599546/experiment_data.npy",
    "experiments/2025-08-31_02-26-55_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_695cfa492bad484186684ba7641ef36a_proc_1599553/experiment_data.npy",
]

# -----------------------------------------------------------
# load all runs
# -----------------------------------------------------------
all_experiment_data = []
try:
    ai_root = os.getenv("AI_SCIENTIST_ROOT", "")
    for p in experiment_data_path_list:
        full_p = os.path.join(ai_root, p) if not os.path.isabs(p) else p
        all_experiment_data.append(np.load(full_p, allow_pickle=True).item())
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []


# -----------------------------------------------------------
# helper to build padded matrix
# -----------------------------------------------------------
def build_padded_matrix(list_of_lists, fill_val=np.nan):
    max_len = max(len(x) for x in list_of_lists)
    mat = np.full((len(list_of_lists), max_len), fill_val, dtype=float)
    for i, seq in enumerate(list_of_lists):
        mat[i, : len(seq)] = seq
    return mat


# -----------------------------------------------------------
# iterate over datasets
# -----------------------------------------------------------
dataset_names = set()
for ed in all_experiment_data:
    dataset_names.update(ed.keys())
if not dataset_names:
    print("No datasets found in experiment_data.")

for dname in dataset_names:
    # gather per-run data
    train_losses_runs, val_losses_runs = [], []
    val_metrics_runs = {k: [] for k in ["CWA", "SWA", "GCWA"]}
    test_metrics_vals = {k: [] for k in ["CWA", "SWA", "GCWA"]}

    for ed in all_experiment_data:
        ds = ed.get(dname, {})
        # losses
        train_losses_runs.append(ds.get("losses", {}).get("train", []))
        val_losses_runs.append(ds.get("losses", {}).get("val", []))
        # epoch-wise val metrics
        vmetrics = ds.get("metrics", {}).get("val", [])
        for m_name in ["CWA", "SWA", "GCWA"]:
            val_metrics_runs[m_name].append([m.get(m_name, np.nan) for m in vmetrics])
        # final test metrics
        tmetrics = ds.get("metrics", {}).get("test", {})
        for m_name in ["CWA", "SWA", "GCWA"]:
            if m_name in tmetrics:
                test_metrics_vals[m_name].append(tmetrics[m_name])

    # ------- LOSS CURVES ---------------------------------------------------
    try:
        train_mat = build_padded_matrix(train_losses_runs)
        val_mat = build_padded_matrix(val_losses_runs)

        epochs = np.arange(train_mat.shape[1]) + 1
        mean_train = np.nanmean(train_mat, axis=0)
        mean_val = np.nanmean(val_mat, axis=0)
        stderr_train = np.nanstd(train_mat, axis=0, ddof=0) / np.sqrt(
            np.sum(~np.isnan(train_mat), axis=0)
        )
        stderr_val = np.nanstd(val_mat, axis=0, ddof=0) / np.sqrt(
            np.sum(~np.isnan(val_mat), axis=0)
        )

        plt.figure()
        plt.plot(epochs, mean_train, label="Train Loss (mean)")
        plt.fill_between(
            epochs, mean_train - stderr_train, mean_train + stderr_train, alpha=0.3
        )
        plt.plot(epochs, mean_val, label="Val Loss (mean)")
        plt.fill_between(
            epochs, mean_val - stderr_val, mean_val + stderr_val, alpha=0.3
        )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dname} Aggregated Loss Curves\nMean ± Standard Error across runs")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname}_aggregated_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dname}: {e}")
        plt.close()

    # ------- VALIDATION METRICS CURVES ------------------------------------
    try:
        plt.figure()
        for m_name, runs in val_metrics_runs.items():
            if not any(len(r) for r in runs):
                continue
            mat = build_padded_matrix(runs)
            epochs = np.arange(mat.shape[1]) + 1
            mean_v = np.nanmean(mat, axis=0)
            stderr_v = np.nanstd(mat, axis=0, ddof=0) / np.sqrt(
                np.sum(~np.isnan(mat), axis=0)
            )
            plt.plot(epochs, mean_v, label=f"{m_name} (mean)")
            plt.fill_between(epochs, mean_v - stderr_v, mean_v + stderr_v, alpha=0.3)
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(f"{dname} Validation Metrics\nMean ± Standard Error across runs")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname}_aggregated_val_metrics.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated val metrics plot for {dname}: {e}")
        plt.close()

    # ------- TEST METRICS BAR CHART ---------------------------------------
    try:
        labels, means, stderrs = [], [], []
        for m_name, vals in test_metrics_vals.items():
            if vals:  # only include if present
                labels.append(m_name)
                means.append(np.mean(vals))
                stderrs.append(np.std(vals, ddof=0) / np.sqrt(len(vals)))
        if labels:
            x = np.arange(len(labels))
            plt.figure()
            plt.bar(
                x,
                means,
                yerr=stderrs,
                capsize=5,
                color=plt.cm.tab10.colors[: len(labels)],
            )
            plt.xticks(x, labels)
            plt.ylim(0, 1)
            plt.ylabel("Score")
            plt.title(
                f"{dname} Test Metrics\nBar = Mean, Error = Standard Error (n={len(vals)})"
            )
            fname = os.path.join(
                working_dir, f"{dname}_aggregated_test_metrics_bar.png"
            )
            plt.savefig(fname)
            plt.close()
            # print aggregated numbers
            print(f"Aggregated test metrics for {dname}:")
            for l, m, se in zip(labels, means, stderrs):
                print(f"  {l}: {m:.3f} ± {se:.3f}")
        else:
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated test metric bar chart for {dname}: {e}")
        plt.close()

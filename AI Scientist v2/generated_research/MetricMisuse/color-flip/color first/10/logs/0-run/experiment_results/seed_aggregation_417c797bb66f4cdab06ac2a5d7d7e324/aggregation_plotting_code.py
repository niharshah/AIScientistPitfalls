import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------ #
# Basic setup
# ------------------------------------------------------------------ #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------ #
# Load every experiment result
# ------------------------------------------------------------------ #
try:
    experiment_data_path_list = [
        "experiments/2025-08-31_14-12-13_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_748b0a4076e041fe8e209a1755bd446f_proc_1726539/experiment_data.npy",
        "experiments/2025-08-31_14-12-13_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_16a435c724444e22a993563045411103_proc_1726540/experiment_data.npy",
        "experiments/2025-08-31_14-12-13_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_b4be1286cf56420e96e1703b31ca4671_proc_1726538/experiment_data.npy",
    ]
    all_experiment_data = []
    for pth in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), pth)
        all_experiment_data.append(np.load(full_path, allow_pickle=True).item())
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []


# ------------------------------------------------------------------ #
# Helper functions
# ------------------------------------------------------------------ #
def collect_series(all_runs, key_path):
    """returns dict[k] -> list_of_arrays  (one array per run, maybe length diff)"""
    series = {}
    for run in all_runs:
        bench = run.get("num_clusters_k", {}).get("SPR_BENCH", {})
        for k, kd in bench.items():
            tmp = kd
            for kp in key_path:
                tmp = tmp.get(kp, [])
            series.setdefault(k, []).append(np.asarray(tmp, dtype=float))
    return series


def align_and_stat(list_of_arr):
    """Trim to shortest length, return mean and stderr along axis 0."""
    if not list_of_arr:
        return np.array([]), np.array([])
    min_len = min(len(a) for a in list_of_arr)
    clipped = np.stack([a[:min_len] for a in list_of_arr], axis=0)  # shape (runs, T)
    mean = np.nanmean(clipped, axis=0)
    stderr = np.nanstd(clipped, axis=0, ddof=1) / np.sqrt(clipped.shape[0])
    return mean, stderr


def collect_scalar(all_runs, key_path):
    out = {}
    for run in all_runs:
        bench = run.get("num_clusters_k", {}).get("SPR_BENCH", {})
        for k, kd in bench.items():
            tmp = kd
            for kp in key_path:
                tmp = tmp.get(kp, np.nan)
            out.setdefault(k, []).append(float(tmp))
    return out


# ------------------------------------------------------------------ #
# Collect curves
# ------------------------------------------------------------------ #
loss_val_series = collect_series(all_experiment_data, ["losses", "val"])
compwa_val_series = collect_series(all_experiment_data, ["metrics", "val_CompWA"])
cwa_final_series = collect_scalar(all_experiment_data, ["metrics", "final_CWA"])
swa_final_series = collect_scalar(all_experiment_data, ["metrics", "final_SWA"])

k_vals = sorted(
    loss_val_series.keys(), key=lambda s: int(s.split("=")[1])
)  # keep numeric order

# ------------------------------------------------------------------ #
# 1) Validation Loss mean ± SE
# ------------------------------------------------------------------ #
try:
    plt.figure()
    for k in k_vals:
        mean, se = align_and_stat(loss_val_series.get(k, []))
        if mean.size == 0:
            continue
        epochs = np.arange(len(mean))
        plt.plot(epochs, mean, label=f"{k} mean")
        plt.fill_between(epochs, mean - se, mean + se, alpha=0.3, label=f"{k} ±SE")
    plt.title(
        "SPR_BENCH Validation Loss (mean ± SE)\nLeft: Ground Truth, Right: Generated Samples"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Binary-Cross-Entropy Loss")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_loss_mean_se.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 2) Validation CompWA mean ± SE
# ------------------------------------------------------------------ #
try:
    plt.figure()
    for k in k_vals:
        mean, se = align_and_stat(compwa_val_series.get(k, []))
        if mean.size == 0:
            continue
        epochs = np.arange(len(mean))
        plt.plot(epochs, mean, label=f"{k} mean")
        plt.fill_between(epochs, mean - se, mean + se, alpha=0.3, label=f"{k} ±SE")
    plt.title(
        "SPR_BENCH Validation CompWA (mean ± SE)\nLeft: Ground Truth, Right: Generated Samples"
    )
    plt.xlabel("Epoch")
    plt.ylabel("CompWA")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_CompWA_mean_se.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated CompWA plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 3) Final CWA Bar chart with error bars
# ------------------------------------------------------------------ #
try:
    plt.figure()
    means = [np.nanmean(cwa_final_series.get(k, [np.nan])) for k in k_vals]
    ses = [
        np.nanstd(cwa_final_series.get(k, [np.nan]), ddof=1)
        / np.sqrt(len(cwa_final_series.get(k, [])))
        for k in k_vals
    ]
    plt.bar(k_vals, means, yerr=ses, capsize=5)
    plt.title("SPR_BENCH Final Color-Weighted-Accuracy (mean ± SE)")
    plt.ylabel("CWA")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_CWA_mean_se.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated CWA bar plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 4) Final SWA Bar chart with error bars
# ------------------------------------------------------------------ #
try:
    plt.figure()
    means = [np.nanmean(swa_final_series.get(k, [np.nan])) for k in k_vals]
    ses = [
        np.nanstd(swa_final_series.get(k, [np.nan]), ddof=1)
        / np.sqrt(len(swa_final_series.get(k, [])))
        for k in k_vals
    ]
    plt.bar(k_vals, means, yerr=ses, capsize=5)
    plt.title("SPR_BENCH Final Shape-Weighted-Accuracy (mean ± SE)")
    plt.ylabel("SWA")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_SWA_mean_se.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated SWA bar plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# Print summary
# ------------------------------------------------------------------ #
print("Aggregated Final Metrics (mean ± SE):")
for k in k_vals:
    cwa_arr = np.array(cwa_final_series.get(k, []), dtype=float)
    swa_arr = np.array(swa_final_series.get(k, []), dtype=float)
    if cwa_arr.size:
        cwa_mean = np.nanmean(cwa_arr)
        cwa_se = np.nanstd(cwa_arr, ddof=1) / np.sqrt(cwa_arr.size)
    else:
        cwa_mean = cwa_se = np.nan
    if swa_arr.size:
        swa_mean = np.nanmean(swa_arr)
        swa_se = np.nanstd(swa_arr, ddof=1) / np.sqrt(swa_arr.size)
    else:
        swa_mean = swa_se = np.nan
    print(f"{k}: CWA={cwa_mean:.4f}±{cwa_se:.4f}, SWA={swa_mean:.4f}±{swa_se:.4f}")

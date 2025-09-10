import matplotlib.pyplot as plt
import numpy as np
import os
import math

# ----------------- paths & working dir -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data_path_list = [
    "experiments/2025-08-31_03-13-24_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_d75eb463b1654b7593ade07763ccba78_proc_1619794/experiment_data.npy",
    "experiments/2025-08-31_03-13-24_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_2958b049061a42d8a34d509737271a99_proc_1619795/experiment_data.npy",
    "experiments/2025-08-31_03-13-24_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_fcb74c59d4784719ba03e071fa4f1ad4_proc_1619792/experiment_data.npy",
]

# ----------------- load all runs -----------------
all_runs = []
try:
    root = os.getenv("AI_SCIENTIST_ROOT", "")
    for p in experiment_data_path_list:
        full_path = os.path.join(root, p)
        run_data = np.load(full_path, allow_pickle=True).item()
        all_runs.append(run_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_runs = []

if not all_runs:
    quit()

# Assume every run contains the same dataset keys; here we use the first run
dataset_names = list(all_runs[0].keys())

for dname in dataset_names:
    # ------------ collect per-run arrays ------------
    train_losses, val_losses = [], []
    val_metrics_dicts = []
    for run in all_runs:
        data = run[dname]
        train_losses.append(np.array(data["losses"]["train"]))
        val_losses.append(np.array(data["losses"]["val"]))
        val_metrics_dicts.append(data["metrics"]["val"])

    # Align to shortest run
    min_len = min(map(len, train_losses))
    train_losses = np.stack([tl[:min_len] for tl in train_losses], axis=0)
    val_losses = np.stack([vl[:min_len] for vl in val_losses], axis=0)
    epochs = np.arange(1, min_len + 1)

    # ------------ helper for mean & sem ------------
    def mean_sem(arr):
        mean = arr.mean(axis=0)
        sem = arr.std(axis=0, ddof=1) / math.sqrt(arr.shape[0])
        return mean, sem

    # ------------ plot loss curves ------------
    try:
        plt.figure()
        m_tr, s_tr = mean_sem(train_losses)
        m_val, s_val = mean_sem(val_losses)
        plt.plot(epochs, m_tr, label="Train Loss (mean)")
        plt.fill_between(epochs, m_tr - s_tr, m_tr + s_tr, alpha=0.3, label="Train SEM")
        plt.plot(epochs, m_val, label="Val Loss (mean)")
        plt.fill_between(
            epochs, m_val - s_val, m_val + s_val, alpha=0.3, label="Val SEM"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dname}: Train & Val Loss (Mean ± SEM) over Runs")
        plt.legend()
        fpath = os.path.join(working_dir, f"{dname}_agg_loss_curve.png")
        plt.savefig(fpath)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dname}: {e}")
        plt.close()

    # ------------ aggregate validation metrics ------------
    # Collect metrics into dict of lists: metric_name -> list of runs (each run shape [epochs])
    metric_names = val_metrics_dicts[0][0].keys()
    metric_stacks = {m: [] for m in metric_names}
    for run_metrics in val_metrics_dicts:
        # run_metrics is list of dicts per epoch
        truncated = run_metrics[:min_len]
        for m in metric_names:
            metric_stacks[m].append(np.array([ep[m] for ep in truncated]))

    # Convert to np.arrays shape [runs, epochs]
    for m in metric_names:
        metric_stacks[m] = np.stack(metric_stacks[m], axis=0)

    # ------------ plot validation metrics ------------
    try:
        plt.figure()
        for m in ["acc", "cwa", "swa", "pcwa"]:
            if m not in metric_stacks:
                continue
            mean_vals, sem_vals = mean_sem(metric_stacks[m])
            plt.plot(epochs, mean_vals, label=f"{m.upper()} (mean)")
            plt.fill_between(
                epochs,
                mean_vals - sem_vals,
                mean_vals + sem_vals,
                alpha=0.3,
                label=f"{m.upper()} SEM",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(f"{dname}: Validation Metrics (Mean ± SEM) over Runs")
        plt.legend()
        fpath = os.path.join(working_dir, f"{dname}_agg_val_metrics.png")
        plt.savefig(fpath)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated metric plot for {dname}: {e}")
        plt.close()

    # ------------ print final epoch aggregated metrics ------------
    try:
        print(f"\nFinal-epoch Validation Metrics (epoch {epochs[-1]}) for {dname}:")
        for m in metric_names:
            if m == "epoch":
                continue
            final_vals = metric_stacks[m][:, -1]
            print(f"  {m}: {final_vals.mean():.4f} ± {final_vals.std(ddof=1):.4f}")
    except Exception as e:
        print(f"Error printing final metrics for {dname}: {e}")

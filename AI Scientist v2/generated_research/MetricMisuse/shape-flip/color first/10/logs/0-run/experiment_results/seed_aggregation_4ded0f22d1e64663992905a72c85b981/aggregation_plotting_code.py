import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
# set up working directory and load data
# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# paths to experiment_data.npy files provided by the platform
# ------------------------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-30_23-24-25_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_8b12169a72d54bb585e3688176004a5d_proc_1557386/experiment_data.npy",
    # The next two lines are placeholders that will fail to load; that’s ok.
    "None/experiment_data.npy",
    "None/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        full_p = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        ed = np.load(full_p, allow_pickle=True).item()
        all_experiment_data.append(ed)
        print(f"Loaded data from {p}")
    except Exception as e:
        print(f"Error loading experiment data from {p}: {e}")

if not all_experiment_data:
    print("No experiment data could be loaded – nothing to plot.")
    exit()

# ------------------------------------------------------------------
# Aggregate across experiments
# ------------------------------------------------------------------
datasets = set.intersection(*[set(d.keys()) for d in all_experiment_data])
aggregated = {
    ds: {"losses": {"train": [], "val": []}, "metrics": {"val": [], "test": []}}
    for ds in datasets
}

for run in all_experiment_data:
    for ds in datasets:
        # losses
        aggregated[ds]["losses"]["train"].append(np.array(run[ds]["losses"]["train"]))
        aggregated[ds]["losses"]["val"].append(np.array(run[ds]["losses"]["val"]))
        # validation metrics (list of dicts per epoch)
        aggregated[ds]["metrics"]["val"].append(run[ds]["metrics"]["val"])
        # test metrics (single dict)
        aggregated[ds]["metrics"]["test"].append(run[ds]["metrics"]["test"])


def pad_to_max(arr_list, pad_val=np.nan):
    max_len = max(len(a) for a in arr_list)
    padded = []
    for a in arr_list:
        if len(a) < max_len:
            pad = np.full(max_len - len(a), pad_val)
            padded.append(np.concatenate([a, pad]))
        else:
            padded.append(a[:max_len])
    return np.vstack(padded)


# ------------------------------------------------------------------
# Plotting per dataset
# ------------------------------------------------------------------
for ds in datasets:
    # ===== 1. Loss curves with mean ± SEM =====
    try:
        train_mat = pad_to_max(aggregated[ds]["losses"]["train"])
        val_mat = pad_to_max(aggregated[ds]["losses"]["val"])
        n_runs = train_mat.shape[0]
        epochs = np.arange(1, train_mat.shape[1] + 1)

        train_mean = np.nanmean(train_mat, axis=0)
        train_sem = np.nanstd(train_mat, axis=0, ddof=1) / np.sqrt(n_runs)
        val_mean = np.nanmean(val_mat, axis=0)
        val_sem = np.nanstd(val_mat, axis=0, ddof=1) / np.sqrt(n_runs)

        plt.figure()
        plt.plot(epochs, train_mean, label="Train Loss", color="tab:blue")
        plt.fill_between(
            epochs,
            train_mean - train_sem,
            train_mean + train_sem,
            color="tab:blue",
            alpha=0.3,
            label="Train ± SEM",
        )
        plt.plot(epochs, val_mean, label="Val Loss", color="tab:orange")
        plt.fill_between(
            epochs,
            val_mean - val_sem,
            val_mean + val_sem,
            color="tab:orange",
            alpha=0.3,
            label="Val ± SEM",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{ds} Mean Loss Curves (n={n_runs})\nShaded: ± Standard Error")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds.lower()}_mean_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss curves for {ds}: {e}")
        plt.close()

    # ===== 2. Validation metric curves (CWA, SWA, CpxWA) =====
    try:
        # collect metric arrays per run
        metric_keys = ["cwa", "swa", "cpxwa"]
        per_metric = {k: [] for k in metric_keys}
        max_len = 0
        for run_list in aggregated[ds]["metrics"]["val"]:
            max_len = max(max_len, len(run_list))
        for k in metric_keys:
            for run_list in aggregated[ds]["metrics"]["val"]:
                vals = np.array([m[k] for m in run_list])
                if len(vals) < max_len:
                    vals = np.concatenate([vals, np.full(max_len - len(vals), np.nan)])
                per_metric[k].append(vals)
            per_metric[k] = np.vstack(per_metric[k])

        epochs = np.arange(1, max_len + 1)
        plt.figure()
        colors = {"cwa": "tab:green", "swa": "tab:red", "cpxwa": "tab:purple"}
        for k in metric_keys:
            mean = np.nanmean(per_metric[k], axis=0)
            sem = np.nanstd(per_metric[k], axis=0, ddof=1) / np.sqrt(
                per_metric[k].shape[0]
            )
            plt.plot(epochs, mean, label=f"{k.upper()} Mean", color=colors[k])
            plt.fill_between(
                epochs,
                mean - sem,
                mean + sem,
                color=colors[k],
                alpha=0.25,
                label=f"{k.upper()} ± SEM",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.title(f"{ds} Validation Metrics (Mean ± SEM, n={per_metric[k].shape[0]})")
        plt.legend(ncol=2, fontsize=8)
        fname = os.path.join(working_dir, f"{ds.lower()}_val_metrics_mean_sem.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated validation metric curves for {ds}: {e}")
        plt.close()

    # ===== 3. Test metric bar chart with error bars =====
    try:
        metric_keys = ["cwa", "swa", "cpxwa"]
        test_values = {k: [] for k in metric_keys}
        for t in aggregated[ds]["metrics"]["test"]:
            for k in metric_keys:
                test_values[k].append(t[k])
        means = [np.mean(test_values[k]) for k in metric_keys]
        sems = [
            np.std(test_values[k], ddof=1) / np.sqrt(len(test_values[k]))
            for k in metric_keys
        ]

        plt.figure()
        x = np.arange(len(metric_keys))
        plt.bar(
            x,
            means,
            yerr=sems,
            capsize=5,
            color=["tab:blue", "tab:orange", "tab:green"],
        )
        plt.xticks(x, [k.upper() for k in metric_keys])
        plt.ylim(0, 1.05)
        plt.ylabel("Score")
        plt.title(
            f"{ds} Test Metrics Across Runs\nBars: Mean, Error: ± SEM (n={len(test_values['cwa'])})"
        )
        for i, v in enumerate(means):
            plt.text(i, v + 0.02, f"{v:.2f}±{sems[i]:.2f}", ha="center")
        fname = os.path.join(working_dir, f"{ds.lower()}_test_metrics_bar_mean_sem.png")
        plt.savefig(fname)
        plt.close()

        print(
            f"{ds}: TEST metrics mean±SEM -> "
            + ", ".join(
                [
                    f"{k.upper()}: {means[i]:.3f}±{sems[i]:.3f}"
                    for i, k in enumerate(metric_keys)
                ]
            )
        )
    except Exception as e:
        print(f"Error creating aggregated test metric bar chart for {ds}: {e}")
        plt.close()

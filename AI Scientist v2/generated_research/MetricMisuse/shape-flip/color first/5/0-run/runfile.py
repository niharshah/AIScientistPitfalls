import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

###############################################################################
# --------------------------- load all experiments ----------------------------
###############################################################################
try:
    experiment_data_path_list = [
        "experiments/2025-08-30_20-55-31_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_4bb6a8bc04f348989d102ec490b4e9cf_proc_1497855/experiment_data.npy",
        "experiments/2025-08-30_20-55-31_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_613dadca68994d989063e7b4ab2110df_proc_1497854/experiment_data.npy",
        "experiments/2025-08-30_20-55-31_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_f36f779507014ac1a13a55f7a65d8911_proc_1497856/experiment_data.npy",
    ]
    all_experiment_data = []
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp_data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# Abort early if nothing was loaded
if not all_experiment_data:
    exit()


###############################################################################
# -------------------------- helper: aggregate runs ---------------------------
###############################################################################
def collect_curve(field_path, variant):
    """
    field_path - list of nested keys e.g. ['losses', 'train']
    returns stacked array (n_runs, n_epochs_aligned)
    """
    curves = []
    for exp in all_experiment_data:
        data = exp["edge_direction"][variant]
        # drill-down
        d = data
        for k in field_path:
            d = d[k]
        curves.append(np.asarray(d, dtype=float))
    min_len = min(len(c) for c in curves)
    curves = np.stack([c[:min_len] for c in curves], axis=0)  # (n_runs, n_epochs)
    return curves


def mean_sem(arr, axis=0):
    mean = arr.mean(axis=axis)
    sem = arr.std(axis=axis, ddof=1) / np.sqrt(arr.shape[axis])
    return mean, sem


variants = ["directed", "undirected"]

###############################################################################
# ----------------------------- Figure 1: Loss --------------------------------
###############################################################################
try:
    plt.figure()
    for v in variants:
        tr_curve = collect_curve(["losses", "train"], v)
        vl_curve = collect_curve(["losses", "val"], v)

        mean_tr, sem_tr = mean_sem(tr_curve)
        mean_vl, sem_vl = mean_sem(vl_curve)

        epochs = np.arange(1, len(mean_tr) + 1)

        plt.plot(epochs, mean_tr, label=f"{v}-train (mean)")
        plt.fill_between(epochs, mean_tr - sem_tr, mean_tr + sem_tr, alpha=0.2)

        plt.plot(epochs, mean_vl, linestyle="--", label=f"{v}-val (mean)")
        plt.fill_between(epochs, mean_vl - sem_vl, mean_vl + sem_vl, alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Aggregated Loss Curves per Variant (synthetic_or_SPR dataset)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "agg_loss_curves_synthetic_or_SPR.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss curve: {e}")
    plt.close()

###############################################################################
# ---------------------------- Figure 2: CmpWA --------------------------------
###############################################################################
try:
    plt.figure()
    for v in variants:
        tr_curve = collect_curve(["metrics", "train", "CmpWA"], v)
        vl_curve = collect_curve(["metrics", "val", "CmpWA"], v)

        mean_tr, sem_tr = mean_sem(tr_curve)
        mean_vl, sem_vl = mean_sem(vl_curve)

        epochs = np.arange(1, len(mean_tr) + 1)

        plt.plot(epochs, mean_tr, label=f"{v}-train (mean)")
        plt.fill_between(epochs, mean_tr - sem_tr, mean_tr + sem_tr, alpha=0.2)

        plt.plot(epochs, mean_vl, linestyle="--", label=f"{v}-val (mean)")
        plt.fill_between(epochs, mean_vl - sem_vl, mean_vl + sem_vl, alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Complexity-Weighted Accuracy")
    plt.title("Aggregated CmpWA Curves per Variant (synthetic_or_SPR dataset)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "agg_CmpWA_curves_synthetic_or_SPR.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated CmpWA curve: {e}")
    plt.close()

###############################################################################
# -------------------------- Figure 3: Test metrics ---------------------------
###############################################################################
try:
    metric_names = ["loss", "CWA", "SWA", "CmpWA"]
    x = np.arange(len(metric_names))
    width = 0.35

    plt.figure()
    for idx, v in enumerate(variants):
        # gather per-run values for each metric
        vals = {m: [] for m in metric_names}
        for exp in all_experiment_data:
            tm = exp["edge_direction"][v]["test_metrics"]
            for m in metric_names:
                vals[m].append(tm[m])
        # compute mean & sem
        means = [np.mean(vals[m]) for m in metric_names]
        sems = [np.std(vals[m], ddof=1) / np.sqrt(len(vals[m])) for m in metric_names]

        plt.bar(x + idx * width, means, width=width, yerr=sems, capsize=4, label=v)

        # print table-style output
        printable = ", ".join(
            [
                f"{m}: {mu:.4f} ± {se:.4f}"
                for m, mu, se in zip(metric_names, means, sems)
            ]
        )
        print(f"Aggregated test metrics for {v}: {printable}")

    plt.xticks(x + width / 2, metric_names)
    plt.ylabel("Score")
    plt.title("Aggregated Test Metrics Comparison (synthetic_or_SPR dataset)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "agg_test_metrics_synthetic_or_SPR.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated test metric bar plot: {e}")
    plt.close()

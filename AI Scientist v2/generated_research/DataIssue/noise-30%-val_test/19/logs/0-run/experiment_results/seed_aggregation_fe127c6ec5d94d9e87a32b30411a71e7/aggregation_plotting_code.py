import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths & data loading ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# List of candidate experiment files (relative to AI_SCIENTIST_ROOT)
experiment_data_path_list = [
    "None/experiment_data.npy",
    "experiments/2025-08-17_18-48-09_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_bddd18583831440382c2306e3de549ee_proc_3336029/experiment_data.npy",
    "experiments/2025-08-17_18-48-09_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_93504be1bf044b04bf53f2fbe54f6b93_proc_3336032/experiment_data.npy",
]

all_experiment_data = []
for rel_path in experiment_data_path_list:
    try:
        abs_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), rel_path)
        data = np.load(abs_path, allow_pickle=True).item()
        all_experiment_data.append(data)
    except Exception as e:
        print(f"Error loading {rel_path}: {e}")

# ---------- aggregate ----------
# Structure: aggregates[dropout][epoch] -> dict of lists for each metric
aggregates = {}
for exp in all_experiment_data:
    logs = exp.get("transformer", {})
    epochs_info = logs.get("epochs", [])  # list of (dropout, epoch_idx)
    tr_loss = logs.get("losses", {}).get("train", [])
    val_loss = logs.get("losses", {}).get("val", [])
    tr_mcc = logs.get("metrics", {}).get("train_MCC", [])
    val_mcc = logs.get("metrics", {}).get("val_MCC", [])
    for i, (dp, ep) in enumerate(epochs_info):
        ag = aggregates.setdefault(dp, {}).setdefault(
            ep, {"tr_loss": [], "val_loss": [], "tr_mcc": [], "val_mcc": []}
        )
        if i < len(tr_loss):
            ag["tr_loss"].append(tr_loss[i])
        if i < len(val_loss):
            ag["val_loss"].append(val_loss[i])
        if i < len(tr_mcc):
            ag["tr_mcc"].append(tr_mcc[i])
        if i < len(val_mcc):
            ag["val_mcc"].append(val_mcc[i])


# Helper to compute mean and sem safely
def mean_sem(lst):
    arr = np.array(lst, dtype=float)
    if arr.size == 0:
        return np.nan, np.nan
    mean = np.nanmean(arr)
    sem = np.nanstd(arr, ddof=1) / np.sqrt(np.count_nonzero(~np.isnan(arr)))
    return mean, sem


# -------------------- 1. aggregated loss curves --------------------
try:
    plt.figure(figsize=(6, 4))
    plotted_any = False
    for dp in sorted(aggregates.keys()):
        epochs = sorted(aggregates[dp].keys())
        means_tr, sems_tr = [], []
        means_val, sems_val = [], []
        for ep in epochs:
            m, s = mean_sem(aggregates[dp][ep]["tr_loss"])
            means_tr.append(m)
            sems_tr.append(s)
            m, s = mean_sem(aggregates[dp][ep]["val_loss"])
            means_val.append(m)
            sems_val.append(s)

        if not np.all(np.isnan(means_tr)):
            plotted_any = True
            means_tr = np.array(means_tr)
            sems_tr = np.array(sems_tr)
            plt.plot(epochs, means_tr, label=f"Train μ dp={dp}")
            plt.fill_between(epochs, means_tr - sems_tr, means_tr + sems_tr, alpha=0.3)
        if not np.all(np.isnan(means_val)):
            plotted_any = True
            means_val = np.array(means_val)
            sems_val = np.array(sems_val)
            plt.plot(epochs, means_val, linestyle="--", label=f"Val μ dp={dp}")
            plt.fill_between(
                epochs, means_val - sems_val, means_val + sems_val, alpha=0.3
            )
    if plotted_any:
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title("Loss Curves (mean ± SEM) — synthetic SPR_BENCH")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "spr_bench_loss_curves_agg.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss curves: {e}")
    plt.close()

# -------------------- 2. aggregated MCC curves --------------------
try:
    plt.figure(figsize=(6, 4))
    plotted_any = False
    for dp in sorted(aggregates.keys()):
        epochs = sorted(aggregates[dp].keys())
        means_tr, sems_tr = [], []
        means_val, sems_val = [], []
        for ep in epochs:
            m, s = mean_sem(aggregates[dp][ep]["tr_mcc"])
            means_tr.append(m)
            sems_tr.append(s)
            m, s = mean_sem(aggregates[dp][ep]["val_mcc"])
            means_val.append(m)
            sems_val.append(s)

        if not np.all(np.isnan(means_tr)):
            plotted_any = True
            means_tr = np.array(means_tr)
            sems_tr = np.array(sems_tr)
            plt.plot(epochs, means_tr, label=f"Train μ dp={dp}")
            plt.fill_between(epochs, means_tr - sems_tr, means_tr + sems_tr, alpha=0.3)
        if not np.all(np.isnan(means_val)):
            plotted_any = True
            means_val = np.array(means_val)
            sems_val = np.array(sems_val)
            plt.plot(epochs, means_val, linestyle="--", label=f"Val μ dp={dp}")
            plt.fill_between(
                epochs, means_val - sems_val, means_val + sems_val, alpha=0.3
            )
    if plotted_any:
        plt.xlabel("Epoch")
        plt.ylabel("MCC")
        plt.title("MCC Curves (mean ± SEM) — synthetic SPR_BENCH")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "spr_bench_mcc_curves_agg.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated MCC curves: {e}")
    plt.close()

# -------------------- 3. final dev MCC bar w/ error bars --------------------
try:
    dps, means, sems = [], [], []
    for dp in sorted(aggregates.keys()):
        # look at the highest epoch number available per run, collect val_mcc
        final_vals = []
        for ep in aggregates[dp]:
            final_vals.extend(aggregates[dp][ep]["val_mcc"])
        if final_vals:
            mu, se = mean_sem(final_vals)
            dps.append(str(dp))
            means.append(mu)
            sems.append(se)
    if dps:
        x = np.arange(len(dps))
        plt.figure(figsize=(5, 4))
        plt.bar(x, means, yerr=sems, capsize=5, color="steelblue", label="Mean ± SEM")
        plt.xticks(x, dps)
        plt.xlabel("Dropout")
        plt.ylabel("Final Dev MCC")
        plt.ylim(0, 1)
        plt.title("Final Dev MCC by Dropout (mean ± SEM) — synthetic SPR_BENCH")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "spr_bench_dev_mcc_bar_agg.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated dev MCC bar chart: {e}")
    plt.close()

import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- set up ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load all experiment files ----------
experiment_data_path_list = [
    "experiments/2025-08-17_18-48-06_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_4446f682f90649ae8a2d17bbd7e675c1_proc_3330951/experiment_data.npy",
    "experiments/2025-08-17_18-48-06_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_28ad3b8fb6414163b032cec557308c09_proc_3330950/experiment_data.npy",
    "experiments/2025-08-17_18-48-06_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_f2742b7adc764989ad02a6a9becebe80_proc_3330949/experiment_data.npy",
]

all_runs = []
try:
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        run_data = np.load(full_path, allow_pickle=True).item()
        all_runs.append(run_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_runs = []


# Helper to collect data safely
def safe_get(dic, keys, default=None):
    cur = dic
    for k in keys:
        if k not in cur:
            return default
        cur = cur[k]
    return cur


# ---------- aggregate SPR_BENCH ----------
agg = {}  # structure: agg[wd]['epochs'] , 'losses_train' list-of-arrays, etc.
for run in all_runs:
    spr = safe_get(run, ["weight_decay", "SPR_BENCH"])
    if spr is None:
        continue
    for wd, info in spr.items():
        wd_dict = agg.setdefault(
            wd,
            {
                "epochs": None,
                "loss_train": [],
                "loss_val": [],
                "f1_train": [],
                "f1_val": [],
                "test_f1": [],
            },
        )
        if wd_dict["epochs"] is None:
            wd_dict["epochs"] = np.array(info["epochs"])
        # collect curves
        wd_dict["loss_train"].append(np.array(info["losses"]["train"]))
        wd_dict["loss_val"].append(np.array(info["losses"]["val"]))
        wd_dict["f1_train"].append(np.array(info["metrics"]["train"]))
        wd_dict["f1_val"].append(np.array(info["metrics"]["val"]))
        wd_dict["test_f1"].append(info["test_f1"])

# sort weight decays numerically
wds_sorted = sorted(agg.keys(), key=lambda x: float(x))


# ---------- plotting helpers ----------
def mean_stderr(arr_list):
    stacked = np.stack(arr_list, axis=0)
    mean = stacked.mean(axis=0)
    stderr = stacked.std(axis=0) / np.sqrt(stacked.shape[0])
    return mean, stderr


# ========== 1. Aggregated Loss Curves ==========
try:
    plt.figure()
    for wd in wds_sorted:
        d = agg[wd]
        mean_tr, err_tr = mean_stderr(d["loss_train"])
        mean_val, err_val = mean_stderr(d["loss_val"])
        epochs = d["epochs"]
        plt.plot(epochs, mean_tr, label=f"train wd={wd}")
        plt.fill_between(epochs, mean_tr - err_tr, mean_tr + err_tr, alpha=0.2)
        plt.plot(epochs, mean_val, linestyle="--", label=f"val wd={wd}")
        plt.fill_between(epochs, mean_val - err_val, mean_val + err_val, alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss (mean ± stderr)")
    plt.legend(fontsize="small")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves_agg.png"), dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss curves: {e}")
    plt.close()

# ========== 2. Aggregated F1 Curves ==========
try:
    plt.figure()
    for wd in wds_sorted:
        d = agg[wd]
        mean_tr, err_tr = mean_stderr(d["f1_train"])
        mean_val, err_val = mean_stderr(d["f1_val"])
        epochs = d["epochs"]
        plt.plot(epochs, mean_tr, label=f"train wd={wd}")
        plt.fill_between(epochs, mean_tr - err_tr, mean_tr + err_tr, alpha=0.2)
        plt.plot(epochs, mean_val, linestyle="--", label=f"val wd={wd}")
        plt.fill_between(epochs, mean_val - err_val, mean_val + err_val, alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH: Training vs Validation F1 (mean ± stderr)")
    plt.legend(fontsize="small")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_f1_curves_agg.png"), dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated F1 curves: {e}")
    plt.close()

# ========== 3. Final Dev F1 vs WD ==========
try:
    plt.figure()
    dev_mean, dev_err, xs = [], [], []
    for wd in wds_sorted:
        val_curves = agg[wd]["f1_val"]
        final_vals = [c[-1] for c in val_curves]
        dev_mean.append(np.mean(final_vals))
        dev_err.append(np.std(final_vals) / np.sqrt(len(final_vals)))
        xs.append(float(wd))
    plt.errorbar(xs, dev_mean, yerr=dev_err, fmt="o-", capsize=3)
    plt.xscale("log")
    plt.xlabel("Weight Decay")
    plt.ylabel("Final Dev Macro-F1")
    plt.title("SPR_BENCH: Dev F1 vs Weight Decay (mean ± stderr)")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_devF1_vs_wd_agg.png"), dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated Dev-F1 plot: {e}")
    plt.close()

# ========== 4. Final Test F1 vs WD ==========
try:
    plt.figure()
    test_mean, test_err, xs = [], [], []
    for wd in wds_sorted:
        vals = agg[wd]["test_f1"]
        test_mean.append(np.mean(vals))
        test_err.append(np.std(vals) / np.sqrt(len(vals)))
        xs.append(float(wd))
    plt.errorbar(xs, test_mean, yerr=test_err, fmt="s-", color="green", capsize=3)
    plt.xscale("log")
    plt.xlabel("Weight Decay")
    plt.ylabel("Test Macro-F1")
    plt.title("SPR_BENCH: Test F1 vs Weight Decay (mean ± stderr)")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_testF1_vs_wd_agg.png"), dpi=150)
    plt.close()
    print(
        "Aggregated Test F1 (mean ± stderr):",
        {wd: (m, e) for wd, m, e in zip(wds_sorted, test_mean, test_err)},
    )
except Exception as e:
    print(f"Error creating aggregated Test-F1 plot: {e}")
    plt.close()

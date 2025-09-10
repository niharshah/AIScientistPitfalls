import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

# ---------- set up working dir ------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load all experiment data -----------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-30_20-55-34_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_01a41ffaf7174044b202d0bdfc69fb94_proc_1493252/experiment_data.npy",
    "experiments/2025-08-30_20-55-34_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_f1112e3e1d424da080930ebcd5200e6e_proc_1493250/experiment_data.npy",
    "experiments/2025-08-30_20-55-34_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_4285f9bbefc045f3aa4cb0c57e980b31_proc_1493251/experiment_data.npy",
]

all_runs = []
for p in experiment_data_path_list:
    try:
        run = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_runs.append(run)
    except Exception as e:
        print(f"Error loading experiment data at {p}: {e}")

# ---------- aggregate by dataset ---------------------------------------------
datasets = defaultdict(list)  # dataset_name -> list of run_dicts with that dataset
for run in all_runs:
    for ds_name, ds_dict in run.items():
        if isinstance(ds_dict, dict):
            datasets[ds_name].append(ds_dict)

summary_test_cplx = {}  # dataset -> (mean, sem)

for ds_name, runs in datasets.items():
    # ---------------- align epochs -----------------
    # find shortest epoch length present in all runs
    epoch_lengths = [len(r.get("epochs", [])) for r in runs if r.get("epochs", [])]
    if not epoch_lengths:
        continue
    min_len = min(epoch_lengths)
    epochs = runs[0]["epochs"][:min_len]  # use first run's epoch list as reference

    # ---------------- helper to stack metric arrays ---------------------------
    def stack_metric(path_keys):
        """Extract metric arrays (same length) from all runs and stack."""
        arrays = []
        for r in runs:
            cur = r
            try:
                for k in path_keys:
                    cur = cur[k]
                cur = np.asarray(cur)[:min_len]
                arrays.append(cur)
            except Exception:
                continue  # skip if any key missing
        if not arrays:
            return None, None
        arr = np.stack(arrays, axis=0)  # shape (n_runs, min_len)
        mean = arr.mean(axis=0)
        sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        return mean, sem

    # ---------------- plot loss curve -----------------------------------------
    try:
        mean_train, sem_train = stack_metric(["losses", "train"])
        mean_val, sem_val = stack_metric(["losses", "val"])
        if mean_train is not None and mean_val is not None:
            plt.figure()
            plt.plot(epochs, mean_train, label="Train mean", color="tab:blue")
            plt.fill_between(
                epochs,
                mean_train - sem_train,
                mean_train + sem_train,
                color="tab:blue",
                alpha=0.3,
                label="Train ± SEM",
            )
            plt.plot(epochs, mean_val, label="Val mean", color="tab:orange")
            plt.fill_between(
                epochs,
                mean_val - sem_val,
                mean_val + sem_val,
                color="tab:orange",
                alpha=0.3,
                label="Val ± SEM",
            )
            plt.title(f"{ds_name}: Loss Curve (mean ± SEM over {len(runs)} runs)")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            fname = f"loss_curve_mean_{ds_name}.png"
            plt.savefig(os.path.join(working_dir, fname))
    except Exception as e:
        print(f"Error creating aggregated loss curve for {ds_name}: {e}")
    finally:
        plt.close()

    # ---------------- common function for val metrics -------------------------
    for metric in ["CWA", "SWA", "CplxWA"]:
        try:
            mean_m, sem_m = stack_metric(["metrics", "val", metric])
            if mean_m is None:
                continue
            plt.figure()
            plt.plot(epochs, mean_m, label=f"{metric} mean")
            plt.fill_between(
                epochs, mean_m - sem_m, mean_m + sem_m, alpha=0.3, label="± SEM"
            )
            plt.title(
                f"{ds_name}: Validation {metric} (mean ± SEM over {len(runs)} runs)"
            )
            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.legend()
            fname = f"val_{metric.lower()}_mean_{ds_name}.png"
            plt.savefig(os.path.join(working_dir, fname))
        except Exception as e:
            print(f"Error creating aggregated {metric} plot for {ds_name}: {e}")
        finally:
            plt.close()

    # ---------------- aggregate test metrics ----------------------------------
    test_vals = defaultdict(list)  # metric -> list of values
    for r in runs:
        tst = r.get("metrics", {}).get("test", {})
        for m in ["CWA", "SWA", "CplxWA"]:
            if m in tst:
                test_vals[m].append(tst[m])
    # compute means and sems
    for m, vals in test_vals.items():
        vals = np.asarray(vals)
        mean_v = vals.mean()
        sem_v = vals.std(ddof=1) / np.sqrt(len(vals))
        if m == "CplxWA":
            summary_test_cplx[ds_name] = (mean_v, sem_v)
        print(f"{ds_name} Test {m}: mean={mean_v:.3f} ± {sem_v:.3f} (n={len(vals)})")

# ---------- summary bar plot for test CplxWA ---------------------------------
try:
    if summary_test_cplx:
        plt.figure()
        names = list(summary_test_cplx.keys())
        means = [summary_test_cplx[n][0] for n in names]
        sems = [summary_test_cplx[n][1] for n in names]
        x = np.arange(len(names))
        plt.bar(x, means, yerr=sems, capsize=4)
        plt.xticks(x, names, rotation=45, ha="right")
        plt.title("Test CplxWA by Dataset (mean ± SEM)")
        plt.xlabel("Dataset")
        plt.ylabel("Test CplxWA")
        fname = "summary_test_cplxwa_mean.png"
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, fname))
    else:
        print("No aggregated test CplxWA data found for summary plot.")
except Exception as e:
    print(f"Error creating aggregated summary bar plot: {e}")
finally:
    plt.close()

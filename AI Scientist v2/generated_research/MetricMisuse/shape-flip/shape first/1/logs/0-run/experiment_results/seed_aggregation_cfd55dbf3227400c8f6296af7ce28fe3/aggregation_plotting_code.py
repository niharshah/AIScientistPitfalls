import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------ load all experiment_data ------------------------
try:
    experiment_data_path_list = [
        "experiments/2025-07-27_23-49-14_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_59454030c3a8444ea580fc76da81c7e9_proc_319696/experiment_data.npy",
        "None/experiment_data.npy",
        "experiments/2025-07-27_23-49-14_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_290befb617fd4ef39315a184bf288a9c_proc_319694/experiment_data.npy",
    ]
    all_experiment_data = []
    for p in experiment_data_path_list:
        try:
            full_p = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
            ed = np.load(full_p, allow_pickle=True).item()
            all_experiment_data.append(ed)
        except Exception as e:
            print(f"Skipping {p}: {e}")
except Exception as e:
    print(f"Error loading experiment data list: {e}")
    all_experiment_data = []

# ------------------------ aggregate across runs ---------------------------
dataset = "SPR_BENCH"
section = "hidden_dim_tuning"

runs = []
for ed in all_experiment_data:
    try:
        runs.append(ed[section][dataset])
    except Exception as e:
        print(f"Missing {section}/{dataset} in one run: {e}")

if not runs:
    print("No valid runs found; aborting plots.")
    quit()

hidden_dims = sorted({int(k.split("_")[-1]) for r in runs for k in r.keys()})
num_runs = len(runs)


# helper to gather metric across runs
def gather_metric(metric_key, hidden_dim, inner_key="metrics"):
    """Return stacked np.array shape (num_runs, num_epochs) or (num_runs,)"""
    arrs = []
    for r in runs:
        hd_key = f"hidden_{hidden_dim}"
        if hd_key in r:
            try:
                arrs.append(np.asarray(r[hd_key][inner_key][metric_key]))
            except Exception:
                pass
    return np.vstack(arrs) if arrs and arrs[0].ndim == 1 else np.array(arrs)


# determine epochs length from first available run
sample_hd = next(iter(runs[0].keys()))
epochs = list(range(1, 1 + len(runs[0][sample_hd]["metrics"]["train_acc"])))

# ------------------------ FIGURE 1 : accuracy curves (mean ± SEM) --------
try:
    plt.figure(figsize=(6, 4))
    for hd in hidden_dims:
        tr = gather_metric("train_acc", hd)  # shape (n_runs, n_epochs)
        val = gather_metric("val_acc", hd)
        if tr.size == 0 or val.size == 0:
            continue
        tr_mean, tr_sem = tr.mean(0), tr.std(0, ddof=1) / np.sqrt(tr.shape[0])
        val_mean, val_sem = val.mean(0), val.std(0, ddof=1) / np.sqrt(val.shape[0])

        plt.plot(epochs, tr_mean, label=f"train hd{hd}")
        plt.fill_between(epochs, tr_mean - tr_sem, tr_mean + tr_sem, alpha=0.2)
        plt.plot(epochs, val_mean, linestyle="--", label=f"val hd{hd}")
        plt.fill_between(epochs, val_mean - val_sem, val_mean + val_sem, alpha=0.2)
    plt.title("SPR_BENCH: Train & Val Accuracy (mean ± SEM)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_hidden_dim_accuracy_mean_sem.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating aggregated accuracy plot: {e}")
    plt.close()

# ------------------------ FIGURE 2 : loss curves (mean ± SEM) ------------
try:
    plt.figure(figsize=(6, 4))
    for hd in hidden_dims:
        tr = gather_metric("train", hd, inner_key="losses")
        val = gather_metric("val", hd, inner_key="losses")
        if tr.size == 0 or val.size == 0:
            continue
        tr_mean, tr_sem = tr.mean(0), tr.std(0, ddof=1) / np.sqrt(tr.shape[0])
        val_mean, val_sem = val.mean(0), val.std(0, ddof=1) / np.sqrt(val.shape[0])
        plt.plot(epochs, tr_mean, label=f"train hd{hd}")
        plt.fill_between(epochs, tr_mean - tr_sem, tr_mean + tr_sem, alpha=0.2)
        plt.plot(epochs, val_mean, linestyle="--", label=f"val hd{hd}")
        plt.fill_between(epochs, val_mean - val_sem, val_mean + val_sem, alpha=0.2)
    plt.title("SPR_BENCH: Train & Val Loss (mean ± SEM)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_hidden_dim_loss_mean_sem.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

# ------------------------ FIGURE 3 : final val accuracy bar --------------
try:
    plt.figure(figsize=(5, 4))
    means, sems = [], []
    for hd in hidden_dims:
        val = gather_metric("val_acc", hd)
        if val.size == 0:
            means.append(np.nan)
            sems.append(np.nan)
            continue
        finals = val[:, -1]
        means.append(finals.mean())
        sems.append(finals.std(ddof=1) / np.sqrt(finals.shape[0]))
    x = np.arange(len(hidden_dims))
    plt.bar(x, means, yerr=sems, capsize=5, color="skyblue")
    plt.xticks(x, [str(hd) for hd in hidden_dims])
    plt.title("SPR_BENCH: Final Validation Accuracy (mean ± SEM)")
    plt.xlabel("Hidden Dimension")
    plt.ylabel("Accuracy (last epoch)")
    fname = os.path.join(working_dir, "SPR_BENCH_final_val_accuracy_mean_sem.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    for hd, m, s in zip(hidden_dims, means, sems):
        print(f"Hidden {hd}: ValAcc {m:.3f} ± {s:.3f}")
    plt.close()
except Exception as e:
    print(f"Error creating aggregated final val accuracy bar: {e}")
    plt.close()

# ------------------------ FIGURE 4 : ZSRTA bar ---------------------------
try:
    plt.figure(figsize=(5, 4))
    means, sems = [], []
    for hd in hidden_dims:
        zs = gather_metric("ZSRTA", hd)
        if zs.size == 0:
            means.append(np.nan)
            sems.append(np.nan)
            continue
        zs = zs[:, 0] if zs.ndim == 2 else zs
        means.append(zs.mean())
        sems.append(zs.std(ddof=1) / np.sqrt(zs.shape[0]))
    x = np.arange(len(hidden_dims))
    plt.bar(x, means, yerr=sems, capsize=5, color="salmon")
    plt.xticks(x, [str(hd) for hd in hidden_dims])
    plt.title("SPR_BENCH: Zero-Shot Rule Transfer Accuracy (mean ± SEM)")
    plt.xlabel("Hidden Dimension")
    plt.ylabel("ZSRTA")
    fname = os.path.join(working_dir, "SPR_BENCH_ZSRTA_mean_sem.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    for hd, m, s in zip(hidden_dims, means, sems):
        print(f"Hidden {hd}: ZSRTA {m:.3f} ± {s:.3f}")
    plt.close()
except Exception as e:
    print(f"Error creating aggregated ZSRTA bar: {e}")
    plt.close()

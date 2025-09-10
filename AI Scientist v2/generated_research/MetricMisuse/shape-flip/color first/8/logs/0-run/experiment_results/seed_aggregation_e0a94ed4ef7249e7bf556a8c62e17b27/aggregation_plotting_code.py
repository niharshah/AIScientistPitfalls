import matplotlib.pyplot as plt
import numpy as np
import os

# --- paths ---
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# 1) LOAD ALL EXPERIMENTS -------------------------------------------------
# ------------------------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-30_21-49-50_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_b82096ead9084ff29cd0b649f7327c7d_proc_1518068/experiment_data.npy",
    "experiments/2025-08-30_21-49-50_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_1a6c715e8d4e4acfa59342ed2ac27f31_proc_1518071/experiment_data.npy",
    "experiments/2025-08-30_21-49-50_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_87082babbb704a40a98ab873b2274001_proc_1518069/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        full_p = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        edata = np.load(full_p, allow_pickle=True).item()
        all_experiment_data.append(edata)
    except Exception as e:
        print(f"Error loading {p}: {e}")


# ------------------------------------------------------------------
# 2) HELPER ---------------------------------------------------------
# ------------------------------------------------------------------
def get_nested(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, {})
    return d if d != {} else default


abl = "token_level"
dset = "SPR_BENCH"


def collect(split_list_key):
    """split_list_key is ('losses','train') or similar"""
    collected = []
    for edata in all_experiment_data:
        arr = get_nested(edata, abl, dset, *split_list_key, default=None)
        if arr is not None and len(arr):
            collected.append(np.asarray(arr, dtype=float))
    return collected


loss_tr_runs = collect(("losses", "train"))
loss_val_runs = collect(("losses", "val"))
acc_tr_runs = collect(("metrics", "train"))
acc_val_runs = collect(("metrics", "val"))


def aggregate(runs):
    """Trim to shortest run length, return mean and sem arrays"""
    if not runs:
        return None, None
    min_len = min(len(r) for r in runs)
    runs_trim = np.stack([r[:min_len] for r in runs], axis=0)
    mean = runs_trim.mean(axis=0)
    sem = runs_trim.std(axis=0, ddof=1) / np.sqrt(runs_trim.shape[0])
    return mean, sem


mean_loss_tr, sem_loss_tr = aggregate(loss_tr_runs)
mean_loss_val, sem_loss_val = aggregate(loss_val_runs)
mean_acc_tr, sem_acc_tr = aggregate(acc_tr_runs)
mean_acc_val, sem_acc_val = aggregate(acc_val_runs)

epochs_loss = np.arange(1, len(mean_loss_tr) + 1) if mean_loss_tr is not None else None
epochs_acc = np.arange(1, len(mean_acc_tr) + 1) if mean_acc_tr is not None else None

# ------------------------------------------------------------------
# 3) PLOTS ----------------------------------------------------------
# ------------------------------------------------------------------
# 3.1 Aggregated Loss Curve
try:
    if mean_loss_tr is not None:
        plt.figure()
        plt.plot(epochs_loss, mean_loss_tr, label="Train Mean")
        plt.fill_between(
            epochs_loss,
            mean_loss_tr - sem_loss_tr,
            mean_loss_tr + sem_loss_tr,
            alpha=0.3,
            label="Train ±1SE",
        )
        if mean_loss_val is not None:
            plt.plot(epochs_loss, mean_loss_val, label="Val Mean")
            plt.fill_between(
                epochs_loss,
                mean_loss_val - sem_loss_val,
                mean_loss_val + sem_loss_val,
                alpha=0.3,
                label="Val ±1SE",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(
            f"{dset} – Aggregated Loss Curve\n(mean ± SE across {len(loss_tr_runs)} runs)"
        )
        plt.legend()
        fname = os.path.join(working_dir, f"{dset}_aggregated_loss_curve.png")
        plt.savefig(fname)
    else:
        print("No loss data found, skipping aggregated loss plot.")
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss curve: {e}")
    plt.close()

# 3.2 Aggregated Accuracy Curve
try:
    if mean_acc_tr is not None:
        plt.figure()
        plt.plot(epochs_acc, mean_acc_tr, label="Train Mean")
        plt.fill_between(
            epochs_acc,
            mean_acc_tr - sem_acc_tr,
            mean_acc_tr + sem_acc_tr,
            alpha=0.3,
            label="Train ±1SE",
        )
        if mean_acc_val is not None:
            plt.plot(epochs_acc, mean_acc_val, label="Val Mean")
            plt.fill_between(
                epochs_acc,
                mean_acc_val - sem_acc_val,
                mean_acc_val + sem_acc_val,
                alpha=0.3,
                label="Val ±1SE",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(
            f"{dset} – Aggregated Accuracy Curve\n(mean ± SE across {len(acc_tr_runs)} runs)"
        )
        plt.legend()
        fname = os.path.join(working_dir, f"{dset}_aggregated_accuracy_curve.png")
        plt.savefig(fname)
    else:
        print("No accuracy data found, skipping aggregated accuracy plot.")
    plt.close()
except Exception as e:
    print(f"Error creating aggregated accuracy curve: {e}")
    plt.close()

# ------------------------------------------------------------------
# 4) PRINT AGGREGATED FINAL METRIC ---------------------------------
# ------------------------------------------------------------------
if acc_val_runs:
    final_vals = [run[-1] for run in acc_val_runs]
    print(
        f"Mean final validation accuracy over {len(final_vals)} runs: {np.mean(final_vals):.3f} ± {np.std(final_vals,ddof=1)/np.sqrt(len(final_vals)):.3f} (SE)"
    )

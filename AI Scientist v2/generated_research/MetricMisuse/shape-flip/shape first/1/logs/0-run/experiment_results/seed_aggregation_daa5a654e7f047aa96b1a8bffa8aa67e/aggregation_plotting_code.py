import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------------------------------------------- #
# basic setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------------------------- #
# load ALL experiment_data.npy we can see in the Experiment Data Path   #
experiment_data_path_list = [
    os.path.join(working_dir, "experiment_data.npy"),
    "experiments/2025-07-27_23-49-14_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_b8c645f9a8444983917cffd60b8ae015_proc_312327/experiment_data.npy",
]
all_experiment_data = []
for p in experiment_data_path_list:
    try:
        if p is None or not os.path.isfile(p):
            continue
        d = np.load(p, allow_pickle=True).item()
        all_experiment_data.append(d)
    except Exception as e:
        print(f"Error loading {p}: {e}")

# concatenate experiments that match the study we care about ---------- #
runs = []
for d in all_experiment_data:
    try:
        exp_dict = d["hidden_dim_tuning"]["SPR_BENCH"]
        for run_key, run_val in exp_dict.items():
            runs.append(run_val)  # each element has .['metrics'] and .['losses']
    except Exception as e:
        # Skip anything that doesn't have this sub-dict
        continue

if not runs:
    print("No runs found for aggregation; exiting early.")
    exit()


# -------------------------------------------------------------------- #
# Utilities to stack metrics
def stack_metric(metric_name, subkey=None):
    """Return 2-D array shape (n_runs, n_epochs)"""
    vals = []
    for r in runs:
        if subkey is None:
            vals.append(r["metrics"][metric_name])
        else:
            vals.append(r[subkey][metric_name])
    return np.array(vals)


# Determine epoch count from first run
n_epochs = len(runs[0]["metrics"]["train_acc"])
epochs = np.arange(1, n_epochs + 1)
n_runs = len(runs)

# -------------------------------------------------------------------- #
# FIGURE 1 : aggregated accuracy curves with stderr shading
try:
    tr_acc_arr = stack_metric("train_acc")
    val_acc_arr = stack_metric("val_acc")

    tr_mean, tr_std = tr_acc_arr.mean(axis=0), tr_acc_arr.std(axis=0, ddof=1)
    val_mean, val_std = val_acc_arr.mean(axis=0), val_acc_arr.std(axis=0, ddof=1)
    tr_se, val_se = tr_std / np.sqrt(n_runs), val_std / np.sqrt(n_runs)

    plt.figure()
    plt.plot(epochs, tr_mean, color="blue", label="Train (mean)")
    plt.fill_between(
        epochs,
        tr_mean - tr_se,
        tr_mean + tr_se,
        color="blue",
        alpha=0.2,
        label="Train ± stderr",
    )
    plt.plot(epochs, val_mean, color="orange", label="Val (mean)")
    plt.fill_between(
        epochs,
        val_mean - val_se,
        val_mean + val_se,
        color="orange",
        alpha=0.2,
        label="Val ± stderr",
    )
    plt.title("SPR_BENCH: Aggregated Train & Val Accuracy\n(mean ± standard error)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_aggregated_accuracy_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating aggregated accuracy plot: {e}")
    plt.close()

# -------------------------------------------------------------------- #
# FIGURE 2 : aggregated loss curves with stderr shading
try:
    tr_loss_arr = stack_metric("train", subkey="losses")
    val_loss_arr = stack_metric("val", subkey="losses")

    tr_mean, tr_std = tr_loss_arr.mean(axis=0), tr_loss_arr.std(axis=0, ddof=1)
    val_mean, val_std = val_loss_arr.mean(axis=0), val_loss_arr.std(axis=0, ddof=1)
    tr_se, val_se = tr_std / np.sqrt(n_runs), val_std / np.sqrt(n_runs)

    plt.figure()
    plt.plot(epochs, tr_mean, color="green", label="Train Loss (mean)")
    plt.fill_between(
        epochs,
        tr_mean - tr_se,
        tr_mean + tr_se,
        color="green",
        alpha=0.2,
        label="Train ± stderr",
    )
    plt.plot(epochs, val_mean, color="red", label="Val Loss (mean)")
    plt.fill_between(
        epochs,
        val_mean - val_se,
        val_mean + val_se,
        color="red",
        alpha=0.2,
        label="Val ± stderr",
    )
    plt.title("SPR_BENCH: Aggregated Train & Val Loss\n(mean ± standard error)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_aggregated_loss_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

# -------------------------------------------------------------------- #
# FIGURE 3 : final validation accuracy across runs with error bar
try:
    final_val_acc = np.array([r["metrics"]["val_acc"][-1] for r in runs])
    mean_final = final_val_acc.mean()
    se_final = final_val_acc.std(ddof=1) / np.sqrt(n_runs)

    plt.figure()
    plt.bar(["Aggregated"], [mean_final], yerr=[se_final], color="skyblue", capsize=5)
    plt.title("SPR_BENCH: Final Validation Accuracy\n(mean ± standard error)")
    plt.ylabel("Accuracy (last epoch)")
    fname = os.path.join(working_dir, "SPR_BENCH_final_val_accuracy_aggregated.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating aggregated final val accuracy bar: {e}")
    plt.close()

# -------------------------------------------------------------------- #
# FIGURE 4 : ZSRTA aggregated bar
try:
    zsrtas = np.array([r["metrics"]["ZSRTA"][0] for r in runs])
    mean_zs = zsrtas.mean()
    se_zs = zsrtas.std(ddof=1) / np.sqrt(n_runs)

    plt.figure()
    plt.bar(["Aggregated"], [mean_zs], yerr=[se_zs], color="salmon", capsize=5)
    plt.title(
        "SPR_BENCH: Zero-Shot Rule Transfer Accuracy (ZSRTA)\n(mean ± standard error)"
    )
    plt.ylabel("ZSRTA")
    fname = os.path.join(working_dir, "SPR_BENCH_ZSRTA_aggregated.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating aggregated ZSRTA bar: {e}")
    plt.close()

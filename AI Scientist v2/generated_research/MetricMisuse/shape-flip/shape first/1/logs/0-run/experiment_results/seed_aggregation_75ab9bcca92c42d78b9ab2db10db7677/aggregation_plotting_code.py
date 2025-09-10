import matplotlib.pyplot as plt
import numpy as np
import os

# ------------ set up work dir -------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------ locate experiment files -----
experiment_data_path_list = [
    "None/experiment_data.npy",
    "None/experiment_data.npy",
    "experiments/2025-07-27_23-49-14_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_841e59e95c3445b0bf7bf71ccd94a9d2_proc_316521/experiment_data.npy",
]
all_experiment_data = []
for p in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(data)
    except Exception as e:
        print(f"Error loading {p}: {e}")

# ------------- aggregate  -----------------
spr_runs = []
for exp in all_experiment_data:
    try:
        spr_runs.append(exp["hidden_dim_tuning"]["SPR_BENCH"])
    except Exception as e:
        print(f"Bad experiment structure skipped: {e}")

if len(spr_runs) == 0:
    print("No valid SPR_BENCH runs found; aborting plotting.")
    exit()

# determine hidden dims present in every run
all_hidden_dims = sorted(
    set(int(k.split("_")[-1]) for run in spr_runs for k in run.keys())
)
# keep at most 5 for readability
hidden_dims = all_hidden_dims[:5]

# epochs length assumes all runs same length; fall back otherwise
first_run = next(iter(spr_runs[0].values()))
epochs = list(range(1, 1 + len(first_run["metrics"]["val_acc"])))

# helper dict to store arrays per hd
acc_tensor = {hd: [] for hd in hidden_dims}
loss_train_tensor = {hd: [] for hd in hidden_dims}
loss_val_tensor = {hd: [] for hd in hidden_dims}
final_val_acc = {hd: [] for hd in hidden_dims}
zsrtas = {hd: [] for hd in hidden_dims}

for run in spr_runs:
    for hd in hidden_dims:
        key = f"hidden_{hd}"
        if key not in run:
            continue
        acc_tensor[hd].append(run[key]["metrics"]["val_acc"])
        loss_train_tensor[hd].append(run[key]["losses"]["train"])
        loss_val_tensor[hd].append(run[key]["losses"]["val"])
        final_val_acc[hd].append(run[key]["metrics"]["val_acc"][-1])
        zsrtas[hd].append(run[key]["metrics"]["ZSRTA"][0])

# ------------ FIGURE 1: Val Accuracy mean ± stderr -----------
try:
    plt.figure()
    for hd in hidden_dims:
        if len(acc_tensor[hd]) == 0:
            continue
        acc_arr = np.array(acc_tensor[hd])  # shape (runs, epochs)
        mean = acc_arr.mean(axis=0)
        stderr = acc_arr.std(axis=0, ddof=1) / np.sqrt(acc_arr.shape[0])
        plt.plot(epochs, mean, marker="o", label=f"hd{hd} mean")
        plt.fill_between(epochs, mean - stderr, mean + stderr, alpha=0.2)
    plt.title("SPR_BENCH: Validation Accuracy (mean ± s.e.)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_accuracy_mean_stderr.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating aggregated accuracy plot: {e}")
    plt.close()

# ------------ FIGURE 2: Loss curves mean ± stderr -----------
try:
    plt.figure()
    for hd in hidden_dims:
        if len(loss_train_tensor[hd]) == 0:
            continue
        tr = np.array(loss_train_tensor[hd])
        vl = np.array(loss_val_tensor[hd])
        tr_m, tr_se = tr.mean(axis=0), tr.std(axis=0, ddof=1) / np.sqrt(tr.shape[0])
        vl_m, vl_se = vl.mean(axis=0), vl.std(axis=0, ddof=1) / np.sqrt(vl.shape[0])
        plt.plot(epochs, tr_m, marker="o", label=f"train hd{hd} mean")
        plt.fill_between(epochs, tr_m - tr_se, tr_m + tr_se, alpha=0.15)
        plt.plot(epochs, vl_m, marker="x", linestyle="--", label=f"val hd{hd} mean")
        plt.fill_between(epochs, vl_m - vl_se, vl_m + vl_se, alpha=0.15)
    plt.title("SPR_BENCH: Cross-Entropy Loss (mean ± s.e.)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_mean_stderr.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

# ------------ FIGURE 3: Final Val Acc & ZSRTA bars ----------
try:
    plt.figure(figsize=(10, 4))
    x = np.arange(len(hidden_dims))
    bar_w = 0.35

    # Final validation accuracy
    val_means = [np.mean(final_val_acc[hd]) for hd in hidden_dims]
    val_se = [
        (
            np.std(final_val_acc[hd], ddof=1) / np.sqrt(len(final_val_acc[hd]))
            if len(final_val_acc[hd]) > 1
            else 0
        )
        for hd in hidden_dims
    ]
    plt.bar(
        x - bar_w / 2,
        val_means,
        yerr=val_se,
        width=bar_w,
        label="Final Val Acc",
        color="skyblue",
        capsize=5,
    )

    # ZSRTA
    z_means = [np.mean(zsrtas[hd]) for hd in hidden_dims]
    z_se = [
        (
            np.std(zsrtas[hd], ddof=1) / np.sqrt(len(zsrtas[hd]))
            if len(zsrtas[hd]) > 1
            else 0
        )
        for hd in hidden_dims
    ]
    plt.bar(
        x + bar_w / 2,
        z_means,
        yerr=z_se,
        width=bar_w,
        label="ZSRTA",
        color="salmon",
        capsize=5,
    )

    plt.xticks(x, [str(hd) for hd in hidden_dims])
    plt.title("SPR_BENCH: Final Validation Accuracy and ZSRTA (mean ± s.e.)")
    plt.xlabel("Hidden Dimension")
    plt.ylabel("Score")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_final_metrics_bar.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating aggregated bar plot: {e}")
    plt.close()

# ------------- print numerical summary -------------
for hd in hidden_dims:
    if len(final_val_acc[hd]) == 0:
        continue
    mean = np.mean(final_val_acc[hd])
    se = (
        np.std(final_val_acc[hd], ddof=1) / np.sqrt(len(final_val_acc[hd]))
        if len(final_val_acc[hd]) > 1
        else 0
    )
    print(
        f"Hidden {hd}: Final Val Acc = {mean:.3f} ± {se:.3f} (n={len(final_val_acc[hd])})"
    )

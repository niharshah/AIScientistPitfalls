import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------ paths & loading ------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data_path_list = [
    "experiments/2025-08-15_16-42-50_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_9ec46ede532e48fda74554e8d9fcea65_proc_2950687/experiment_data.npy",
    "experiments/2025-08-15_16-42-50_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_62c74caf072a439a913c16d4e8ed65ec_proc_2950686/experiment_data.npy",
    "experiments/2025-08-15_16-42-50_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_e3f2473a836047c39473814a2cedd3bf_proc_2950689/experiment_data.npy",
]

all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        all_experiment_data.append(np.load(full_path, allow_pickle=True).item())
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []


# ------------------------------------ aggregation helpers --------------------------------
def nansem(a, axis=0):
    """nan-aware standard error of the mean"""
    cnt = np.sum(~np.isnan(a), axis=axis)
    return np.nanstd(a, axis=axis, ddof=1) / np.maximum(cnt, 1) ** 0.5


# ------------------------------------ aggregate & plot -----------------------------------
# collect all dataset names that appear in any run
dataset_names = set()
for exp in all_experiment_data:
    dataset_names.update(exp.keys())

for ds_name in sorted(dataset_names):
    # gather this dataset across runs
    runs = [exp[ds_name] for exp in all_experiment_data if ds_name in exp]
    if not runs:
        continue
    num_runs = len(runs)

    # ---------------- align epoch lengths ----------------
    max_epochs = max(len(r["losses"]["train"]) for r in runs)
    train_loss_mat = np.full((num_runs, max_epochs), np.nan)
    val_loss_mat = np.full((num_runs, max_epochs), np.nan)
    val_swa_mat = np.full((num_runs, max_epochs), np.nan)

    for i, r in enumerate(runs):
        tl = np.asarray(r["losses"]["train"])
        vl = np.asarray(r["losses"]["val"])
        vs = np.asarray(r["metrics"]["val"])
        train_loss_mat[i, : len(tl)] = tl
        val_loss_mat[i, : len(vl)] = vl
        val_swa_mat[i, : len(vs)] = vs

    epochs = np.arange(1, max_epochs + 1)

    # ---------------- plot 1: aggregated loss curves ----------------
    try:
        plt.figure()
        mean_tl = np.nanmean(train_loss_mat, axis=0)
        sem_tl = nansem(train_loss_mat, axis=0)
        mean_vl = np.nanmean(val_loss_mat, axis=0)
        sem_vl = nansem(val_loss_mat, axis=0)

        plt.plot(epochs, mean_tl, label="Train Mean")
        plt.fill_between(
            epochs, mean_tl - sem_tl, mean_tl + sem_tl, alpha=0.3, label="Train SEM"
        )
        plt.plot(epochs, mean_vl, label="Validation Mean")
        plt.fill_between(
            epochs, mean_vl - sem_vl, mean_vl + sem_vl, alpha=0.3, label="Val SEM"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{ds_name}: Aggregated Loss Curves\n(n={num_runs} runs)")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name}_loss_curve_mean_sem.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {ds_name}: {e}")
        plt.close()

    # ---------------- plot 2: aggregated validation SWA -------------
    try:
        plt.figure()
        mean_vs = np.nanmean(val_swa_mat, axis=0)
        sem_vs = nansem(val_swa_mat, axis=0)

        plt.plot(epochs, mean_vs, marker="o", label="Validation SWA Mean")
        plt.fill_between(
            epochs,
            mean_vs - sem_vs,
            mean_vs + sem_vs,
            alpha=0.3,
            label="Validation SEM",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title(f"{ds_name}: Aggregated Validation SWA\n(n={num_runs} runs)")
        plt.ylim(0, 1.05)
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name}_val_SWA_mean_sem.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated SWA plot for {ds_name}: {e}")
        plt.close()

    # ---------------- aggregated best SWA printout ------------------
    try:
        best_swa_each_run = [np.nanmax(vs) for vs in val_swa_mat]
        best_swa_each_run = np.asarray(best_swa_each_run)
        mean_best = np.nanmean(best_swa_each_run)
        sem_best = nansem(best_swa_each_run, axis=0)
        print(
            f"{ds_name}: best Validation SWA = {mean_best:.4f} ± {sem_best:.4f} (mean±SEM over {num_runs} runs)"
        )
    except Exception as e:
        print(f"Error computing aggregated best SWA for {ds_name}: {e}")

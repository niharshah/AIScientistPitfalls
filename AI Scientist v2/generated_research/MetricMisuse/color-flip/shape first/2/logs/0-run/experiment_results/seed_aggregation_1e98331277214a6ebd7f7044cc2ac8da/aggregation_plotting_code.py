import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- experiment data paths (provided) ----------
experiment_data_path_list = [
    "experiments/2025-08-15_22-25-14_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_f282b607879344a486768597829e2fcd_proc_2999654/experiment_data.npy",
    "experiments/2025-08-15_22-25-14_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_34e80b7298b946e7a40653fbb7efd2df_proc_2999653/experiment_data.npy",
    "experiments/2025-08-15_22-25-14_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_d7ef14d2653e42a587556ce306e0802e_proc_2999656/experiment_data.npy",
]

all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        if os.path.isfile(full_path):
            all_experiment_data.append(np.load(full_path, allow_pickle=True).item())
        else:
            print(f"Warning: file not found -> {full_path}")
except Exception as e:
    print(f"Error loading experiment data: {e}")

n_runs = len(all_experiment_data)
if n_runs == 0:
    print("No experiment files were successfully loaded.")
    exit()

# ---------- collect & align per-epoch arrays ----------
train_losses, val_losses = [], []
swa_vals, cwa_vals, ccwa_vals = [], []

for exp in all_experiment_data:
    jt = exp.get("joint_training", {})
    tr = jt.get("losses", {}).get("train", [])
    vl = jt.get("losses", {}).get("val", [])
    mets = jt.get("metrics", {}).get("val", [])

    # losses
    if tr and vl:
        train_losses.append(np.array(tr))
        val_losses.append(np.array(vl))

    # metric lists
    if mets:
        swa_vals.append(np.array([m["swa"] for m in mets]))
        cwa_vals.append(np.array([m["cwa"] for m in mets]))
        ccwa_vals.append(np.array([m["ccwa"] for m in mets]))


# helper to truncate to min length
def _align(arr_list):
    min_len = min(len(a) for a in arr_list)
    return np.array([a[:min_len] for a in arr_list]), min_len


# ---------- aggregated loss curves ----------
try:
    if train_losses and val_losses:
        tr_arr, n_ep = _align(train_losses)
        vl_arr, _ = _align(val_losses)

        ep = np.arange(1, n_ep + 1)
        tr_mean, tr_sem = tr_arr.mean(0), tr_arr.std(0, ddof=1) / np.sqrt(n_runs)
        vl_mean, vl_sem = vl_arr.mean(0), vl_arr.std(0, ddof=1) / np.sqrt(n_runs)

        plt.figure(figsize=(7, 4))
        plt.plot(ep, tr_mean, label="Train (mean)")
        plt.fill_between(
            ep, tr_mean - tr_sem, tr_mean + tr_sem, alpha=0.3, label="Train SEM"
        )
        plt.plot(ep, vl_mean, label="Validation (mean)", color="orange")
        plt.fill_between(
            ep,
            vl_mean - vl_sem,
            vl_mean + vl_sem,
            alpha=0.3,
            color="orange",
            label="Val SEM",
        )
        plt.title("SPR_BENCH Aggregated Loss Curves\n(mean ± SEM across runs)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_aggregated_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

# ---------- aggregated metric curves ----------
try:
    if swa_vals and cwa_vals and ccwa_vals:
        swa_arr, n_ep_m = _align(swa_vals)
        cwa_arr, _ = _align(cwa_vals)
        ccwa_arr, _ = _align(ccwa_vals)

        ep = np.arange(1, n_ep_m + 1)
        for arr, name, color in zip(
            [swa_arr, cwa_arr, ccwa_arr],
            ["SWA", "CWA", "CCWA"],
            ["tab:blue", "tab:green", "tab:red"],
        ):
            mean = arr.mean(0)
            sem = arr.std(0, ddof=1) / np.sqrt(n_runs)
            plt.plot(ep, mean, label=f"{name} (mean)", color=color)
            plt.fill_between(ep, mean - sem, mean + sem, alpha=0.25, color=color)

        plt.title("SPR_BENCH Aggregated Validation Metrics\n(mean ± SEM across runs)")
        plt.xlabel("Epoch")
        plt.ylabel("Metric value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_aggregated_metric_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated metric plot: {e}")
    plt.close()

# ---------- summary of best CCWA ----------
best_ccwas = []
for arr in ccwa_vals:
    best_ccwas.append(arr.max())
if best_ccwas:
    best_ccwas = np.array(best_ccwas)
    mean_best = best_ccwas.mean()
    sem_best = best_ccwas.std(ddof=1) / np.sqrt(len(best_ccwas))
    print("Per-run best CCWA:", best_ccwas.round(4))
    print(f"Mean best CCWA: {mean_best:.4f} ± {sem_best:.4f}")

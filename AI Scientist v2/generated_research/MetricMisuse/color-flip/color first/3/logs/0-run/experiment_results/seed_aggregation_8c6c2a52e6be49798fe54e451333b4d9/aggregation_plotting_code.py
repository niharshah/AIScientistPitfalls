import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------ #
# 0. House-keeping                                                   #
# ------------------------------------------------------------------ #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------ #
# 1. Locate and load all experiment_data.npy files                   #
# ------------------------------------------------------------------ #
try:
    experiment_data_path_list = [
        "experiments/2025-08-31_02-26-58_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_744e1521b0454c698fa88f91cffae906_proc_1605338/experiment_data.npy",
        "experiments/2025-08-31_02-26-58_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_7ec433b7b0264904b6ae1736ce55a71a_proc_1605337/experiment_data.npy",
        "experiments/2025-08-31_02-26-58_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_5d8314148bf54c4e8d05273c922f3a02_proc_1605336/experiment_data.npy",
    ]
    all_experiment_data = []
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        all_experiment_data.append(np.load(full_path, allow_pickle=True).item())
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []


# Helper
def unpack(list_of_tuples, idx):
    return [t[idx] for t in list_of_tuples]


# ------------------------------------------------------------------ #
# 2. Aggregate metrics across all runs & experiments                 #
# ------------------------------------------------------------------ #
agg = {
    "train_loss": {},  # epoch -> list of vals
    "val_loss": {},  # epoch -> list of vals
    "val_hcs": {},  # epoch -> list of vals
}

best_hcs_per_run = {}  # run_name -> best HCSA
dataset_name = "SPR_BENCH"

for exp_idx, exp in enumerate(all_experiment_data):
    try:
        runs = exp["epochs_tuning"][dataset_name]["runs"]
    except Exception:
        continue
    for run_name, run in runs.items():
        # Train loss
        for epoch, loss in run["losses"]["train"]:
            agg["train_loss"].setdefault(epoch, []).append(loss)
        # Val loss
        for epoch, loss in run["losses"]["val"]:
            agg["val_loss"].setdefault(epoch, []).append(loss)
        # Val metrics
        for tup in run["metrics"]["val"]:
            epoch, hcs = tup[0], tup[3]
            agg["val_hcs"].setdefault(epoch, []).append(hcs)
        # Best HCSA for this run
        hcs_vals = [t[3] for t in run["metrics"]["val"]]
        if hcs_vals:
            best_hcs_per_run[f"{exp_idx}-{run_name}"] = np.max(hcs_vals)


# Convenience for computing mean & SEM
def mean_sem(v):
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return np.nan, np.nan
    return float(np.mean(v)), (
        float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    )


# ------------------------------------------------------------------ #
# 3. Figure 1 — Mean±SEM Train & Val Loss                            #
# ------------------------------------------------------------------ #
try:
    epochs = sorted(set(agg["train_loss"].keys()) | set(agg["val_loss"].keys()))
    tr_mean, tr_sem, val_mean, val_sem = [], [], [], []
    for ep in epochs:
        m, s = mean_sem(agg["train_loss"].get(ep, []))
        tr_mean.append(m)
        tr_sem.append(s)
        m, s = mean_sem(agg["val_loss"].get(ep, []))
        val_mean.append(m)
        val_sem.append(s)

    plt.figure()
    plt.plot(epochs, tr_mean, "--", color="tab:blue", label="Train mean")
    plt.fill_between(
        epochs,
        np.array(tr_mean) - np.array(tr_sem),
        np.array(tr_mean) + np.array(tr_sem),
        color="tab:blue",
        alpha=0.3,
        label="Train ± SEM",
    )
    plt.plot(epochs, val_mean, "-", color="tab:orange", label="Val mean")
    plt.fill_between(
        epochs,
        np.array(val_mean) - np.array(val_sem),
        np.array(val_mean) + np.array(val_sem),
        color="tab:orange",
        alpha=0.3,
        label="Val ± SEM",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title(f"{dataset_name}: Mean Train vs. Val Loss\n(Shaded: ±1 SEM across runs)")
    plt.legend(fontsize=7)
    fname = os.path.join(working_dir, f"{dataset_name}_agg_loss_mean_sem.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 4. Figure 2 — Mean±SEM Validation HCSA                             #
# ------------------------------------------------------------------ #
try:
    epochs = sorted(agg["val_hcs"].keys())
    hcs_mean, hcs_sem = [], []
    for ep in epochs:
        m, s = mean_sem(agg["val_hcs"][ep])
        hcs_mean.append(m)
        hcs_sem.append(s)

    plt.figure()
    plt.plot(epochs, hcs_mean, color="tab:green", label="HCSA mean")
    plt.fill_between(
        epochs,
        np.array(hcs_mean) - np.array(hcs_sem),
        np.array(hcs_mean) + np.array(hcs_sem),
        color="tab:green",
        alpha=0.3,
        label="± SEM",
    )
    plt.xlabel("Epoch")
    plt.ylabel("HCSA")
    plt.title(f"{dataset_name}: Validation HCSA\n(Mean ± SEM across runs)")
    plt.legend(fontsize=7)
    fname = os.path.join(working_dir, f"{dataset_name}_agg_val_HCSA_mean_sem.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating aggregated HCSA plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 5. Figure 3 — Best HCSA per run                                    #
# ------------------------------------------------------------------ #
try:
    run_names = list(best_hcs_per_run.keys())
    best_vals = [best_hcs_per_run[n] for n in run_names]
    sort_idx = np.argsort(best_vals)[::-1]
    run_names = [run_names[i] for i in sort_idx]
    best_vals = [best_vals[i] for i in sort_idx]

    plt.figure(figsize=(6, 3))
    plt.bar(range(len(best_vals)), best_vals, tick_label=run_names)
    plt.ylabel("Best Validation HCSA")
    plt.title(f"{dataset_name}: Best HCSA for each run")
    plt.xticks(rotation=45, ha="right", fontsize=6)
    plt.tight_layout()
    fname = os.path.join(working_dir, f"{dataset_name}_best_HCSA_each_run.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating best-HCSA bar chart: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 6. Text summary                                                    #
# ------------------------------------------------------------------ #
try:
    # Overall best epoch statistics
    overall_best_vals = [v for v in best_hcs_per_run.values()]
    m, s = mean_sem(overall_best_vals)
    print(
        f"\nOverall best validation HCSA across all runs: {m:.3f} ± {s:.3f} (SEM, n={len(overall_best_vals)})"
    )
except Exception as e:
    print(f"Error printing summary statistics: {e}")

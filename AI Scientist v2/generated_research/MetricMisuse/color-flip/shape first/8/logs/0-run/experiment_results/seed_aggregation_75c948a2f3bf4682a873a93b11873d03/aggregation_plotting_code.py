import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


def safe_get(d, *keys, default=list()):
    for k in keys:
        d = d.get(k, {})
    return d if isinstance(d, list) else default


# ---------------------------------------------------------------
# LOAD ALL EXPERIMENT RUNS
# ---------------------------------------------------------------
try:
    experiment_data_path_list = [
        "experiments/2025-08-16_02-31-02_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_8fc508f9af294d3898227f0141bf2135_proc_3100052/experiment_data.npy",
        "experiments/2025-08-16_02-31-02_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_a6fdcd68f93c48bcbe0f1e4d8d61cea4_proc_3100051/experiment_data.npy",
        "experiments/2025-08-16_02-31-02_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_d74e81ece0564ae9bfac3b684f0199d3_proc_3100050/experiment_data.npy",
    ]
    all_experiment_data = []
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp_d = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_d)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# ---------------------------------------------------------------
# AGGREGATE AND PLOT
# ---------------------------------------------------------------
from collections import defaultdict

runs = len(all_experiment_data)

# collect dataset names present in at least one run
dset_names = set()
for run_d in all_experiment_data:
    dset_names.update(run_d.keys())

for dset_name in dset_names:
    # containers keyed by metric type
    losses_tr_runs, losses_val_runs, ccwa_val_runs = [], [], []

    # gather arrays from every run where they exist
    for run_d in all_experiment_data:
        if dset_name not in run_d:
            continue
        logs = run_d[dset_name]
        tr = safe_get(logs, "losses", "train")
        val = safe_get(logs, "losses", "val")
        ccwa = safe_get(logs, "metrics", "val_CCWA")
        if tr and val:
            losses_tr_runs.append(np.array(tr))
            losses_val_runs.append(np.array(val))
        if ccwa:
            ccwa_val_runs.append(np.array(ccwa))

    # skip if fewer than 2 runs found
    if len(losses_tr_runs) < 2:
        continue

    # align lengths to the minimum epoch length across runs
    min_len_loss = min(map(len, losses_tr_runs))
    losses_tr_runs = [x[:min_len_loss] for x in losses_tr_runs]
    losses_val_runs = [x[:min_len_loss] for x in losses_val_runs]

    if ccwa_val_runs:
        min_len_ccwa = min(map(len, ccwa_val_runs))
        ccwa_val_runs = [x[:min_len_ccwa] for x in ccwa_val_runs]

    # convert to arrays
    tr_arr = np.vstack(losses_tr_runs)
    val_arr = np.vstack(losses_val_runs)

    # -----------------------------------------------------------
    # 1) Mean Train & Val Loss with SE band
    # -----------------------------------------------------------
    try:
        epochs = np.arange(1, min_len_loss + 1)
        tr_mean, tr_se = tr_arr.mean(axis=0), tr_arr.std(axis=0, ddof=1) / np.sqrt(
            len(tr_arr)
        )
        val_mean, val_se = val_arr.mean(axis=0), val_arr.std(axis=0, ddof=1) / np.sqrt(
            len(val_arr)
        )

        plt.figure()
        plt.plot(epochs, tr_mean, label="Train Mean", color="tab:blue")
        plt.fill_between(
            epochs,
            tr_mean - tr_se,
            tr_mean + tr_se,
            color="tab:blue",
            alpha=0.3,
            label="Train ±SE",
        )

        plt.plot(epochs, val_mean, label="Val Mean", color="tab:orange")
        plt.fill_between(
            epochs,
            val_mean - val_se,
            val_mean + val_se,
            color="tab:orange",
            alpha=0.3,
            label="Val ±SE",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dset_name}: Mean Train/Val Loss Across {runs} Runs")
        plt.legend()
        fname = f"{dset_name}_agg_loss_curves.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot ({dset_name}): {e}")
        plt.close()

    # -----------------------------------------------------------
    # 2) Mean Validation CCWA with error bars every 5 epochs
    # -----------------------------------------------------------
    try:
        if ccwa_val_runs:
            ccwa_arr = np.vstack(ccwa_val_runs)
            epochs_ccwa = np.arange(1, min_len_ccwa + 1)
            ccwa_mean = ccwa_arr.mean(axis=0)
            ccwa_se = ccwa_arr.std(axis=0, ddof=1) / np.sqrt(len(ccwa_arr))

            plt.figure()
            plt.plot(epochs_ccwa, ccwa_mean, label="CCWA Mean", color="tab:green")
            # sparse error bars (<=5)
            step = max(1, len(epochs_ccwa) // 5)
            plt.errorbar(
                epochs_ccwa[::step],
                ccwa_mean[::step],
                yerr=ccwa_se[::step],
                fmt="o",
                color="tab:green",
                ecolor="gray",
                capsize=3,
                label="±SE",
            )

            plt.xlabel("Epoch")
            plt.ylabel("CCWA")
            plt.title(f"{dset_name}: Mean Validation CCWA Across {runs} Runs")
            plt.legend()
            fname = f"{dset_name}_agg_val_CCWA.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated CCWA plot ({dset_name}): {e}")
        plt.close()

    # -----------------------------------------------------------
    # 3) Final-epoch CCWA per run + group mean ±SE
    # -----------------------------------------------------------
    try:
        if ccwa_val_runs:
            final_vals = [arr[-1] for arr in ccwa_val_runs]
            indices = np.arange(len(final_vals))
            mean_final = np.mean(final_vals)
            se_final = np.std(final_vals, ddof=1) / np.sqrt(len(final_vals))

            plt.figure()
            plt.bar(indices, final_vals, color="skyblue", label="Individual Runs")
            plt.errorbar(
                len(indices) + 0.5,
                mean_final,
                yerr=se_final,
                fmt="D",
                color="red",
                capsize=5,
                label="Mean ±SE",
            )
            plt.xticks(
                list(indices) + [len(indices) + 0.5],
                [f"Run {i}" for i in indices] + ["Mean"],
            )
            plt.ylabel("Final CCWA")
            plt.title(f"{dset_name}: Final-epoch CCWA Across Runs")
            plt.legend()
            fname = f"{dset_name}_final_CCWA_runs.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error creating final CCWA bar plot ({dset_name}): {e}")
        plt.close()

print("Aggregated plotting complete – figures saved in", working_dir)

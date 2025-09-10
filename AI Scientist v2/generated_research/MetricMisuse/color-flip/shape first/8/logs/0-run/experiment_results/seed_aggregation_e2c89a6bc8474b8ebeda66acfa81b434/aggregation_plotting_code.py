import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------
# 1. Load all experiment_data dicts
experiment_data_path_list = [
    "experiments/2025-08-16_02-31-02_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_c0ee3e4f8a1e482187a690ec50900bcf_proc_3096398/experiment_data.npy",
    "experiments/2025-08-16_02-31-02_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_b65da683c6be43b1b2fce20186d6c934_proc_3096395/experiment_data.npy",
    "experiments/2025-08-16_02-31-02_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_5fa7c7c76a194f05b1250c667257800a_proc_3096396/experiment_data.npy",
]

all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        edict = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(edict)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# -------------------------------------------------
# 2. Aggregate by dataset
datasets = {}
for run_dict in all_experiment_data:
    for dset_name, dset_dict in run_dict.items():
        entry = datasets.setdefault(
            dset_name,
            {
                "losses_tr": [],
                "losses_val": [],
                "scwa": [],
            },
        )
        entry["losses_tr"].append(np.asarray(dset_dict["losses"]["train"], dtype=float))
        entry["losses_val"].append(np.asarray(dset_dict["losses"]["val"], dtype=float))
        entry["scwa"].append(np.asarray(dset_dict["metrics"]["val_SCWA"], dtype=float))


# -------------------------------------------------
# 3. Helper to trim runs to common length (min length across runs)
def stack_and_trim(list_of_1d_arrays):
    if not list_of_1d_arrays:
        return np.empty((0, 0))
    min_len = min(len(a) for a in list_of_1d_arrays)
    trimmed = np.stack([a[:min_len] for a in list_of_1d_arrays], axis=0)
    return trimmed  # shape (n_runs, min_len)


# -------------------------------------------------
for dset_name, dset_data in datasets.items():
    # ----- Aggregate Loss Curves -----
    try:
        tr_mat = stack_and_trim(dset_data["losses_tr"])
        val_mat = stack_and_trim(dset_data["losses_val"])
        if tr_mat.size and val_mat.size:
            epochs = np.arange(1, tr_mat.shape[1] + 1)
            tr_mean, tr_sem = np.nanmean(tr_mat, axis=0), np.nanstd(
                tr_mat, axis=0
            ) / np.sqrt(tr_mat.shape[0])
            val_mean, val_sem = np.nanmean(val_mat, axis=0), np.nanstd(
                val_mat, axis=0
            ) / np.sqrt(val_mat.shape[0])

            plt.figure()
            plt.plot(epochs, tr_mean, label="Train Mean", color="tab:blue")
            plt.fill_between(
                epochs,
                tr_mean - tr_sem,
                tr_mean + tr_sem,
                alpha=0.3,
                color="tab:blue",
                label="Train ±SEM",
            )
            plt.plot(epochs, val_mean, label="Val Mean", color="tab:orange")
            plt.fill_between(
                epochs,
                val_mean - val_sem,
                val_mean + val_sem,
                alpha=0.3,
                color="tab:orange",
                label="Val ±SEM",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(
                f"{dset_name}: Loss Curve (Mean ± SEM)\nAggregated over {tr_mat.shape[0]} runs"
            )
            plt.legend()
            save_name = os.path.join(
                working_dir, f"{dset_name}_aggregated_loss_curve.png"
            )
            plt.savefig(save_name)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss curve for {dset_name}: {e}")
        plt.close()

    # ----- Aggregate SCWA Curves -----
    try:
        scwa_mat = stack_and_trim(dset_data["scwa"])
        if scwa_mat.size:
            epochs = np.arange(1, scwa_mat.shape[1] + 1)
            scwa_mean = np.nanmean(scwa_mat, axis=0)
            scwa_sem = np.nanstd(scwa_mat, axis=0) / np.sqrt(scwa_mat.shape[0])

            plt.figure()
            plt.plot(
                epochs, scwa_mean, marker="o", color="tab:green", label="Val SCWA Mean"
            )
            plt.fill_between(
                epochs,
                scwa_mean - scwa_sem,
                scwa_mean + scwa_sem,
                alpha=0.3,
                color="tab:green",
                label="Val SCWA ±SEM",
            )
            plt.xlabel("Epoch")
            plt.ylabel("SCWA")
            plt.ylim(0, 1)
            plt.title(
                f"{dset_name}: Validation SCWA (Mean ± SEM)\nAggregated over {scwa_mat.shape[0]} runs"
            )
            plt.legend()
            save_name = os.path.join(
                working_dir, f"{dset_name}_aggregated_val_SCWA_curve.png"
            )
            plt.savefig(save_name)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated SCWA curve for {dset_name}: {e}")
        plt.close()

    # ----- Print aggregate metrics -----
    try:
        if dset_data["scwa"]:
            final_vals = [arr[-1] for arr in dset_data["scwa"] if len(arr)]
            best_vals = [np.nanmax(arr) for arr in dset_data["scwa"] if len(arr)]
            if final_vals:
                final_mean = np.nanmean(final_vals)
                final_sem = np.nanstd(final_vals) / np.sqrt(len(final_vals))
                best_mean = np.nanmean(best_vals)
                best_sem = np.nanstd(best_vals) / np.sqrt(len(best_vals))
                print(
                    f"{dset_name}: Final Val SCWA = {final_mean:.4f} ± {final_sem:.4f} | "
                    f"Best Val SCWA = {best_mean:.4f} ± {best_sem:.4f} (N={len(final_vals)})"
                )
    except Exception as e:
        print(f"Error printing aggregate metrics for {dset_name}: {e}")

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import f1_score

# ------------------ paths & data ------------------
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# List of experiment_data.npy paths provided by the user
experiment_data_path_list = [
    "experiments/2025-08-17_00-44-36_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_217faea8ea774fbeafcea740fb89c6ec_proc_3168671/experiment_data.npy",
    "experiments/2025-08-17_00-44-36_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_e24188633cba4965b24975e4e9d869bd_proc_3168673/experiment_data.npy",
    "experiments/2025-08-17_00-44-36_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_34ab141a59f04cc4a97c84fdb2e32b55_proc_3168672/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        ed = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_experiment_data.append(ed)
    except Exception as e:
        print(f"Error loading {p}: {e}")

# -----------------------------------------------------------------
# Aggregate dictionaries keyed by dataset name
# -----------------------------------------------------------------
datasets = {}
for run_idx, edict in enumerate(all_experiment_data):
    for ds_name, content in edict.items():
        d = datasets.setdefault(
            ds_name, {"metrics": {}, "epochs": [], "preds": [], "gts": []}
        )
        # store per-run things in lists for later stacking
        for key, arr in content.get("metrics", {}).items():
            d["metrics"].setdefault(key, []).append(np.array(arr))
        d["epochs"].append(np.array(content.get("epochs", [])))
        d["preds"].append(np.array(content.get("predictions", [])))
        d["gts"].append(np.array(content.get("ground_truth", [])))


# -----------------------------------------------------------------
# Helper to compute mean and sem with shape matching
# -----------------------------------------------------------------
def _mean_sem(arr_list):
    stacked = np.stack(arr_list, axis=0)  # shape (runs, len)
    mean = stacked.mean(axis=0)
    sem = (
        stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0])
        if stacked.shape[0] > 1
        else np.zeros_like(mean)
    )
    return mean, sem


# -----------------------------------------------------------------
# Plotting per dataset
# -----------------------------------------------------------------
final_f1_agg = {}  # {ds_name: (mean, sem)}

for ds_name, ds_data in datasets.items():
    epochs_list = ds_data["epochs"]
    # Align epochs by taking the minimum common length
    min_len = min(len(e) for e in epochs_list) if epochs_list else 0
    if min_len == 0:
        continue
    epochs = epochs_list[0][:min_len]

    # -------- Loss Curves (train + val) with SEM ribbons --------
    try:
        train_loss_runs = [
            m[:min_len] for m in ds_data["metrics"].get("train_loss", [])
        ]
        val_loss_runs = [m[:min_len] for m in ds_data["metrics"].get("val_loss", [])]
        if train_loss_runs and val_loss_runs:
            train_mean, train_sem = _mean_sem(train_loss_runs)
            val_mean, val_sem = _mean_sem(val_loss_runs)

            plt.figure()
            plt.plot(epochs, train_mean, color="blue", label="Train Loss (mean)")
            plt.fill_between(
                epochs,
                train_mean - train_sem,
                train_mean + train_sem,
                color="blue",
                alpha=0.3,
                label="Train SEM",
            )
            plt.plot(epochs, val_mean, color="orange", label="Validation Loss (mean)")
            plt.fill_between(
                epochs,
                val_mean - val_sem,
                val_mean + val_sem,
                color="orange",
                alpha=0.3,
                label="Val SEM",
            )
            plt.title(f"{ds_name} Loss Curves (Aggregated)\nMean ± SEM across runs")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{ds_name}_aggregated_loss_curves.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss for {ds_name}: {e}")
        plt.close()

    # -------- Validation F1 with error bars every k epochs --------
    try:
        val_f1_runs = [m[:min_len] for m in ds_data["metrics"].get("val_f1", [])]
        if val_f1_runs:
            f1_mean, f1_sem = _mean_sem(val_f1_runs)
            plt.figure()
            plt.errorbar(
                epochs,
                f1_mean,
                yerr=f1_sem,
                fmt="-o",
                ecolor="gray",
                elinewidth=1,
                capsize=3,
                label="Macro-F1 (mean±SEM)",
            )
            plt.title(f"{ds_name} Validation Macro-F1 (Aggregated)")
            plt.xlabel("Epoch")
            plt.ylabel("Macro-F1")
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{ds_name}_aggregated_val_f1_curve.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated val-F1 for {ds_name}: {e}")
        plt.close()

    # -------- Final test F1 (predictions vs ground truth) --------
    final_f1_list = []
    try:
        for preds, gts in zip(ds_data["preds"], ds_data["gts"]):
            if preds.size and gts.size:
                final_f1_list.append(f1_score(gts, preds, average="macro"))
        if final_f1_list:
            mean_f1 = np.mean(final_f1_list)
            sem_f1 = (
                np.std(final_f1_list, ddof=1) / np.sqrt(len(final_f1_list))
                if len(final_f1_list) > 1
                else 0.0
            )
            final_f1_agg[ds_name] = (mean_f1, sem_f1)
    except Exception as e:
        print(f"Error computing final F1 for {ds_name}: {e}")

# -----------------------------------------------------------------
# Bar chart comparing datasets (mean ± SEM)
# -----------------------------------------------------------------
if final_f1_agg:
    try:
        plt.figure()
        names = list(final_f1_agg.keys())
        means = [final_f1_agg[n][0] for n in names]
        sems = [final_f1_agg[n][1] for n in names]
        x = np.arange(len(names))
        plt.bar(x, means, yerr=sems, capsize=5, color="skyblue")
        plt.xticks(x, names, rotation=45, ha="right")
        plt.ylim(0, 1)
        plt.title("Final Test Macro-F1 Across Datasets\nMean ± SEM over runs")
        plt.ylabel("Macro-F1")
        plt.tight_layout()
        fname = os.path.join(working_dir, "datasets_final_f1_aggregated.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating dataset comparison plot: {e}")
        plt.close()

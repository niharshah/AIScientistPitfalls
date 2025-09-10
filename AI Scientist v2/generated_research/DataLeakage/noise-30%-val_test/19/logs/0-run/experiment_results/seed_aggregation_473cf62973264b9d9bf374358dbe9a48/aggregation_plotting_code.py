import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

# ---------------------------------------------------------------------
# basic setup
# ---------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
# load every experiment_data.npy that actually exists
# ---------------------------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-17_18-48-09_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_595d6e9b770e4115b1ed7b242c547043_proc_3341729/experiment_data.npy",
    "experiments/2025-08-17_18-48-09_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_32dd5b2adbf54c99875f895d48d2b04d_proc_3341727/experiment_data.npy",
    "experiments/2025-08-17_18-48-09_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_5abd3c72bf2d46d0ba9264c6715a2a96_proc_3341730/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_experiment_data.append(data)
    except Exception as e:
        print(f"Error loading {p}: {e}")

# ---------------------------------------------------------------------
# aggregate per (model, dataset)
# ---------------------------------------------------------------------
agg_curves = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
agg_preds = defaultdict(lambda: defaultdict(lambda: {"preds": [], "gts": []}))
agg_tests = defaultdict(lambda: defaultdict(lambda: {"mcc": [], "f1": []}))

for run in all_experiment_data:
    for model_name, datasets in run.items():
        for dset_name, rec in datasets.items():
            # store curves
            agg_curves[model_name][dset_name]["loss_train"].append(
                np.asarray(rec["losses"]["train"])
            )
            agg_curves[model_name][dset_name]["loss_val"].append(
                np.asarray(rec["losses"]["val"])
            )
            agg_curves[model_name][dset_name]["mcc_train"].append(
                np.asarray(rec["metrics"]["train_MCC"])
            )
            agg_curves[model_name][dset_name]["mcc_val"].append(
                np.asarray(rec["metrics"]["val_MCC"])
            )
            # store test metrics
            agg_tests[model_name][dset_name]["mcc"].append(rec.get("test_MCC", np.nan))
            agg_tests[model_name][dset_name]["f1"].append(rec.get("test_F1", np.nan))
            # store predictions for confusion aggregation if available
            if "predictions" in rec and "ground_truth" in rec:
                agg_preds[model_name][dset_name]["preds"].extend(rec["predictions"])
                agg_preds[model_name][dset_name]["gts"].extend(rec["ground_truth"])


# ---------------------------------------------------------------------
# helper to compute mean & sem (aligned to shortest run)
# ---------------------------------------------------------------------
def mean_sem(list_of_1d_arrays):
    min_len = min(arr.shape[0] for arr in list_of_1d_arrays)
    stacked = np.stack([arr[:min_len] for arr in list_of_1d_arrays], axis=0)
    mean = stacked.mean(axis=0)
    sem = stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0])
    return mean, sem, np.arange(1, min_len + 1)


# ---------------------------------------------------------------------
# create aggregate plots
# ---------------------------------------------------------------------
for model_name, dsets in agg_curves.items():
    for dset_name, curves in dsets.items():

        # -------------------- aggregate loss curves --------------------
        try:
            m_tr, se_tr, epochs = mean_sem(curves["loss_train"])
            m_val, se_val, _ = mean_sem(curves["loss_val"])
            plt.figure()
            plt.plot(epochs, m_tr, label="train mean")
            plt.fill_between(
                epochs, m_tr - se_tr, m_tr + se_tr, alpha=0.3, label="train ± SEM"
            )
            plt.plot(epochs, m_val, label="val mean")
            plt.fill_between(
                epochs, m_val - se_val, m_val + se_val, alpha=0.3, label="val ± SEM"
            )
            plt.xlabel("Epoch")
            plt.ylabel("BCE Loss")
            plt.title(
                f"{dset_name} – {model_name}\nLoss (mean ± SEM over {len(curves['loss_train'])} runs)"
            )
            plt.legend()
            fname = f"{dset_name}_{model_name}_loss_curve_aggregate.png"
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(
                f"Error creating aggregate loss plot for {dset_name}-{model_name}: {e}"
            )
            plt.close()

        # -------------------- aggregate MCC curves ---------------------
        try:
            m_tr, se_tr, epochs = mean_sem(curves["mcc_train"])
            m_val, se_val, _ = mean_sem(curves["mcc_val"])
            plt.figure()
            plt.plot(epochs, m_tr, label="train mean")
            plt.fill_between(
                epochs, m_tr - se_tr, m_tr + se_tr, alpha=0.3, label="train ± SEM"
            )
            plt.plot(epochs, m_val, label="val mean")
            plt.fill_between(
                epochs, m_val - se_val, m_val + se_val, alpha=0.3, label="val ± SEM"
            )
            plt.xlabel("Epoch")
            plt.ylabel("MCC")
            plt.title(
                f"{dset_name} – {model_name}\nMCC (mean ± SEM over {len(curves['mcc_train'])} runs)"
            )
            plt.legend()
            fname = f"{dset_name}_{model_name}_mcc_curve_aggregate.png"
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(
                f"Error creating aggregate MCC plot for {dset_name}-{model_name}: {e}"
            )
            plt.close()

        # -------------------- aggregate confusion counts ---------------
        try:
            preds = np.asarray(agg_preds[model_name][dset_name]["preds"])
            gts = np.asarray(agg_preds[model_name][dset_name]["gts"])
            if preds.size and gts.size:
                tp = np.sum((preds == 1) & (gts == 1))
                fp = np.sum((preds == 1) & (gts == 0))
                tn = np.sum((preds == 0) & (gts == 0))
                fn = np.sum((preds == 0) & (gts == 1))
                plt.figure()
                plt.bar(
                    ["TP", "FP", "TN", "FN"],
                    [tp, fp, tn, fn],
                    color=["g", "r", "b", "y"],
                )
                plt.ylabel("Count")
                plt.title(f"{dset_name} – {model_name}\nConfusion Counts (all runs)")
                fname = f"{dset_name}_{model_name}_confusion_counts_aggregate.png"
                plt.tight_layout()
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(
                f"Error creating aggregate confusion plot for {dset_name}-{model_name}: {e}"
            )
            plt.close()

        # -------------------- print aggregated final test metrics -------
        mcc_arr = np.asarray(agg_tests[model_name][dset_name]["mcc"], dtype=float)
        f1_arr = np.asarray(agg_tests[model_name][dset_name]["f1"], dtype=float)
        if mcc_arr.size:
            print(
                f"{dset_name} – {model_name}: "
                f"Test MCC = {np.nanmean(mcc_arr):.3f} ± {np.nanstd(mcc_arr, ddof=1):.3f} "
                f"(N={len(mcc_arr)}) ; "
                f"macro-F1 = {np.nanmean(f1_arr):.3f} ± {np.nanstd(f1_arr, ddof=1):.3f}"
            )

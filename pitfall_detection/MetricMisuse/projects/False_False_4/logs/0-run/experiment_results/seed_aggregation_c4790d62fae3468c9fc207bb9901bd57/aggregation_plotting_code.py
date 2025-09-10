import matplotlib.pyplot as plt
import numpy as np
import os

# ----------- paths -----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------- experiment data paths -----------
experiment_data_path_list = [
    "experiments/2025-08-14_12-19-19_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_050ba0f54fbe48da89ecf753405b7dc4_proc_2640121/experiment_data.npy",
    "experiments/2025-08-14_12-19-19_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_511f8c5402a24872b2e5142f421aae37_proc_2640123/experiment_data.npy",
    "experiments/2025-08-14_12-19-19_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_206b2b90404b411a8a3f07eed9ae3840_proc_2640122/experiment_data.npy",
]

# ----------- data loading ----
all_experiment_data = []
try:
    root = os.getenv("AI_SCIENTIST_ROOT", "")
    for p in experiment_data_path_list:
        full_path = os.path.join(root, p) if root else p
        ed = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(ed)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []


# helper ---------------------------------------------------------------
def aggregate_curves(curve_list):
    """curve_list: list of 1-D np.arrays"""
    if len(curve_list) == 0:
        return None, None
    min_len = min(len(c) for c in curve_list)
    stacked = np.stack(
        [c[:min_len] for c in curve_list], axis=0
    )  # shape (n_runs, min_len)
    mean = np.mean(stacked, axis=0)
    se = (
        np.std(stacked, axis=0, ddof=1) / np.sqrt(stacked.shape[0])
        if stacked.shape[0] > 1
        else np.zeros_like(mean)
    )
    return mean, se


# ---------------------------------------------------------------------
# find union of dataset names across runs
dataset_names = set()
for ed in all_experiment_data:
    dataset_names.update(ed.keys())

# --------------- plotting aggregated results per dataset -------------
for ds_name in dataset_names:
    # Collect curves across runs ------------------------------------------------
    losses_train, losses_val = [], []
    acc_train, acc_val = [], []
    swa_train, swa_val = [], []
    test_metrics_runs = []  # list of dicts

    for ed in all_experiment_data:
        ds = ed.get(ds_name, {})
        # losses
        if "losses" in ds and ds["losses"]:
            if "train" in ds["losses"]:
                losses_train.append(np.asarray(ds["losses"]["train"]))
            if "val" in ds["losses"]:
                losses_val.append(np.asarray(ds["losses"]["val"]))
        # metrics / accuracy
        if "metrics" in ds and ds["metrics"]:
            if "train" in ds["metrics"]:
                acc_train.append(np.asarray(ds["metrics"]["train"]))
            if "val" in ds["metrics"]:
                acc_val.append(np.asarray(ds["metrics"]["val"]))
        # swa
        if "swa" in ds and ds["swa"]:
            if "train" in ds["swa"]:
                swa_train.append(np.asarray(ds["swa"]["train"]))
            if "val" in ds["swa"]:
                swa_val.append(np.asarray(ds["swa"]["val"]))
        # test metrics
        if "test_metrics" in ds:
            test_metrics_runs.append(ds["test_metrics"])

    n_runs = len(all_experiment_data)

    # ------------ 1. Aggregated Loss curves ---------------------------
    try:
        mean_tr, se_tr = aggregate_curves(losses_train)
        mean_val, se_val = aggregate_curves(losses_val)
        if mean_tr is not None and mean_val is not None:
            epochs = np.arange(len(mean_tr))
            plt.figure()
            plt.plot(epochs, mean_tr, label="train mean", color="tab:blue")
            plt.fill_between(
                epochs,
                mean_tr - se_tr,
                mean_tr + se_tr,
                color="tab:blue",
                alpha=0.2,
                label="train ±SE",
            )
            plt.plot(
                epochs, mean_val, label="val mean", color="tab:orange", linestyle="--"
            )
            plt.fill_between(
                epochs,
                mean_val - se_val,
                mean_val + se_val,
                color="tab:orange",
                alpha=0.2,
                label="val ±SE",
            )
            plt.title(f"{ds_name} Aggregated Loss Curves\nMean ± SE over {n_runs} runs")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy")
            plt.legend(fontsize=6)
            fname = f"{ds_name}_aggregated_loss_curves.png"
            plt.savefig(os.path.join(working_dir, fname))
        else:
            raise ValueError("Loss curves missing in one or more runs")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss curves for {ds_name}: {e}")
        plt.close()

    # ------------ 2. Aggregated Accuracy curves -----------------------
    try:
        mean_tr, se_tr = aggregate_curves(acc_train)
        mean_val, se_val = aggregate_curves(acc_val)
        if mean_tr is not None and mean_val is not None:
            epochs = np.arange(len(mean_tr))
            plt.figure()
            plt.plot(epochs, mean_tr, label="train mean", color="tab:green")
            plt.fill_between(
                epochs,
                mean_tr - se_tr,
                mean_tr + se_tr,
                color="tab:green",
                alpha=0.2,
                label="train ±SE",
            )
            plt.plot(
                epochs, mean_val, label="val mean", color="tab:red", linestyle="--"
            )
            plt.fill_between(
                epochs,
                mean_val - se_val,
                mean_val + se_val,
                color="tab:red",
                alpha=0.2,
                label="val ±SE",
            )
            plt.title(
                f"{ds_name} Aggregated Accuracy Curves\nMean ± SE over {n_runs} runs"
            )
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend(fontsize=6)
            fname = f"{ds_name}_aggregated_accuracy_curves.png"
            plt.savefig(os.path.join(working_dir, fname))
        else:
            raise ValueError("Accuracy curves missing in one or more runs")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated accuracy curves for {ds_name}: {e}")
        plt.close()

    # ------------ 3. Aggregated SWA curves ----------------------------
    try:
        mean_tr, se_tr = aggregate_curves(swa_train)
        mean_val, se_val = aggregate_curves(swa_val)
        if mean_tr is not None and mean_val is not None:
            epochs = np.arange(len(mean_tr))
            plt.figure()
            plt.plot(epochs, mean_tr, label="train mean", color="tab:purple")
            plt.fill_between(
                epochs,
                mean_tr - se_tr,
                mean_tr + se_tr,
                color="tab:purple",
                alpha=0.2,
                label="train ±SE",
            )
            plt.plot(
                epochs, mean_val, label="val mean", color="tab:gray", linestyle="--"
            )
            plt.fill_between(
                epochs,
                mean_val - se_val,
                mean_val + se_val,
                color="tab:gray",
                alpha=0.2,
                label="val ±SE",
            )
            plt.title(f"{ds_name} Aggregated SWA Curves\nMean ± SE over {n_runs} runs")
            plt.xlabel("Epoch")
            plt.ylabel("Shape-Weighted Accuracy")
            plt.legend(fontsize=6)
            fname = f"{ds_name}_aggregated_swa_curves.png"
            plt.savefig(os.path.join(working_dir, fname))
        else:
            raise ValueError("SWA curves missing in one or more runs")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated SWA curves for {ds_name}: {e}")
        plt.close()

    # ------------ 4. Aggregated Final Test metrics --------------------
    try:
        if len(test_metrics_runs) > 0:
            bars = ["loss", "acc", "swa"]
            vals = []
            ses = []
            for k in bars:
                arr = np.array(
                    [tm.get(k, np.nan) for tm in test_metrics_runs], dtype=float
                )
                vals.append(np.nanmean(arr))
                if len(arr) > 1:
                    ses.append(np.nanstd(arr, ddof=1) / np.sqrt(len(arr)))
                else:
                    ses.append(0.0)
            x = np.arange(len(bars))
            plt.figure()
            plt.bar(x, vals, yerr=ses, capsize=4, color="skyblue", edgecolor="black")
            plt.xticks(x, bars)
            plt.title(
                f"{ds_name} Aggregated Final Test Metrics\nMean ± SE over {n_runs} runs"
            )
            plt.ylabel("Score")
            fname = f"{ds_name}_aggregated_test_metrics.png"
            plt.savefig(os.path.join(working_dir, fname))
        else:
            raise ValueError("No test_metrics available across runs")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated test metrics for {ds_name}: {e}")
        plt.close()

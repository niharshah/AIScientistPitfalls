import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# 1. Load every experiment result file that is provided
# ------------------------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-30_23-24-25_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_f753388927c949f18b167c7453b3b774_proc_1541632/experiment_data.npy",
    "experiments/2025-08-30_23-24-25_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_540c0d5a0a2e4c2bb3634509fbe3f2d1_proc_1541636/experiment_data.npy",
    "experiments/2025-08-30_23-24-25_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_b889da0427c1404aaa00b01671f8cc24_proc_1541635/experiment_data.npy",
]

all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(data)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# ------------------------------------------------------------------
# 2. Aggregate by dataset name
# ------------------------------------------------------------------
aggregated = {}
for run_dict in all_experiment_data:
    for dname, ddata in run_dict.items():
        bucket = aggregated.setdefault(
            dname,
            {
                "train_losses": [],
                "val_losses": [],
                "cwa": [],
                "swa": [],
                "hwa": [],
                "test_metrics": [],
            },
        )
        bucket["train_losses"].append(
            np.asarray(ddata["losses"].get("train", []), dtype=float)
        )
        bucket["val_losses"].append(
            np.asarray(ddata["losses"].get("val", []), dtype=float)
        )

        val_metrics = ddata["metrics"].get("val", [])
        bucket["cwa"].append(
            np.asarray([m["cwa"] for m in val_metrics])
            if val_metrics
            else np.asarray([])
        )
        bucket["swa"].append(
            np.asarray([m["swa"] for m in val_metrics])
            if val_metrics
            else np.asarray([])
        )
        bucket["hwa"].append(
            np.asarray([m["hwa"] for m in val_metrics])
            if val_metrics
            else np.asarray([])
        )

        test_metrics = ddata["metrics"].get("test", {})
        if test_metrics:
            bucket["test_metrics"].append(test_metrics)


# ------------------------------------------------------------------
# 3. Helper: compute mean & sem after truncating to the minimum length
# ------------------------------------------------------------------
def mean_sem(list_of_arrays):
    if not list_of_arrays:
        return np.asarray([]), np.asarray([])
    min_len = min(arr.shape[0] for arr in list_of_arrays if arr.size)
    if min_len == 0:
        return np.asarray([]), np.asarray([])
    stacked = np.vstack([arr[:min_len] for arr in list_of_arrays])
    mean = stacked.mean(axis=0)
    sem = stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0])
    return mean, sem


# ------------------------------------------------------------------
# 4. Create aggregated plots
# ------------------------------------------------------------------
for dname, ddata in aggregated.items():
    # ---------- a) Loss curves -------------------------------------------------
    try:
        train_mean, train_sem = mean_sem(ddata["train_losses"])
        val_mean, val_sem = mean_sem(ddata["val_losses"])
        if train_mean.size and val_mean.size:
            epochs = np.arange(1, len(train_mean) + 1)
            plt.figure()
            plt.plot(epochs, train_mean, label="Train Mean", color="tab:blue")
            plt.fill_between(
                epochs,
                train_mean - train_sem,
                train_mean + train_sem,
                color="tab:blue",
                alpha=0.25,
                label="Train ±1 SEM",
            )
            plt.plot(epochs, val_mean, label="Val Mean", color="tab:orange")
            plt.fill_between(
                epochs,
                val_mean - val_sem,
                val_mean + val_sem,
                color="tab:orange",
                alpha=0.25,
                label="Val ±1 SEM",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(
                f"{dname} Aggregated Loss Curves\n(n={len(ddata['train_losses'])} runs)"
            )
            plt.legend()
            fname = f"{dname}_aggregated_loss_curves.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dname}: {e}")
        plt.close()

    # ---------- b) Validation metric curves -----------------------------------
    try:
        has_any_metric = False
        plt.figure()
        for metric_key, color in zip(
            ["cwa", "swa", "hwa"], ["tab:green", "tab:red", "tab:purple"]
        ):
            mean_curve, sem_curve = mean_sem(ddata[metric_key])
            if mean_curve.size:
                has_any_metric = True
                epochs = np.arange(1, len(mean_curve) + 1)
                plt.plot(
                    epochs, mean_curve, label=f"{metric_key.upper()} Mean", color=color
                )
                plt.fill_between(
                    epochs,
                    mean_curve - sem_curve,
                    mean_curve + sem_curve,
                    color=color,
                    alpha=0.25,
                    label=f"{metric_key.upper()} ±1 SEM",
                )
        if has_any_metric:
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(
                f"{dname} Aggregated Validation Metrics\n(n={len(ddata['cwa'])} runs)"
            )
            plt.legend()
            fname = f"{dname}_aggregated_metric_curves.png"
            plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated metric plot for {dname}: {e}")
        plt.close()

    # ---------- c) Print mean±std of test metrics -----------------------------
    try:
        if ddata["test_metrics"]:
            # collect keys that exist in every dict
            keys = set.intersection(*[set(tm.keys()) for tm in ddata["test_metrics"]])
            for k in sorted(keys):
                vals = np.asarray([tm[k] for tm in ddata["test_metrics"]], dtype=float)
                print(
                    f"{dname} TEST {k.upper()}: {vals.mean():.3f} ± {vals.std(ddof=1):.3f} "
                    f"(n={len(vals)})"
                )
    except Exception as e:
        print(f"Error summarising test metrics for {dname}: {e}")

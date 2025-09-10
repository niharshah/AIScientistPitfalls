import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------
# Set-up
# ------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Paths provided in the problem statement
experiment_data_path_list = [
    "experiments/2025-08-15_16-42-50_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_3e65bfc70c8d48d69938bfe9d3aa4420_proc_2942475/experiment_data.npy",
    "experiments/2025-08-15_16-42-50_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_e7081880f2ae47f6975aae4e3194d7b0_proc_2942476/experiment_data.npy",
    "experiments/2025-08-15_16-42-50_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_3d1baac6cf0e47c1813dc4d635b6e3a8_proc_2942477/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(data)
    except Exception as e:
        print(f"Error loading experiment data from {p}: {e}")

# ------------------------------------------------------------
# Aggregate runs by dataset name
# ------------------------------------------------------------
agg = (
    {}
)  # {dataset : {'train_loss': [...], 'val_loss': [...], 'val_metrics': [...], 'test_metrics': [...]}}

for run in all_experiment_data:
    for dset, rec in run.items():
        losses = rec.get("losses", {})
        metrics = rec.get("metrics", {})
        agg.setdefault(
            dset,
            {"train_loss": [], "val_loss": [], "val_metrics": [], "test_metrics": []},
        )
        if losses.get("train"):
            agg[dset]["train_loss"].append(np.array(losses["train"]))
        if losses.get("val"):
            agg[dset]["val_loss"].append(np.array(losses["val"]))
        if metrics.get("val"):
            # store list of list-of-dicts → easier to align later
            agg[dset]["val_metrics"].append(metrics["val"])
        if metrics.get("test"):
            agg[dset]["test_metrics"].append(metrics["test"])


# ------------------------------------------------------------
# Helper to truncate all runs to the minimum common length
# ------------------------------------------------------------
def align_arrays(arr_list):
    if not arr_list:
        return []
    min_len = min(len(a) for a in arr_list)
    return np.stack([a[:min_len] for a in arr_list], axis=0)  # shape (runs, epochs)


def align_val_metrics(runs_val_metrics, key):
    """returns ndarray shape (runs, epochs) for given metric key"""
    if not runs_val_metrics:
        return np.array([])
    min_len = min(len(r) for r in runs_val_metrics)
    data = []
    for r in runs_val_metrics:
        vals = [m[key] for m in r[:min_len]]
        data.append(vals)
    return np.array(data)  # (runs, epochs)


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------
for dname, ddata in agg.items():
    # --------------------- Loss curves ---------------------
    try:
        train_mat = align_arrays(ddata["train_loss"])
        val_mat = align_arrays(ddata["val_loss"])
        if train_mat.size or val_mat.size:
            epochs = np.arange(
                min(
                    train_mat.shape[1] if train_mat.size else val_mat.shape[1],
                    val_mat.shape[1] if val_mat.size else train_mat.shape[1],
                )
            )
            plt.figure()
            if train_mat.size:
                mean_tr = train_mat.mean(0)
                sem_tr = train_mat.std(0, ddof=1) / np.sqrt(train_mat.shape[0])
                plt.plot(epochs, mean_tr, label="Train Loss (mean)")
                plt.fill_between(
                    epochs,
                    mean_tr - sem_tr,
                    mean_tr + sem_tr,
                    alpha=0.3,
                    label="Train ± SEM",
                )
            if val_mat.size:
                mean_val = val_mat.mean(0)
                sem_val = val_mat.std(0, ddof=1) / np.sqrt(val_mat.shape[0])
                plt.plot(epochs, mean_val, label="Val Loss (mean)")
                plt.fill_between(
                    epochs,
                    mean_val - sem_val,
                    mean_val + sem_val,
                    alpha=0.3,
                    label="Val ± SEM",
                )
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(
                f"{dname} Mean Training and Validation Loss\n(Mean ± SEM across {max(train_mat.shape[0], val_mat.shape[0])} runs)"
            )
            plt.legend()
            plt.tight_layout()
            save_path = os.path.join(working_dir, f"{dname}_agg_loss_curves.png")
            plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dname}: {e}")
        plt.close()

    # --------------------- Validation metrics ---------------------
    try:
        val_runs = ddata["val_metrics"]
        if val_runs:
            crwa_mat = align_val_metrics(val_runs, "CRWA")
            swa_mat = align_val_metrics(val_runs, "SWA")
            cwa_mat = align_val_metrics(val_runs, "CWA")
            epochs = np.arange(
                min(crwa_mat.shape[1], swa_mat.shape[1], cwa_mat.shape[1])
            )
            plt.figure()
            for mat, name, color in [
                (crwa_mat, "CRWA", "tab:blue"),
                (swa_mat, "SWA", "tab:orange"),
                (cwa_mat, "CWA", "tab:green"),
            ]:
                if mat.size:
                    mean = mat.mean(0)
                    sem = mat.std(0, ddof=1) / np.sqrt(mat.shape[0])
                    plt.plot(epochs, mean, label=f"{name} (mean)", color=color)
                    plt.fill_between(
                        epochs,
                        mean - sem,
                        mean + sem,
                        color=color,
                        alpha=0.3,
                        label=f"{name} ± SEM",
                    )
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title(
                f"{dname} Validation Metrics\n(Mean ± SEM across {len(val_runs)} runs)"
            )
            plt.legend()
            plt.tight_layout()
            save_path = os.path.join(working_dir, f"{dname}_agg_val_metric_curves.png")
            plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated validation metric plot for {dname}: {e}")
        plt.close()

    # --------------------- Test metrics ---------------------
    try:
        test_runs = ddata["test_metrics"]
        if test_runs:
            metric_names = list(test_runs[0].keys())
            values = np.array(
                [[run[m] for m in metric_names] for run in test_runs]
            )  # shape (runs, metrics)
            means = values.mean(0)
            sems = values.std(0, ddof=1) / np.sqrt(values.shape[0])
            plt.figure()
            x = np.arange(len(metric_names))
            plt.bar(x, means, yerr=sems, capsize=5)
            plt.xticks(x, metric_names)
            plt.ylim(0, 1)
            plt.ylabel("Score")
            plt.title(
                f"{dname} Final Test Metrics\n(Mean ± SEM across {values.shape[0]} runs)"
            )
            plt.tight_layout()
            save_path = os.path.join(working_dir, f"{dname}_agg_test_metrics_bar.png")
            plt.savefig(save_path)
            print(
                f"{dname} aggregated test metrics (mean):",
                {m: round(means[i], 4) for i, m in enumerate(metric_names)},
            )
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated test metric bar for {dname}: {e}")
        plt.close()

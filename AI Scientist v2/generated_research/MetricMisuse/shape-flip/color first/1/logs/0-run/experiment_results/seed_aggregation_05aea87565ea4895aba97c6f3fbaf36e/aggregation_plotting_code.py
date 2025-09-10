import matplotlib.pyplot as plt
import numpy as np
import os
import math

# ------------------------------------------------------------------
# directory / data loading
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data_path_list = [
    "experiments/2025-08-30_17-49-30_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_3d72bce5bcc043758ff7ed4905aa4f0c_proc_1445295/experiment_data.npy",
    "experiments/2025-08-30_17-49-30_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_1dc2bcaf11bf440e86d9dd3e95028c99_proc_1445296/experiment_data.npy",
    "experiments/2025-08-30_17-49-30_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_a0d62a1c0b394cd59e9e875ad0c9dc74_proc_1445297/experiment_data.npy",
]

all_runs_by_dataset = {}

# ------------------ Load every file ------------------
for p in experiment_data_path_list:
    try:
        abs_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        edict = np.load(abs_path, allow_pickle=True).item()
        for dset_name, run in edict.items():
            all_runs_by_dataset.setdefault(dset_name, []).append(run)
    except Exception as e:
        print(f"Error loading {p}: {e}")


# ------------------------------------------------------------------
# Utility to make mean ± s.e.m.  Truncates to min length.
def stack_and_trim(list_of_lists):
    if not list_of_lists:
        return None
    min_len = min(len(x) for x in list_of_lists)
    arr = np.stack([np.asarray(x[:min_len]) for x in list_of_lists])
    return arr


# ------------------------------------------------------------------
# Iterate over datasets and create aggregated plots
for dset_name, runs in all_runs_by_dataset.items():
    # ----------------------- collect series -----------------------
    train_losses = stack_and_trim([r["losses"]["train"] for r in runs if "losses" in r])
    val_losses = stack_and_trim([r["losses"]["val"] for r in runs if "losses" in r])
    epochs_vec = np.arange(train_losses.shape[1]) if train_losses is not None else None

    cwa = stack_and_trim(
        [[m["CWA"] for m in r["metrics"]["val"]] for r in runs if "metrics" in r]
    )
    swa = stack_and_trim(
        [[m["SWA"] for m in r["metrics"]["val"]] for r in runs if "metrics" in r]
    )
    hpa = stack_and_trim(
        [[m["HPA"] for m in r["metrics"]["val"]] for r in runs if "metrics" in r]
    )

    preds_all = np.concatenate([np.asarray(r.get("predictions", [])) for r in runs])
    gts_all = np.concatenate([np.asarray(r.get("ground_truth", [])) for r in runs])

    n_runs = len(runs)
    # ----------------------- 1. Loss plot -------------------------
    try:
        if train_losses is not None and val_losses is not None:
            plt.figure(figsize=(6, 4), dpi=120)
            mean_tr = train_losses.mean(axis=0)
            sem_tr = train_losses.std(axis=0, ddof=1) / math.sqrt(n_runs)
            mean_val = val_losses.mean(axis=0)
            sem_val = val_losses.std(axis=0, ddof=1) / math.sqrt(n_runs)

            plt.fill_between(
                epochs_vec,
                mean_tr - sem_tr,
                mean_tr + sem_tr,
                alpha=0.2,
                label="Train ± s.e.m.",
            )
            plt.plot(epochs_vec, mean_tr, label="Train mean", color="blue")
            plt.fill_between(
                epochs_vec,
                mean_val - sem_val,
                mean_val + sem_val,
                alpha=0.2,
                label="Val ± s.e.m.",
                color="orange",
            )
            plt.plot(epochs_vec, mean_val, label="Val mean", color="orange")

            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(
                f"{dset_name} – Aggregated Train/Val Loss\nShaded: ±1 standard error over {n_runs} runs"
            )
            plt.legend()
            fname = os.path.join(working_dir, f"{dset_name}_agg_loss.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dset_name}: {e}")
        plt.close()

    # ----------------------- 2. Accuracy metrics ------------------
    try:
        if cwa is not None and swa is not None and hpa is not None:
            plt.figure(figsize=(10, 4), dpi=120)
            metrics = {"CWA": cwa, "SWA": swa, "HPA": hpa}
            colors = {"CWA": "tab:blue", "SWA": "tab:green", "HPA": "tab:red"}
            for i, (name, arr) in enumerate(metrics.items()):
                mean = arr.mean(axis=0)
                sem = arr.std(axis=0, ddof=1) / math.sqrt(n_runs)
                plt.fill_between(
                    epochs_vec, mean - sem, mean + sem, alpha=0.15, color=colors[name]
                )
                plt.plot(epochs_vec, mean, label=f"{name} mean", color=colors[name])
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title(f"{dset_name} – Validation Metrics (mean ± s.e.m., n={n_runs})")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset_name}_agg_val_metrics.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating metric plot for {dset_name}: {e}")
        plt.close()

    # ----------------------- 3. Label distribution ---------------
    try:
        if preds_all.size and gts_all.size:
            labels = sorted(set(gts_all))
            gt_counts = [(gts_all == l).sum() for l in labels]
            pred_counts = [(preds_all == l).sum() for l in labels]

            width = 0.35
            x = np.arange(len(labels))
            plt.figure(figsize=(6, 4), dpi=120)
            plt.bar(x - width / 2, gt_counts, width=width, label="Ground Truth")
            plt.bar(x + width / 2, pred_counts, width=width, label="Predictions")
            plt.xticks(x, labels)
            plt.xlabel("Class Label")
            plt.ylabel("Count (aggregated)")
            plt.title(
                f"{dset_name} – Label Frequencies\nLeft: Ground Truth, Right: Predictions (all runs)"
            )
            plt.legend()
            fname = os.path.join(working_dir, f"{dset_name}_agg_label_distribution.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating label distribution for {dset_name}: {e}")
        plt.close()

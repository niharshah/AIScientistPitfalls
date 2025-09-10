import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------------------------------------------------------------
# Collect every experiment_data.npy supplied by the host
experiment_data_path_list = [
    "experiments/2025-08-15_16-42-50_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_845932789bf54c1a970c17933c0314ec_proc_2952779/experiment_data.npy",
    "experiments/2025-08-15_16-42-50_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_7b944347b81c460387c55a55dfe4564d_proc_2952778/experiment_data.npy",
    "experiments/2025-08-15_16-42-50_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_d5fd905b777f435abca27b171b4f8fa8_proc_2952777/experiment_data.npy",
]

all_experiment_data = []
for rel_path in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), rel_path)
        ed = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(ed)
    except Exception as e:
        print(f"Error loading {rel_path}: {e}")


# ----------------------------------------------------------------------
# Helper to flatten entries into a dict keyed by (model, dataset)
def _flatten_entries(edict):
    out = {}
    for model_k, model_v in edict.items():
        for dset_k, dset_v in model_v.items():
            out.setdefault((model_k, dset_k), []).append(dset_v)
    return out


grouped_runs = {}
for exp in all_experiment_data:
    grouped = _flatten_entries(exp)
    for k, v in grouped.items():
        grouped_runs.setdefault(k, []).extend(v)

if not grouped_runs:
    print("No experiment data found, nothing to aggregate.")
# ----------------------------------------------------------------------
for (model_name, dset_name), runs in grouped_runs.items():
    # Collect arrays; first infer global epoch grid (assume identical grids)
    epoch_lists = [np.array(r["epochs"]) for r in runs]
    # Use the shortest epoch array length to align
    min_len = min(len(e) for e in epoch_lists)
    epochs = epoch_lists[0][:min_len]

    def _stack(key_path):
        arrs = []
        for r in runs:
            a = r
            for k in key_path:
                a = a[k]
            arrs.append(np.array(a)[:min_len])
        return np.vstack(arrs).astype(float)

    train_stack = _stack(["losses", "train"])
    val_stack = _stack(["losses", "val"])
    swa_stack = _stack(["metrics", "val"])

    # Mean and SEM
    def mean_sem(stack):
        mean = np.nanmean(stack, axis=0)
        sem = np.nanstd(stack, axis=0, ddof=1) / np.sqrt(stack.shape[0])
        return mean, sem

    train_mean, train_sem = mean_sem(train_stack)
    val_mean, val_sem = mean_sem(val_stack)
    swa_mean, swa_sem = mean_sem(swa_stack)

    # ------------------- Plot 1: Loss curves (mean ± SEM) ----------------
    try:
        plt.figure()
        plt.plot(epochs, train_mean, label="Train Loss (mean)", color="tab:blue")
        plt.fill_between(
            epochs,
            train_mean - train_sem,
            train_mean + train_sem,
            alpha=0.3,
            color="tab:blue",
            label="Train Loss (SEM)",
        )
        plt.plot(epochs, val_mean, label="Val Loss (mean)", color="tab:orange")
        plt.fill_between(
            epochs,
            val_mean - val_sem,
            val_mean + val_sem,
            alpha=0.3,
            color="tab:orange",
            label="Val Loss (SEM)",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dset_name}: Train vs Val Loss (mean ± SEM) [{model_name}]")
        plt.legend()
        fname = f"{dset_name}_{model_name}_AGG_loss_mean_sem.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dset_name}: {e}")
        plt.close()

    # --------------- Plot 2: Validation SWA (mean ± SEM) -----------------
    try:
        plt.figure()
        plt.plot(
            epochs, swa_mean, marker="o", label="Val SWA (mean)", color="tab:green"
        )
        plt.fill_between(
            epochs,
            swa_mean - swa_sem,
            swa_mean + swa_sem,
            alpha=0.3,
            color="tab:green",
            label="Val SWA (SEM)",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title(f"{dset_name}: Validation SWA (mean ± SEM) [{model_name}]")
        plt.legend()
        fname = f"{dset_name}_{model_name}_AGG_val_SWA_mean_sem.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated SWA plot for {dset_name}: {e}")
        plt.close()

    # ------------------- Optional Histogram snapshots --------------------
    # Plot at no more than 5 evenly spaced epochs (including last)
    try:
        num_snaps = min(5, len(epochs))
        snap_indices = np.linspace(0, len(epochs) - 1, num_snaps, dtype=int)
        for idx in snap_indices:
            # Each run’s prediction counts; average across runs
            label_counts = {}
            for r in runs:
                gt = r["ground_truth"][idx]
                pr = r["predictions"][idx]
                labels = set(gt) | set(pr)
                for l in labels:
                    label_counts.setdefault(l, {"gt": [], "pr": []})
                    label_counts[l]["gt"].append(gt.count(l))
                    label_counts[l]["pr"].append(pr.count(l))
            labels_sorted = sorted(label_counts.keys())
            gt_means = [np.mean(label_counts[l]["gt"]) for l in labels_sorted]
            pr_means = [np.mean(label_counts[l]["pr"]) for l in labels_sorted]

            x = np.arange(len(labels_sorted))
            width = 0.35
            plt.figure()
            plt.bar(x - width / 2, gt_means, width, label="Ground Truth (mean)")
            plt.bar(x + width / 2, pr_means, width, label="Predicted (mean)")
            plt.xlabel("Label ID")
            plt.ylabel("Mean Count")
            plt.title(
                f"{dset_name}: Label Dist. Epoch {epochs[idx]} (mean of runs) [{model_name}]"
            )
            plt.xticks(x, labels_sorted)
            plt.legend()
            fname = f"{dset_name}_{model_name}_AGG_hist_epoch{epochs[idx]}.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error creating histogram snapshots for {dset_name}: {e}")
        plt.close()

    # ------------------- Print summary metrics ---------------------------
    try:
        best_idx = np.nanargmin(val_mean)
        best_val_loss = val_mean[best_idx]
        best_swa = swa_mean[best_idx]
        print(
            f"[{model_name} | {dset_name}] Best mean Val Loss={best_val_loss:.4f} "
            f"at epoch {int(epochs[best_idx])} | Mean SWA={best_swa:.4f}"
        )
    except Exception as e:
        print(f"Error computing summary metrics for {dset_name}: {e}")

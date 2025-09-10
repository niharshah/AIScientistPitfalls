import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------ #
# basic set-up
# ------------------------------------------------------------------ #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------ #
# paths to all experiment_data.npy files (relative to AI_SCIENTIST_ROOT)
# ------------------------------------------------------------------ #
experiment_data_path_list = [
    "experiments/2025-08-30_21-49-55_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_b66e515a4e894113af85de89da506339_proc_1520779/experiment_data.npy",
    "experiments/2025-08-30_21-49-55_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_ab50cdf8e6f246d9bd677026563e8b10_proc_1520778/experiment_data.npy",
    "experiments/2025-08-30_21-49-55_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_9366a250390a460eabd453871340cb61_proc_1520781/experiment_data.npy",
]

# ------------------------------------------------------------------ #
# load all runs
# ------------------------------------------------------------------ #
all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        ed = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(ed)
except Exception as e:
    print(f"Error loading experiment data: {e}")


# ------------------------------------------------------------------ #
# helper to right-pad with nan so all arrays share length
# ------------------------------------------------------------------ #
def pad_to(arr, L):
    arr = np.asarray(arr, dtype=float)
    if arr.size < L:
        arr = np.concatenate([arr, np.full(L - arr.size, np.nan)])
    return arr[:L]


# ------------------------------------------------------------------ #
# aggregate per dataset
# ------------------------------------------------------------------ #
datasets = set()
for run in all_experiment_data:
    datasets.update(run.keys())

for dname in datasets:
    # collect lists over runs
    loss_tr_runs, loss_val_runs = [], []
    hwa_tr_runs, hwa_val_runs = [], []
    cwa_last, swa_last, hwa_last = [], [], []
    preds_runs, gts_runs = [], []
    epochs_master = None
    for run in all_experiment_data:
        ed = run.get(dname, {})
        if not ed:
            continue
        epochs = ed.get("epochs", [])
        if epochs_master is None or len(epochs) > len(epochs_master):
            epochs_master = epochs  # keep the longest as reference
        loss_tr_runs.append(ed.get("losses", {}).get("train", []))
        loss_val_runs.append(ed.get("losses", {}).get("val", []))
        hwa_tr_runs.append(
            [m.get("HWA", np.nan) for m in ed.get("metrics", {}).get("train", [])]
        )
        hwa_val_runs.append(
            [m.get("HWA", np.nan) for m in ed.get("metrics", {}).get("val", [])]
        )

        val_metrics = ed.get("metrics", {}).get("val", [])
        if val_metrics:
            cwa_last.append(val_metrics[-1].get("CWA", np.nan))
            swa_last.append(val_metrics[-1].get("SWA", np.nan))
            hwa_last.append(val_metrics[-1].get("HWA", np.nan))

        if ed.get("predictions") is not None and ed.get("ground_truth") is not None:
            preds_runs.append(np.asarray(ed["predictions"], dtype=int))
            gts_runs.append(np.asarray(ed["ground_truth"], dtype=int))

    # skip if no runs for this dataset
    if epochs_master is None:
        continue

    L = len(epochs_master)
    epochs_master = np.asarray(epochs_master)

    # convert lists to 2D arrays with padding
    def stack_and_stat(list_of_lists):
        if not list_of_lists:
            return None, None
        stacked = np.vstack([pad_to(x, L) for x in list_of_lists])
        mean = np.nanmean(stacked, axis=0)
        sem = np.nanstd(stacked, axis=0) / np.sqrt(stacked.shape[0])
        return mean, sem

    loss_tr_mean, loss_tr_sem = stack_and_stat(loss_tr_runs)
    loss_val_mean, loss_val_sem = stack_and_stat(loss_val_runs)
    hwa_tr_mean, hwa_tr_sem = stack_and_stat(hwa_tr_runs)
    hwa_val_mean, hwa_val_sem = stack_and_stat(hwa_val_runs)

    # ------------------------------------------------------------------ #
    # 1) Aggregated Train/Val loss with SEM
    # ------------------------------------------------------------------ #
    try:
        if loss_tr_mean is not None and loss_val_mean is not None:
            plt.figure()
            plt.plot(epochs_master, loss_tr_mean, label="Train µ")
            plt.fill_between(
                epochs_master,
                loss_tr_mean - loss_tr_sem,
                loss_tr_mean + loss_tr_sem,
                alpha=0.3,
                label="Train ±SEM",
            )
            plt.plot(epochs_master, loss_val_mean, label="Val µ")
            plt.fill_between(
                epochs_master,
                loss_val_mean - loss_val_sem,
                loss_val_mean + loss_val_sem,
                alpha=0.3,
                label="Val ±SEM",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{dname} – Aggregated Train vs Val Loss (mean ± SEM)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, f"{dname}_agg_loss_curve.png"), dpi=150
            )
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss curve for {dname}: {e}")
        plt.close()

    # ------------------------------------------------------------------ #
    # 2) Aggregated Train/Val HWA with SEM
    # ------------------------------------------------------------------ #
    try:
        if hwa_tr_mean is not None and hwa_val_mean is not None:
            plt.figure()
            plt.plot(epochs_master, hwa_tr_mean, label="Train HWA µ")
            plt.fill_between(
                epochs_master,
                hwa_tr_mean - hwa_tr_sem,
                hwa_tr_mean + hwa_tr_sem,
                alpha=0.3,
                label="Train ±SEM",
            )
            plt.plot(epochs_master, hwa_val_mean, label="Val HWA µ")
            plt.fill_between(
                epochs_master,
                hwa_val_mean - hwa_val_sem,
                hwa_val_mean + hwa_val_sem,
                alpha=0.3,
                label="Val ±SEM",
            )
            plt.xlabel("Epoch")
            plt.ylabel("HWA")
            plt.title(f"{dname} – Aggregated HWA (mean ± SEM)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, f"{dname}_agg_HWA_curve.png"), dpi=150
            )
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated HWA curve for {dname}: {e}")
        plt.close()

    # ------------------------------------------------------------------ #
    # 3) Final metrics bar chart (mean ± SEM)
    # ------------------------------------------------------------------ #
    try:
        if cwa_last and swa_last and hwa_last:
            cats = ["CWA", "SWA", "HWA"]
            means = [np.nanmean(cwa_last), np.nanmean(swa_last), np.nanmean(hwa_last)]
            sems = [
                np.nanstd(cwa_last) / np.sqrt(len(cwa_last)),
                np.nanstd(swa_last) / np.sqrt(len(swa_last)),
                np.nanstd(hwa_last) / np.sqrt(len(hwa_last)),
            ]
            plt.figure()
            plt.bar(
                cats,
                means,
                yerr=sems,
                capsize=5,
                color=["#4c72b0", "#55a868", "#c44e52"],
            )
            plt.ylabel("Metric Value")
            plt.title(f"{dname} – Final Validation Metrics (mean ± SEM)")
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, f"{dname}_agg_final_val_metrics.png"), dpi=150
            )
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated metric bar chart for {dname}: {e}")
        plt.close()

    # ------------------------------------------------------------------ #
    # 4) Aggregated confusion matrix (sum then normalised)
    # ------------------------------------------------------------------ #
    try:
        if preds_runs and gts_runs:
            n_cls = int(
                max([max(p.max(), g.max()) for p, g in zip(preds_runs, gts_runs)]) + 1
            )
            cm_sum = np.zeros((n_cls, n_cls), dtype=int)
            for p, g in zip(preds_runs, gts_runs):
                for t, pr in zip(g, p):
                    cm_sum[t, pr] += 1
            cm_norm = cm_sum / cm_sum.sum(axis=1, keepdims=True)
            plt.figure()
            im = plt.imshow(cm_norm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(
                f"{dname} – Aggregated Confusion Matrix\nLeft: Ground Truth, Right: Predicted"
            )
            for i in range(n_cls):
                for j in range(n_cls):
                    plt.text(
                        j,
                        i,
                        f"{cm_norm[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="white" if cm_norm[i, j] > cm_norm.max() / 2 else "black",
                        fontsize=8,
                    )
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, f"{dname}_agg_confusion_matrix.png"), dpi=150
            )
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated confusion matrix for {dname}: {e}")
        plt.close()

    # ------------------------------------------------------------------ #
    # quick terminal summary
    # ------------------------------------------------------------------ #
    if hwa_last:
        print(f"{dname} – Final-epoch HWA mean across runs: {np.nanmean(hwa_last):.4f}")

import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------ #
# basic set-up
# ------------------------------------------------------------------ #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------ #
# load all experiment files
# ------------------------------------------------------------------ #
try:
    experiment_data_path_list = [
        "experiments/2025-08-30_21-49-55_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_f41905065d8a4fbd9f353f7f0180a312_proc_1517534/experiment_data.npy",
        "experiments/2025-08-30_21-49-55_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_dbd88fdb1d754887afbc6c9d3142ab63_proc_1517536/experiment_data.npy",
        "experiments/2025-08-30_21-49-55_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_794ebaec214e44dbb4af45d4b75d42c2_proc_1517537/experiment_data.npy",
    ]
    all_experiment_data = []
    for p in experiment_data_path_list:
        exp_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp = np.load(exp_path, allow_pickle=True).item()
        all_experiment_data.append(exp)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []


# ------------------------------------------------------------------ #
# helper to gather per-run arrays (variable length allowed)
# ------------------------------------------------------------------ #
def stack_with_nan(list_of_lists, dtype=float):
    max_len = max(len(l) for l in list_of_lists) if list_of_lists else 0
    stacked = np.full((len(list_of_lists), max_len), np.nan, dtype=dtype)
    for i, arr in enumerate(list_of_lists):
        stacked[i, : len(arr)] = arr
    return stacked


dataset_name = "SPR"
runs_loss_tr, runs_loss_val = [], []
runs_hwa_tr, runs_hwa_val = [], []
final_cwa, final_swa, final_hwa = [], []
test_accuracies = []

for exp in all_experiment_data:
    ed = exp.get(dataset_name, {})
    epochs = ed.get("epochs", [])
    losses = ed.get("losses", {})
    mets = ed.get("metrics", {})
    runs_loss_tr.append(losses.get("train", []))
    runs_loss_val.append(losses.get("val", []))
    runs_hwa_tr.append([m.get("HWA", np.nan) for m in mets.get("train", [])])
    runs_hwa_val.append([m.get("HWA", np.nan) for m in mets.get("val", [])])

    # final metrics
    if mets.get("val", []):
        final = mets["val"][-1]
        final_cwa.append(final.get("CWA", np.nan))
        final_swa.append(final.get("SWA", np.nan))
        final_hwa.append(final.get("HWA", np.nan))

    # accuracy
    preds = np.asarray(ed.get("predictions", []), dtype=int)
    gts = np.asarray(ed.get("ground_truth", []), dtype=int)
    if preds.size and gts.size:
        test_accuracies.append((preds == gts).mean())

# ------------------------------------------------------------------ #
# Aggregate arrays
# ------------------------------------------------------------------ #
loss_tr_mat = stack_with_nan(runs_loss_tr)
loss_val_mat = stack_with_nan(runs_loss_val)
hwa_tr_mat = stack_with_nan(runs_hwa_tr)
hwa_val_mat = stack_with_nan(runs_hwa_val)

mean_loss_tr = np.nanmean(loss_tr_mat, axis=0)
sem_loss_tr = np.nanstd(loss_tr_mat, axis=0, ddof=1) / np.sqrt(
    np.sum(~np.isnan(loss_tr_mat), axis=0)
)
mean_loss_val = np.nanmean(loss_val_mat, axis=0)
sem_loss_val = np.nanstd(loss_val_mat, axis=0, ddof=1) / np.sqrt(
    np.sum(~np.isnan(loss_val_mat), axis=0)
)

mean_hwa_tr = np.nanmean(hwa_tr_mat, axis=0)
sem_hwa_tr = np.nanstd(hwa_tr_mat, axis=0, ddof=1) / np.sqrt(
    np.sum(~np.isnan(hwa_tr_mat), axis=0)
)
mean_hwa_val = np.nanmean(hwa_val_mat, axis=0)
sem_hwa_val = np.nanstd(hwa_val_mat, axis=0, ddof=1) / np.sqrt(
    np.sum(~np.isnan(hwa_val_mat), axis=0)
)

epochs_axis = np.arange(len(mean_loss_tr))

# ------------------------------------------------------------------ #
# 1) Aggregated Train / Val loss curve with SEM
# ------------------------------------------------------------------ #
try:
    plt.figure()
    plt.plot(epochs_axis, mean_loss_tr, label="Train Mean", color="#1f77b4")
    plt.fill_between(
        epochs_axis,
        mean_loss_tr - sem_loss_tr,
        mean_loss_tr + sem_loss_tr,
        color="#1f77b4",
        alpha=0.2,
        label="Train SEM",
    )
    plt.plot(epochs_axis, mean_loss_val, label="Val Mean", color="#ff7f0e")
    plt.fill_between(
        epochs_axis,
        mean_loss_val - sem_loss_val,
        mean_loss_val + sem_loss_val,
        color="#ff7f0e",
        alpha=0.2,
        label="Val SEM",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR – Train vs Val Loss (Mean ± SEM over runs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_agg_loss_curve.png"), dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss curve: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 2) Aggregated Train / Val HWA curve with SEM
# ------------------------------------------------------------------ #
try:
    plt.figure()
    plt.plot(epochs_axis, mean_hwa_tr, label="Train HWA Mean", color="#2ca02c")
    plt.fill_between(
        epochs_axis,
        mean_hwa_tr - sem_hwa_tr,
        mean_hwa_tr + sem_hwa_tr,
        color="#2ca02c",
        alpha=0.2,
        label="Train SEM",
    )
    plt.plot(epochs_axis, mean_hwa_val, label="Val HWA Mean", color="#d62728")
    plt.fill_between(
        epochs_axis,
        mean_hwa_val - sem_hwa_val,
        mean_hwa_val + sem_hwa_val,
        color="#d62728",
        alpha=0.2,
        label="Val SEM",
    )
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.title("SPR – Train vs Val HWA (Mean ± SEM over runs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_agg_HWA_curve.png"), dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated HWA curve: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 3) Final CWA / SWA / HWA bar chart with SEM
# ------------------------------------------------------------------ #
try:
    if final_cwa:
        cats = ["CWA", "SWA", "HWA"]
        means = [np.nanmean(final_cwa), np.nanmean(final_swa), np.nanmean(final_hwa)]
        sems = [
            np.nanstd(final_cwa, ddof=1) / np.sqrt(len(final_cwa)),
            np.nanstd(final_swa, ddof=1) / np.sqrt(len(final_swa)),
            np.nanstd(final_hwa, ddof=1) / np.sqrt(len(final_hwa)),
        ]
        plt.figure()
        plt.bar(
            cats, means, yerr=sems, capsize=5, color=["#4c72b0", "#55a868", "#c44e52"]
        )
        plt.ylabel("Metric Value")
        plt.title("SPR – Final Validation Metrics (Mean ± SEM)")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_agg_final_val_metrics.png"), dpi=150)
        plt.close()
except Exception as e:
    print(f"Error creating aggregated metric bar chart: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# Print aggregated test accuracy
# ------------------------------------------------------------------ #
if test_accuracies:
    print(
        f"Test accuracy over {len(test_accuracies)} runs: "
        f"mean={np.mean(test_accuracies):.4f}, std={np.std(test_accuracies, ddof=1):.4f}"
    )

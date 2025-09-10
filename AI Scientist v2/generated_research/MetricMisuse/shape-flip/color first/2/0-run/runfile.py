import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------ #
#  Basic set-up                                                #
# ------------------------------------------------------------ #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------ #
#  Experiment paths (relative to AI_SCIENTIST_ROOT)            #
# ------------------------------------------------------------ #
experiment_data_path_list = [
    "experiments/2025-08-30_17-49-38_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_6ea6384d055646959a1a01cee8ad031e_proc_1448868/experiment_data.npy",
    "experiments/2025-08-30_17-49-38_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_258f6881d93243248eeb6bbfa16530f2_proc_1448865/experiment_data.npy",
    "experiments/2025-08-30_17-49-38_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_e30c6a0ea72847fbb1bb62e792b724ce_proc_1448867/experiment_data.npy",
]


# ------------------------------------------------------------ #
#  Helper metrics                                              #
# ------------------------------------------------------------ #
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def pcwa(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    corr = [w_i if yt == yp else 0 for w_i, yt, yp in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


# ------------------------------------------------------------ #
#  Load all runs                                               #
# ------------------------------------------------------------ #
all_experiment_data = []
for p in experiment_data_path_list:
    try:
        exp = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_experiment_data.append(exp)
    except Exception as e:
        print(f"Error loading {p}: {e}")

num_runs = len(all_experiment_data)
if num_runs == 0:
    print("No experiment data could be loaded – exiting.")
    exit()

# ------------------------------------------------------------ #
#  Collect per-run series                                      #
# ------------------------------------------------------------ #
loss_train_runs, loss_val_runs = [], []
pcwa_train_runs, pcwa_val_runs = [], []
final_metrics_runs = []

for exp in all_experiment_data:
    spr = exp.get("SPR", {})
    lt = [v for _, v in spr.get("losses", {}).get("train", [])]
    lv = [v for _, v in spr.get("losses", {}).get("val", [])]
    pt = [v for _, v in spr.get("metrics", {}).get("train", [])]
    pv = [v for _, v in spr.get("metrics", {}).get("val", [])]
    loss_train_runs.append(np.array(lt, dtype=float))
    loss_val_runs.append(np.array(lv, dtype=float))
    pcwa_train_runs.append(np.array(pt, dtype=float))
    pcwa_val_runs.append(np.array(pv, dtype=float))

    # ---------- final test metrics for this run --------------- #
    seqs = spr.get("sequences", spr.get("ground_truth", []))
    y_true = spr.get("ground_truth", [])
    y_pred = spr.get("predictions", [])
    if seqs and y_true and y_pred:
        acc = sum(int(y == p) for y, p in zip(y_true, y_pred)) / len(y_true)
        pc = pcwa(seqs, y_true, y_pred)
        cwa_num = sum(
            count_color_variety(s) if y == p else 0
            for s, y, p in zip(seqs, y_true, y_pred)
        )
        cwa_den = sum(count_color_variety(s) for s in seqs)
        swa_num = sum(
            count_shape_variety(s) if y == p else 0
            for s, y, p in zip(seqs, y_true, y_pred)
        )
        swa_den = sum(count_shape_variety(s) for s in seqs)
        cwa = cwa_num / cwa_den if cwa_den else 0.0
        swa = swa_num / swa_den if swa_den else 0.0
        final_metrics_runs.append(dict(ACC=acc, PCWA=pc, CWA=cwa, SWA=swa))


# ------------------------------------------------------------ #
#  Utility: stack ragged arrays to (runs, min_len)             #
# ------------------------------------------------------------ #
def stack_and_trim(list_of_1d_arrays):
    min_len = min(arr.shape[0] for arr in list_of_1d_arrays if arr.size)
    return np.stack([arr[:min_len] for arr in list_of_1d_arrays]), np.arange(
        1, min_len + 1
    )


# ------------------------------------------------------------ #
#  Aggregated Loss Curves                                      #
# ------------------------------------------------------------ #
try:
    lt_mat, epochs = stack_and_trim(loss_train_runs)
    lv_mat, _ = stack_and_trim(loss_val_runs)
    mean_lt, se_lt = np.nanmean(lt_mat, 0), np.nanstd(lt_mat, 0, ddof=1) / np.sqrt(
        num_runs
    )
    mean_lv, se_lv = np.nanmean(lv_mat, 0), np.nanstd(lv_mat, 0, ddof=1) / np.sqrt(
        num_runs
    )

    plt.figure(figsize=(8, 4))
    plt.errorbar(epochs, mean_lt, yerr=se_lt, fmt="--o", label="Train (mean ± SE)")
    plt.errorbar(epochs, mean_lv, yerr=se_lv, fmt="-s", label="Validation (mean ± SE)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR – Mean ± SE Loss Curves Across {} Runs".format(num_runs))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_loss_curves_aggregate.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

# ------------------------------------------------------------ #
#  Aggregated PCWA Curves                                      #
# ------------------------------------------------------------ #
try:
    pt_mat, epochs_p = stack_and_trim(pcwa_train_runs)
    pv_mat, _ = stack_and_trim(pcwa_val_runs)
    mean_pt, se_pt = np.nanmean(pt_mat, 0), np.nanstd(pt_mat, 0, ddof=1) / np.sqrt(
        num_runs
    )
    mean_pv, se_pv = np.nanmean(pv_mat, 0), np.nanstd(pv_mat, 0, ddof=1) / np.sqrt(
        num_runs
    )

    plt.figure(figsize=(8, 4))
    plt.errorbar(epochs_p, mean_pt, yerr=se_pt, fmt="--o", label="Train (mean ± SE)")
    plt.errorbar(
        epochs_p, mean_pv, yerr=se_pv, fmt="-s", label="Validation (mean ± SE)"
    )
    plt.xlabel("Epoch")
    plt.ylabel("PCWA")
    plt.title("SPR – Mean ± SE PCWA Curves Across {} Runs".format(num_runs))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_pcwa_curves_aggregate.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated PCWA plot: {e}")
    plt.close()

# ------------------------------------------------------------ #
#  Aggregated Final-Test Metrics                               #
# ------------------------------------------------------------ #
try:
    if final_metrics_runs:
        metric_names = ["ACC", "PCWA", "CWA", "SWA"]
        metric_values = np.array(
            [[run[m] for m in metric_names] for run in final_metrics_runs]
        )
        means = metric_values.mean(axis=0)
        ses = metric_values.std(axis=0, ddof=1) / np.sqrt(metric_values.shape[0])

        x = np.arange(len(metric_names))
        plt.figure(figsize=(6, 4))
        plt.bar(
            x, means, yerr=ses, capsize=5, color="skyblue", alpha=0.9, label="Mean ± SE"
        )
        plt.xticks(x, metric_names)
        plt.title(
            "SPR – Final Test Metrics (Mean ± SE over {} Runs)".format(
                metric_values.shape[0]
            )
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_final_metrics_bar_aggregate.png"))
        plt.close()
        print("Aggregated final-test metrics (mean ± SE):")
        for n, m, s in zip(metric_names, means, ses):
            print(f"  {n}: {m:.4f} ± {s:.4f}")
    else:
        print("No final-test metrics found across runs.")
except Exception as e:
    print(f"Error creating aggregated final metrics plot: {e}")
    plt.close()

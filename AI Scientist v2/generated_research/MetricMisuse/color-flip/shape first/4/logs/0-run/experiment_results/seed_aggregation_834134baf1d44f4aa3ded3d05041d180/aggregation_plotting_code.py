import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- experiment data paths ----------
experiment_data_path_list = [
    "experiments/2025-08-15_23-37-14_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_68bc241fb11542cf828c82d8368f2c26_proc_3013400/experiment_data.npy",
    "experiments/2025-08-15_23-37-14_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_cc00501183544e30ae8bfa40788f4373_proc_3013401/experiment_data.npy",
    "None/experiment_data.npy",
]

# ---------- load data ----------
all_experiment_data = []
for p in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp_data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_data)
    except Exception as e:
        print(f"Error loading {p}: {e}")

if not all_experiment_data:
    print("No experiment files could be loaded – nothing to plot.")
    quit()

# ---------- determine datasets present in every run ----------
datasets = set(all_experiment_data[0].keys())
for data in all_experiment_data[1:]:
    datasets &= set(data.keys())
if not datasets:
    print("No common datasets across runs.")
    quit()

for dset in datasets:
    # gather per-run curves
    losses_tr_runs, losses_val_runs, hwa_runs = [], [], []
    epochs = None
    for run in all_experiment_data:
        try:
            run_d = run[dset]
            ltr = np.asarray(run_d["losses"]["train"], dtype=float)
            lval = np.asarray(run_d["losses"]["val"], dtype=float)
            hwa = np.asarray(run_d["metrics"]["val"], dtype=float)
            # keep only curves with matching length
            min_len = min(len(ltr), len(lval), len(hwa))
            ltr, lval, hwa = ltr[:min_len], lval[:min_len], hwa[:min_len]
            losses_tr_runs.append(ltr)
            losses_val_runs.append(lval)
            hwa_runs.append(hwa)
            if epochs is None or len(epochs) > min_len:
                epochs = np.arange(1, min_len + 1)
        except Exception as e:
            print(f"Skipping run during aggregation ({dset}): {e}")

    if len(losses_tr_runs) < 1:
        print(f"No valid runs for dataset {dset}")
        continue

    # stack & compute statistics
    tr_stack = np.stack(losses_tr_runs)
    val_stack = np.stack(losses_val_runs)
    hwa_stack = np.stack(hwa_runs)

    def mean_se(arr):
        mean = arr.mean(axis=0)
        se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        return mean, se

    tr_mean, tr_se = mean_se(tr_stack)
    val_mean, val_se = mean_se(val_stack)
    hwa_mean, hwa_se = mean_se(hwa_stack)

    # ---------- plot 1: aggregated loss curves ----------
    try:
        plt.figure()
        plt.plot(epochs, tr_mean, label="Train Loss (mean)", color="tab:blue")
        plt.fill_between(
            epochs,
            tr_mean - tr_se,
            tr_mean + tr_se,
            alpha=0.3,
            color="tab:blue",
            label="Train SE",
        )
        plt.plot(epochs, val_mean, label="Val Loss (mean)", color="tab:orange")
        plt.fill_between(
            epochs,
            val_mean - val_se,
            val_mean + val_se,
            alpha=0.3,
            color="tab:orange",
            label="Val SE",
        )
        plt.title(f"{dset} Aggregated Loss Curves\nMean ± Standard Error across runs")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset}_aggregated_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dset}: {e}")
        plt.close()

    # ---------- plot 2: aggregated validation HWA ----------
    try:
        plt.figure()
        plt.plot(epochs, hwa_mean, marker="o", label="HWA (mean)", color="tab:green")
        plt.fill_between(
            epochs,
            hwa_mean - hwa_se,
            hwa_mean + hwa_se,
            alpha=0.3,
            color="tab:green",
            label="HWA SE",
        )
        plt.title(f"{dset} Validation HWA\nMean ± Standard Error across runs")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset}_aggregated_HWA_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated HWA plot for {dset}: {e}")
        plt.close()

    # ---------- optional qualitative class distribution for up to 5 runs ----------
    try:
        for idx, run in enumerate(all_experiment_data[:5]):  # at most 5
            preds_last = run[dset].get("predictions", [None])[-1]
            gts_last = run[dset].get("ground_truth", [None])[-1]
            if preds_last is None or gts_last is None:
                continue
            plt.figure()
            classes = sorted(list(set(gts_last)))
            gt_counts = [gts_last.count(c) for c in classes]
            pred_counts = [preds_last.count(c) for c in classes]
            x = np.arange(len(classes))
            width = 0.35
            plt.bar(x - width / 2, gt_counts, width, label="Ground Truth")
            plt.bar(x + width / 2, pred_counts, width, label="Predictions")
            plt.title(f"{dset} Class Distribution (Run {idx})\nLeft: GT, Right: Pred")
            plt.xlabel("Class")
            plt.ylabel("Frequency")
            plt.xticks(x, classes)
            plt.legend()
            fname = os.path.join(working_dir, f"{dset}_class_dist_run_{idx}.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating class distribution plots for {dset}: {e}")
        plt.close()

    # ---------- console metrics ----------
    final_hwa_vals = hwa_stack[:, -1]
    print(
        f"{dset}: Final HWA mean ± SE = {final_hwa_vals.mean():.4f} ± "
        f"{final_hwa_vals.std(ddof=1)/np.sqrt(len(final_hwa_vals)):.4f}"
    )

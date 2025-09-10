import matplotlib.pyplot as plt
import numpy as np
import os
from math import sqrt

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------------------------------
# 1) load all experiment_data.npy files that were provided
# -------------------------------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-30_20-55-31_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_3ca0cf6c0eba4acbb4b7ef26c7cda1e1_proc_1490516/experiment_data.npy",
    "experiments/2025-08-30_20-55-31_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_99845a6d759a4206a06a91acd65c071f_proc_1490515/experiment_data.npy",
    "experiments/2025-08-30_20-55-31_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_7cf99b6dfb8e494c8247ba4cfba35be2_proc_1490514/experiment_data.npy",
]

all_experiment_data = []
try:
    root = os.getenv("AI_SCIENTIST_ROOT", "")
    for p in experiment_data_path_list:
        full_path = os.path.join(root, p)
        data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# if nothing was loaded we exit early
if not all_experiment_data:
    print("No experiment data found — nothing to plot.")
else:
    # ---------------------------------------------------------------------
    # 2) aggregate by epoch budget
    # ---------------------------------------------------------------------
    def run_key_to_int(k):
        return int(k.split("_")[-1]) if "_" in k else int(k)

    # budget_int -> list[histories (dict)]
    aggregated = {}
    for exp in all_experiment_data:
        try:
            spr_runs = exp["num_epochs"]["SPR"]
        except Exception as e:
            print(f"Experiment missing expected keys: {e}")
            continue
        for k, hist in spr_runs.items():
            b = run_key_to_int(k)
            aggregated.setdefault(b, []).append(hist)

    # keep only first 5 budgets (sorted) to respect plotting guideline
    budgets_sorted = sorted(aggregated.keys())[:5]

    # ---------------------------------------------------------------------
    # 3) Mean ± SEM loss curves
    # ---------------------------------------------------------------------
    try:
        plt.figure()
        for b in budgets_sorted:
            hists = aggregated[b]
            n = len(hists)
            # align epochs (assume identical across replicates)
            epochs = hists[0]["epochs"]
            train_mat = np.vstack([h["losses"]["train"] for h in hists])
            val_mat = np.vstack([h["losses"]["val"] for h in hists])

            mean_train = train_mat.mean(axis=0)
            mean_val = val_mat.mean(axis=0)
            sem_train = (
                train_mat.std(axis=0, ddof=0) / sqrt(n)
                if n > 1
                else np.zeros_like(mean_train)
            )
            sem_val = (
                val_mat.std(axis=0, ddof=0) / sqrt(n)
                if n > 1
                else np.zeros_like(mean_val)
            )

            label_base = f"budget_{b} (n={n})"
            plt.plot(epochs, mean_train, label=f"train {label_base}")
            plt.fill_between(
                epochs, mean_train - sem_train, mean_train + sem_train, alpha=0.2
            )

            plt.plot(epochs, mean_val, "--", label=f"val {label_base}")
            plt.fill_between(epochs, mean_val - sem_val, mean_val + sem_val, alpha=0.2)

        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR dataset – Mean Training vs Validation Loss\n(Shaded: ±SEM)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_loss_curves_mean.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss curves: {e}")
        plt.close()

    # ---------------------------------------------------------------------
    # 4) Mean ± SEM accuracy curves
    # ---------------------------------------------------------------------
    try:
        plt.figure()
        for b in budgets_sorted:
            hists = aggregated[b]
            n = len(hists)
            epochs = hists[0]["epochs"]
            train_mat = np.vstack([h["metrics"]["train"] for h in hists])
            val_mat = np.vstack([h["metrics"]["val"] for h in hists])

            mean_train = train_mat.mean(axis=0)
            mean_val = val_mat.mean(axis=0)
            sem_train = (
                train_mat.std(axis=0, ddof=0) / sqrt(n)
                if n > 1
                else np.zeros_like(mean_train)
            )
            sem_val = (
                val_mat.std(axis=0, ddof=0) / sqrt(n)
                if n > 1
                else np.zeros_like(mean_val)
            )

            label_base = f"budget_{b} (n={n})"
            plt.plot(epochs, mean_train, label=f"train {label_base}")
            plt.fill_between(
                epochs, mean_train - sem_train, mean_train + sem_train, alpha=0.2
            )

            plt.plot(epochs, mean_val, "--", label=f"val {label_base}")
            plt.fill_between(epochs, mean_val - sem_val, mean_val + sem_val, alpha=0.2)

        plt.xlabel("Epoch")
        plt.ylabel("Complexity-Weighted Accuracy")
        plt.title("SPR dataset – Mean Training vs Validation CpxWA\n(Shaded: ±SEM)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_accuracy_curves_mean.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated accuracy curves: {e}")
        plt.close()

    # ---------------------------------------------------------------------
    # 5) Test performance bar chart with error bars
    # ---------------------------------------------------------------------
    try:
        plt.figure()
        xs = []
        means = []
        sems = []
        labels = []
        for b in budgets_sorted:
            hists = aggregated[b]
            xs.append(b)
            vals = np.array([h["test_CpxWA"] for h in hists])
            means.append(vals.mean())
            sems.append(vals.std(ddof=0) / sqrt(len(vals)) if len(vals) > 1 else 0.0)
            labels.append(f"n={len(vals)}")

        bars = plt.bar(xs, means, yerr=sems, capsize=5, color="skyblue", edgecolor="k")
        for bar, txt in zip(bars, labels):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                txt,
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.xlabel("Epoch Budget")
        plt.ylabel("Test Complexity-Weighted Accuracy")
        plt.title("SPR dataset – Test Performance vs Epoch Budget\n(Error bars: ±SEM)")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_test_performance_mean.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated test performance plot: {e}")
        plt.close()

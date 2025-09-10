import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load all experiment_data ----------
experiment_data_path_list = [
    "experiments/2025-08-31_14-12-13_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_2acb90c3729e498ba596e83ac8b2730f_proc_1740734/experiment_data.npy",
    "experiments/2025-08-31_14-12-13_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_f29379efbd6848838f5acf7f0585b290_proc_1740735/experiment_data.npy",
    "experiments/2025-08-31_14-12-13_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_db742545200a436daa782c037a110435_proc_1740733/experiment_data.npy",
]

all_exp = []
try:
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        all_exp.append(np.load(full_path, allow_pickle=True).item())
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_exp = []

if not all_exp:
    print("No experiment data loaded – aborting plotting.")
    quit()

# ---------- find common dataset keys ----------
dataset_keys = set(all_exp[0].keys())
for exp in all_exp[1:]:
    dataset_keys &= set(exp.keys())

# For this task we only expect 'SPR_BENCH' but loop in case more appear
for dset in dataset_keys:
    # gather metrics dicts
    run_metrics = [exp[dset].get("metrics", {}) for exp in all_exp]

    # Helper to stack metric across runs
    def stack_metric(name):
        series_list = [rm.get(name, []) for rm in run_metrics]
        # remove empty lists
        series_list = [np.asarray(s) for s in series_list if len(s)]
        if not series_list:
            return None, None
        min_len = min(len(s) for s in series_list)
        trimmed = np.stack(
            [s[:min_len] for s in series_list], axis=0
        )  # shape (runs, epochs)
        mean = trimmed.mean(axis=0)
        sem = trimmed.std(axis=0, ddof=1) / np.sqrt(trimmed.shape[0])
        return mean, sem

    # ---------- aggregated plots ----------
    # 1) Loss curves ----------------------------------------------------------
    try:
        train_mean, train_sem = stack_metric("train_loss")
        val_mean, val_sem = stack_metric("val_loss")
        if train_mean is not None and val_mean is not None:
            epochs = np.arange(1, len(train_mean) + 1)
            plt.figure()
            plt.plot(epochs, train_mean, label="Train Loss (mean)")
            plt.fill_between(
                epochs,
                train_mean - train_sem,
                train_mean + train_sem,
                alpha=0.3,
                label="Train SEM",
            )
            plt.plot(epochs, val_mean, label="Val Loss (mean)")
            plt.fill_between(
                epochs,
                val_mean - val_sem,
                val_mean + val_sem,
                alpha=0.3,
                label="Val SEM",
            )
            plt.title(
                f"{dset} Aggregated Loss Curves\nMean ± SEM across {len(all_exp)} runs"
            )
            plt.xlabel("Epoch")
            plt.ylabel("BCE Loss")
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{dset}_aggregated_loss_curves.png")
            plt.savefig(fname)
            plt.close()
        else:
            print(f"Skipped loss plot for {dset} – metric missing.")
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dset}: {e}")
        plt.close()

    # 2) Weighted accuracy curves ---------------------------------------------
    try:
        metrics_to_plot = ["val_CWA", "val_SWA", "val_CWA2"]
        colors = ["tab:blue", "tab:orange", "tab:green"]
        any_plotted = False
        plt.figure()
        for m, c in zip(metrics_to_plot, colors):
            mean, sem = stack_metric(m)
            if mean is None:
                continue
            epochs = np.arange(1, len(mean) + 1)
            plt.plot(epochs, mean, label=f"{m.replace('val_', '')} (mean)", color=c)
            plt.fill_between(
                epochs,
                mean - sem,
                mean + sem,
                alpha=0.3,
                color=c,
                label=f"{m.replace('val_', '')} SEM",
            )
            any_plotted = True
        if any_plotted:
            plt.title(
                f"{dset} Aggregated Validation Weighted Accuracies\nMean ± SEM across {len(all_exp)} runs"
            )
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(
                working_dir, f"{dset}_aggregated_weighted_acc_curves.png"
            )
            plt.savefig(fname)
        else:
            print(f"No accuracy curves available for {dset}.")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated accuracy plot for {dset}: {e}")
        plt.close()

    # 3) Final epoch bar chart -------------------------------------------------
    try:
        labels, means, sems = [], [], []
        for m in ["val_CWA", "val_SWA", "val_CWA2"]:
            mean_series, sem_series = stack_metric(m)
            if mean_series is None:
                continue
            labels.append(m.replace("val_", ""))
            means.append(mean_series[-1])
            sems.append(sem_series[-1])
        if labels:
            plt.figure()
            x = np.arange(len(labels))
            plt.bar(x, means, yerr=sems, capsize=5)
            plt.xticks(x, labels)
            plt.ylabel("Accuracy")
            plt.title(
                f"{dset} Final Epoch Accuracies\nMean ± SEM across {len(all_exp)} runs"
            )
            plt.tight_layout()
            fname = os.path.join(
                working_dir, f"{dset}_aggregated_final_weighted_acc.png"
            )
            plt.savefig(fname)
            plt.close()
            # print summary
            summary = ", ".join(
                [f"{l}: {m:.4f}±{s:.4f}" for l, m, s in zip(labels, means, sems)]
            )
            print(f"{dset} - Final aggregated metrics -> {summary}")
        else:
            print(f"No final accuracy metrics present for {dset}.")
    except Exception as e:
        print(f"Error creating aggregated final bar plot for {dset}: {e}")
        plt.close()

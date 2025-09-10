import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# 1) Load every run -------------------------------------------------
# ------------------------------------------------------------------
try:
    experiment_data_path_list = [
        "experiments/2025-08-31_03-13-33_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_ea1c0cf0a47b42069b65c6b1cb6d118e_proc_1653757/experiment_data.npy",
        "experiments/2025-08-31_03-13-33_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_313b24ee63204f6f8b31747030e1caf2_proc_1653756/experiment_data.npy",
        "experiments/2025-08-31_03-13-33_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_58442ac0289542e88ef963afbbcee3d4_proc_1653759/experiment_data.npy",
    ]
    all_experiment_data = []
    for p in experiment_data_path_list:
        p_abs = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp_dict = np.load(p_abs, allow_pickle=True).item()
        all_experiment_data.append(exp_dict)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

if not all_experiment_data:
    print("No experiment data loaded - nothing to plot.")
else:
    # ------------------------------------------------------------------
    # 2) Aggregate per-dataset -----------------------------------------
    # ------------------------------------------------------------------
    # Collect the union of dataset names across runs
    dataset_names = set()
    for run in all_experiment_data:
        dataset_names.update(run.keys())

    # Storage for cross-dataset comparison later
    mean_test_acc = {}

    for ds in sorted(dataset_names):
        # Gather per-run arrays / dicts
        runs_present = [run for run in all_experiment_data if ds in run]
        if not runs_present:
            continue  # just in case

        # ------------------------------ losses ------------------------
        losses_train_list, losses_val_list = [], []
        for run in runs_present:
            ldict = run[ds].get("losses", {})
            if ldict.get("train"):
                losses_train_list.append(np.asarray(ldict["train"], dtype=float))
            if ldict.get("val"):
                losses_val_list.append(np.asarray(ldict["val"], dtype=float))

        # helper to stack with aligned length
        def stack_and_stats(arr_list):
            if not arr_list:
                return None, None, None
            min_len = min(len(a) for a in arr_list)
            trimmed = np.stack([a[:min_len] for a in arr_list])  # shape (runs, T)
            mean = trimmed.mean(axis=0)
            stderr = trimmed.std(axis=0, ddof=1) / np.sqrt(trimmed.shape[0])
            return mean, stderr, np.arange(1, min_len + 1)

        train_mean, train_se, epochs_train = stack_and_stats(losses_train_list)
        val_mean, val_se, epochs_val = stack_and_stats(losses_val_list)

        # ------------------------------ val metrics -------------------
        metric_names = ["acc", "cwa", "swa", "ccwa"]
        val_metrics_dict = {m: [] for m in metric_names}
        for run in runs_present:
            vm_list = run[ds].get("metrics", {}).get("val", [])
            if vm_list:
                # convert list[dict] -> np.array of metric values
                for m in metric_names:
                    vals = [d.get(m, np.nan) for d in vm_list]
                    val_metrics_dict[m].append(np.asarray(vals, dtype=float))
        val_stats = {}  # m -> (mean, se, epochs)
        for m, arr_list in val_metrics_dict.items():
            mean, se, epochs_m = stack_and_stats(arr_list)
            if mean is not None:
                val_stats[m] = (mean, se, epochs_m)

        # ------------------------------ test metrics ------------------
        test_metrics_runs = {m: [] for m in metric_names}
        for run in runs_present:
            tdict = run[ds].get("metrics", {}).get("test", {})
            for m in metric_names:
                if m in tdict and tdict[m] is not None:
                    test_metrics_runs[m].append(float(tdict[m]))
        test_mean = {
            m: (np.mean(v) if v else np.nan) for m, v in test_metrics_runs.items()
        }
        test_se = {
            m: (np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
            for m, v in test_metrics_runs.items()
        }

        # Save for cross-dataset comparison
        if not np.isnan(test_mean.get("acc", np.nan)):
            mean_test_acc[ds] = (test_mean["acc"], test_se["acc"])

        # ------------------------------------------------------------------
        # 3) Plotting -------------------------------------------------------
        # ------------------------------------------------------------------
        # 3.1 Loss curves with SEM
        try:
            if train_mean is not None or val_mean is not None:
                plt.figure(figsize=(6, 4))
                if train_mean is not None:
                    plt.plot(
                        epochs_train, train_mean, label="train mean", color="tab:blue"
                    )
                    plt.fill_between(
                        epochs_train,
                        train_mean - train_se,
                        train_mean + train_se,
                        alpha=0.3,
                        color="tab:blue",
                        label="train ± SEM",
                    )
                if val_mean is not None:
                    plt.plot(epochs_val, val_mean, label="val mean", color="tab:orange")
                    plt.fill_between(
                        epochs_val,
                        val_mean - val_se,
                        val_mean + val_se,
                        alpha=0.3,
                        color="tab:orange",
                        label="val ± SEM",
                    )
                plt.xlabel("Epoch")
                plt.ylabel("Cross-Entropy Loss")
                plt.title(
                    f"{ds} — Mean Train/Val Loss with ±SEM\n(aggregated over {len(runs_present)} runs)"
                )
                plt.legend()
                fname = os.path.join(working_dir, f"{ds}_loss_mean_sem.png")
                plt.savefig(fname)
                plt.close()
        except Exception as e:
            print(f"Error creating aggregated loss plot for {ds}: {e}")
            plt.close()

        # 3.2 Validation metric curves with SEM
        try:
            if val_stats:
                plt.figure(figsize=(6, 4))
                for m, (mean, se, ep) in val_stats.items():
                    plt.plot(ep, mean, label=f"{m.upper()} mean")
                    plt.fill_between(
                        ep, mean - se, mean + se, alpha=0.2, label=f"{m.upper()} ± SEM"
                    )
                plt.xlabel("Epoch")
                plt.ylabel("Score")
                plt.ylim(0, 1)
                plt.title(f"{ds} — Mean Validation Metrics with ±SEM")
                plt.legend(fontsize=8)
                fname = os.path.join(working_dir, f"{ds}_val_metrics_mean_sem.png")
                plt.savefig(fname)
                plt.close()
        except Exception as e:
            print(f"Error creating aggregated val-metric plot for {ds}: {e}")
            plt.close()

        # 3.3 Test metric bar chart with error bars
        try:
            if any(not np.isnan(v) for v in test_mean.values()):
                plt.figure(figsize=(6, 4))
                xs = np.arange(len(metric_names))
                bar_vals = [test_mean[m] for m in metric_names]
                bar_errs = [test_se[m] for m in metric_names]
                plt.bar(xs, bar_vals, yerr=bar_errs, color="skyblue", capsize=4)
                plt.xticks(xs, [m.upper() for m in metric_names])
                plt.ylim(0, 1)
                plt.title(f"{ds} — Mean Test Metrics ±SEM")
                for i, v in enumerate(bar_vals):
                    if not np.isnan(v):
                        plt.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
                fname = os.path.join(working_dir, f"{ds}_test_metrics_mean_sem.png")
                plt.savefig(fname)
                plt.close()
        except Exception as e:
            print(f"Error creating aggregated test-metric plot for {ds}: {e}")
            plt.close()

        # 3.4 Print numeric summary
        print(f"\n{ds} — TEST METRICS (mean ± std):")
        for m in metric_names:
            vals = test_metrics_runs[m]
            if vals:
                print(
                    f"  {m.upper():5s}: {np.mean(vals):.3f} ± {np.std(vals, ddof=1):.3f}  (n={len(vals)})"
                )

    # ------------------------------------------------------------------
    # 4) Cross-dataset comparison of test ACC --------------------------
    # ------------------------------------------------------------------
    try:
        if len(mean_test_acc) > 1:
            plt.figure(figsize=(6, 4))
            names = list(mean_test_acc.keys())
            means = [mean_test_acc[n][0] for n in names]
            errs = [mean_test_acc[n][1] for n in names]
            xs = np.arange(len(names))
            plt.bar(xs, means, yerr=errs, color="lightgreen", capsize=4)
            plt.xticks(xs, names, rotation=15, ha="right")
            plt.ylim(0, 1)
            plt.title("Mean Test Accuracy Across Datasets ±SEM")
            for i, v in enumerate(means):
                if not np.isnan(v):
                    plt.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
            fname = os.path.join(working_dir, "cross_dataset_test_acc_mean_sem.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating cross-dataset ACC plot: {e}")
        plt.close()

import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------------
# mandatory working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
# list of experiment_data paths provided by the user
experiment_data_path_list = [
    "experiments/2025-08-30_17-49-45_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_1b932e299f294e66a8970db537ae8d1f_proc_1458730/experiment_data.npy",
    "experiments/2025-08-30_17-49-45_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_1fc6a63a64de4b828e8ee6d4db632852_proc_1458732/experiment_data.npy",
    "experiments/2025-08-30_17-49-45_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_f468d1abeb844e0ca7e61b9092980e85_proc_1458733/experiment_data.npy",
]

# ---------------------------------------------------------------------
# load all runs
all_experiment_data = []
for p in experiment_data_path_list:
    try:
        run_dict = np.load(p, allow_pickle=True).item()
        all_experiment_data.append(run_dict)
    except Exception as e:
        print(f"Error loading {p}: {e}")

# ---------------------------------------------------------------------
# regroup by dataset name  ->  aggregated_data[ds_name] = [run1_dict, run2_dict, ...]
aggregated_data = {}
for run in all_experiment_data:
    for ds_name, ds_dict in run.items():
        aggregated_data.setdefault(ds_name, []).append(ds_dict)


# ---------------------------------------------------------------------
# helper to compute mean & stderr from list of 1-D arrays (equal length assumed)
def mean_se(arrs):
    data = np.stack(arrs, axis=0)
    mean = data.mean(axis=0)
    se = data.std(axis=0, ddof=1) / np.sqrt(data.shape[0])
    return mean, se


# ---------------------------------------------------------------------
# per-dataset aggregated visualisations
for ds_name, runs in aggregated_data.items():
    n_runs = len(runs)
    if n_runs == 0:
        continue

    # --------------- 1) aggregated loss curve -------------------------
    try:
        # truncate to shortest run so shapes match
        min_len = min(len(r["losses"]["train"]) for r in runs)
        train_losses = [np.array(r["losses"]["train"][:min_len]) for r in runs]
        val_losses = [np.array(r["losses"]["val"][:min_len]) for r in runs]

        mean_train, se_train = mean_se(train_losses)
        mean_val, se_val = mean_se(val_losses)
        epochs = np.arange(1, min_len + 1)

        plt.figure()
        plt.plot(epochs, mean_train, label="Train Loss (mean)", color="tab:blue")
        plt.fill_between(
            epochs,
            mean_train - se_train,
            mean_train + se_train,
            color="tab:blue",
            alpha=0.3,
            label="Train ±SE",
        )
        plt.plot(epochs, mean_val, label="Val Loss (mean)", color="tab:orange")
        plt.fill_between(
            epochs,
            mean_val - se_val,
            mean_val + se_val,
            color="tab:orange",
            alpha=0.3,
            label="Val ±SE",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{ds_name}: Train vs Val Loss (shaded = ±SE, N={n_runs})")
        plt.legend()
        plt.tight_layout()
        fname = f"{ds_name}_loss_curve_aggregated.png"
        plt.savefig(os.path.join(working_dir, fname))
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating aggregated loss curve for {ds_name}: {e}")
    finally:
        plt.close()

    # --------------- 2) aggregated BWA curve -------------------------
    try:
        min_len_bwa = min(len(r["metrics"]["train"]) for r in runs)
        train_bwa = [
            np.array([m["BWA"] for m in r["metrics"]["train"][:min_len_bwa]])
            for r in runs
        ]
        val_bwa = [
            np.array([m["BWA"] for m in r["metrics"]["val"][:min_len_bwa]])
            for r in runs
        ]

        mean_train_bwa, se_train_bwa = mean_se(train_bwa)
        mean_val_bwa, se_val_bwa = mean_se(val_bwa)
        epochs_bwa = np.arange(1, min_len_bwa + 1)

        plt.figure()
        plt.plot(
            epochs_bwa, mean_train_bwa, label="Train BWA (mean)", color="tab:green"
        )
        plt.fill_between(
            epochs_bwa,
            mean_train_bwa - se_train_bwa,
            mean_train_bwa + se_train_bwa,
            color="tab:green",
            alpha=0.3,
            label="Train ±SE",
        )
        plt.plot(epochs_bwa, mean_val_bwa, label="Val BWA (mean)", color="tab:red")
        plt.fill_between(
            epochs_bwa,
            mean_val_bwa - se_val_bwa,
            mean_val_bwa + se_val_bwa,
            color="tab:red",
            alpha=0.3,
            label="Val ±SE",
        )
        plt.xlabel("Epoch")
        plt.ylabel("BWA")
        plt.title(f"{ds_name}: Train vs Val BWA (shaded = ±SE, N={n_runs})")
        plt.legend()
        plt.tight_layout()
        fname = f"{ds_name}_bwa_curve_aggregated.png"
        plt.savefig(os.path.join(working_dir, fname))
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating aggregated BWA curve for {ds_name}: {e}")
    finally:
        plt.close()

    # --------------- 3) aggregated test metrics ----------------------
    try:
        metric_names = ["BWA", "CWA", "SWA", "StrWA"]
        vals_by_run = {m: [] for m in metric_names}
        for r in runs:
            tm = r.get("test_metrics", {})
            for m in metric_names:
                if m in tm:
                    vals_by_run[m].append(tm[m])
        means = []
        ses = []
        for m in metric_names:
            if vals_by_run[m]:
                v = np.array(vals_by_run[m])
                means.append(v.mean())
                ses.append(v.std(ddof=1) / np.sqrt(len(v)))
            else:
                means.append(np.nan)
                ses.append(0.0)

        plt.figure()
        x = np.arange(len(metric_names))
        plt.bar(x, means, yerr=ses, capsize=4, color="skyblue")
        plt.xticks(x, metric_names)
        for i, (mu, se) in enumerate(zip(means, ses)):
            if not np.isnan(mu):
                plt.text(
                    i, mu, f"{mu:.3f}±{se:.3f}", ha="center", va="bottom", fontsize=8
                )
        plt.ylabel("Score")
        plt.title(f"{ds_name}: Test Metrics Mean ±SE (N={n_runs})")
        plt.tight_layout()
        fname = f"{ds_name}_test_metrics_aggregated.png"
        plt.savefig(os.path.join(working_dir, fname))
        print(f"Saved {fname}")
        # also print to stdout
        print(f"{ds_name} aggregated test metrics (mean ±SE):")
        for m, mu, se in zip(metric_names, means, ses):
            if not np.isnan(mu):
                print(f"  {m}: {mu:.4f} ± {se:.4f}")
    except Exception as e:
        print(f"Error creating aggregated test-metric bar chart for {ds_name}: {e}")
    finally:
        plt.close()

# ---------------------------------------------------------------------
# cross-dataset comparison of mean test BWA ---------------------------
try:
    if len(aggregated_data) > 1:
        dsn, means, ses = [], [], []
        for ds_name, runs in aggregated_data.items():
            bwa_vals = [r.get("test_metrics", {}).get("BWA", np.nan) for r in runs]
            bwa_vals = [v for v in bwa_vals if not np.isnan(v)]
            if bwa_vals:
                dsn.append(ds_name)
                v = np.array(bwa_vals)
                means.append(v.mean())
                ses.append(v.std(ddof=1) / np.sqrt(len(v)))
        if dsn:
            plt.figure(figsize=(6, 4))
            x = np.arange(len(dsn))
            plt.bar(x, means, yerr=ses, capsize=4, color="salmon")
            plt.xticks(x, dsn, rotation=45, ha="right")
            for i, (mu, se) in enumerate(zip(means, ses)):
                plt.text(
                    i, mu, f"{mu:.3f}±{se:.3f}", ha="center", va="bottom", fontsize=8
                )
            plt.ylabel("Test BWA")
            plt.title("Dataset Comparison: Test BWA Mean ±SE")
            plt.tight_layout()
            fname = "cross_dataset_test_bwa_aggregated.png"
            plt.savefig(os.path.join(working_dir, fname))
            print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating cross-dataset aggregated comparison: {e}")
finally:
    plt.close()

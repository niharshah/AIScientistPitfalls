import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------------
# mandatory working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
# Load all experiment_data dictionaries that actually exist -------------
experiment_data_path_list = [
    "experiments/2025-08-30_17-49-45_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_321ca94d4f414dc380718c6085100112_proc_1445253/experiment_data.npy",
    "experiments/2025-08-30_17-49-45_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_21405c4999084797a998eac2c2647f5a_proc_1445251/experiment_data.npy",
    "experiments/2025-08-30_17-49-45_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_999c6ddfd2c74187bb7d585669a2248a_proc_1445252/experiment_data.npy",
]
all_experiment_data = []

# try to load from the provided absolute paths first
for experiment_data_path in experiment_data_path_list:
    try:
        root = os.getenv("AI_SCIENTIST_ROOT", "")
        full_path = os.path.join(root, experiment_data_path)
        if os.path.isfile(full_path):
            ed = np.load(full_path, allow_pickle=True).item()
            all_experiment_data.append(ed)
            print(f"Loaded {full_path}")
        else:
            print(f"File not found: {full_path}")
    except Exception as e:
        print(f"Error loading {experiment_data_path}: {e}")

# fallback: local copy dropped into working_dir
fallback_path = os.path.join(working_dir, "experiment_data.npy")
if os.path.isfile(fallback_path):
    try:
        ed = np.load(fallback_path, allow_pickle=True).item()
        all_experiment_data.append(ed)
        print(f"Loaded fallback {fallback_path}")
    except Exception as e:
        print(f"Error loading fallback experiment_data: {e}")

if len(all_experiment_data) == 0:
    print("No experiment data found, terminating plotting script.")
else:
    # -----------------------------------------------------------------
    # Gather per-epoch train/val curves across ALL runs ---------------
    all_train_curves, all_val_curves = [], []
    min_common_len = None
    for ed in all_experiment_data:
        runs = ed.get("num_epochs", {})
        for run_key, run_dict in runs.items():
            train = np.asarray(run_dict["metrics"].get("train", []), dtype=float)
            val = np.asarray(run_dict["metrics"].get("val", []), dtype=float)
            # Only keep runs that have both curves of the same length
            if train.size == 0 or val.size == 0 or train.size != val.size:
                continue
            all_train_curves.append(train)
            all_val_curves.append(val)
            min_common_len = (
                train.size
                if min_common_len is None
                else min(min_common_len, train.size)
            )

    # Trim all curves to the shortest common length so epochs align
    if min_common_len is not None and len(all_train_curves) > 0:
        all_train_curves = np.stack(
            [c[:min_common_len] for c in all_train_curves], axis=0
        )
        all_val_curves = np.stack([c[:min_common_len] for c in all_val_curves], axis=0)
        epochs = np.arange(1, min_common_len + 1)

        train_mean = all_train_curves.mean(axis=0)
        val_mean = all_val_curves.mean(axis=0)
        train_sem = all_train_curves.std(axis=0, ddof=1) / np.sqrt(
            all_train_curves.shape[0]
        )
        val_sem = all_val_curves.std(axis=0, ddof=1) / np.sqrt(all_val_curves.shape[0])
    else:
        train_mean = val_mean = train_sem = val_sem = epochs = None

    # -----------------------------------------------------------------
    # 1) Aggregated learning curve with SEM bands ---------------------
    try:
        if epochs is not None:
            plt.figure()
            plt.plot(epochs, train_mean, label="Train BWA – mean", color="tab:blue")
            plt.fill_between(
                epochs,
                train_mean - train_sem,
                train_mean + train_sem,
                color="tab:blue",
                alpha=0.2,
                label="Train SEM",
            )
            plt.plot(epochs, val_mean, label="Val BWA – mean", color="tab:orange")
            plt.fill_between(
                epochs,
                val_mean - val_sem,
                val_mean + val_sem,
                color="tab:orange",
                alpha=0.2,
                label="Val SEM",
            )
            plt.xlabel("Epoch")
            plt.ylabel("BWA")
            plt.title(
                "SPR-BENCH Mean ± SEM Train/Val BWA Learning Curve\n(Aggregated over all runs)"
            )
            plt.legend()
            plt.tight_layout()
            fname = "spr_bench_bwa_mean_sem_curve.png"
            path = os.path.join(working_dir, fname)
            plt.savefig(path)
            plt.close()
            print(f"Saved {path}")
        else:
            print("Skipping aggregated learning curve – insufficient aligned data.")
    except Exception as e:
        print(f"Error creating aggregated learning-curve plot: {e}")
        plt.close()

    # -----------------------------------------------------------------
    # 2) Bar plot of test-set BWA for each run with overall mean+SEM ---
    try:
        run_names, test_bwa_values = [], []
        for ed in all_experiment_data:
            runs = ed.get("num_epochs", {})
            for rk, rd in runs.items():
                if "test_metrics" in rd and "BWA" in rd["test_metrics"]:
                    run_names.append(rk)
                    test_bwa_values.append(rd["test_metrics"]["BWA"])

        if len(test_bwa_values) > 0:
            test_bwa_values = np.asarray(test_bwa_values, dtype=float)
            overall_mean = test_bwa_values.mean()
            overall_sem = test_bwa_values.std(ddof=1) / np.sqrt(test_bwa_values.size)

            x_pos = np.arange(len(run_names))
            plt.figure(figsize=(max(6, len(run_names) * 0.8), 4))
            plt.bar(x_pos, test_bwa_values, color="skyblue", label="Individual runs")
            plt.errorbar(
                len(run_names) + 0.5,
                overall_mean,
                yerr=overall_sem,
                fmt="o",
                color="red",
                label="Mean ± SEM",
            )
            plt.axhline(overall_mean, color="red", linestyle="--", alpha=0.6)
            plt.xticks(
                list(x_pos) + [len(run_names) + 0.5],
                run_names + ["Mean"],
                rotation=45,
                ha="right",
            )
            plt.ylabel("Test BWA")
            plt.title("SPR-BENCH Test BWA Across Runs (Mean ± SEM)")
            plt.legend()
            plt.tight_layout()
            fname = "spr_bench_test_bwa_runs_and_mean.png"
            path = os.path.join(working_dir, fname)
            plt.savefig(path)
            plt.close()
            print(f"Saved {path}")
        else:
            print("Skipping test BWA bar chart – no test metrics available.")
    except Exception as e:
        print(f"Error creating test-BWA aggregated bar chart: {e}")
        plt.close()

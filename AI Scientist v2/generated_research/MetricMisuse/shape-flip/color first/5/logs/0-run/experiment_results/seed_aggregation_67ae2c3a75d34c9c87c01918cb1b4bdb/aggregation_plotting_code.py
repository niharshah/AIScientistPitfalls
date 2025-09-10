import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
experiment_data_path_list = [
    "experiments/2025-08-30_20-55-31_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_019040f1075541dd83a579dedf17f867_proc_1494367/experiment_data.npy",
    "experiments/2025-08-30_20-55-31_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_c9fa5343dfba4457b0271b84ff4b26e6_proc_1494368/experiment_data.npy",
    "experiments/2025-08-30_20-55-31_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_857d7db31cdb4858acf44d6627f3b206_proc_1494369/experiment_data.npy",
]

all_runs_by_name = {}

try:
    root = os.getenv("AI_SCIENTIST_ROOT", "")
    for p in experiment_data_path_list:
        full_path = os.path.join(root, p)
        exp = np.load(full_path, allow_pickle=True).item()
        for run_name, run_dict in exp.items():
            all_runs_by_name.setdefault(run_name, []).append(run_dict)
except Exception as e:
    print(f"Error loading experiment data: {e}")


def save_close(fig_name):
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, fig_name))
    plt.close()


def _stack(list_of_arrs):
    # Pad to same length if necessary by truncating to min length
    lengths = [len(a) for a in list_of_arrs]
    min_len = min(lengths)
    return np.stack([a[:min_len] for a in list_of_arrs])


for run_name, run_list in all_runs_by_name.items():
    if len(run_list) == 0:
        continue
    # assume epochs are identical or at least same prefix
    epochs = np.asarray(run_list[0]["epochs"])
    n_runs = len(run_list)

    # ---------- 1) aggregated loss curve ----------
    try:
        train_losses = _stack([r["losses"]["train"] for r in run_list])
        val_losses = _stack([r["losses"]["val"] for r in run_list])

        mean_train = train_losses.mean(axis=0)
        sem_train = train_losses.std(axis=0, ddof=1) / np.sqrt(n_runs)
        mean_val = val_losses.mean(axis=0)
        sem_val = val_losses.std(axis=0, ddof=1) / np.sqrt(n_runs)

        plt.figure()
        plt.plot(
            epochs[: len(mean_train)], mean_train, label="train mean", color="tab:blue"
        )
        plt.fill_between(
            epochs[: len(mean_train)],
            mean_train - sem_train,
            mean_train + sem_train,
            alpha=0.3,
            color="tab:blue",
            label="train ± SEM",
        )
        plt.plot(
            epochs[: len(mean_val)],
            mean_val,
            "--",
            label="val mean",
            color="tab:orange",
        )
        plt.fill_between(
            epochs[: len(mean_val)],
            mean_val - sem_val,
            mean_val + sem_val,
            alpha=0.3,
            color="tab:orange",
            label="val ± SEM",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{run_name} – Aggregated Training vs Validation Loss")
        plt.legend()
        save_close(f"{run_name}_Agg_loss_curve.png")
    except Exception as e:
        print(f"Error creating aggregated loss curve for {run_name}: {e}")
        plt.close()

    # ---------- 2-4) aggregated metric curves ----------
    for metric in ["CWA", "SWA", "CmpWA"]:
        try:
            # verify key existence
            if metric not in run_list[0]["metrics"]["train"]:
                continue

            train_metric = _stack([r["metrics"]["train"][metric] for r in run_list])
            val_metric = _stack([r["metrics"]["val"][metric] for r in run_list])

            mean_train = train_metric.mean(axis=0)
            sem_train = train_metric.std(axis=0, ddof=1) / np.sqrt(n_runs)
            mean_val = val_metric.mean(axis=0)
            sem_val = val_metric.std(axis=0, ddof=1) / np.sqrt(n_runs)

            plt.figure()
            plt.plot(
                epochs[: len(mean_train)],
                mean_train,
                label="train mean",
                color="tab:blue",
            )
            plt.fill_between(
                epochs[: len(mean_train)],
                mean_train - sem_train,
                mean_train + sem_train,
                alpha=0.3,
                color="tab:blue",
                label="train ± SEM",
            )
            plt.plot(
                epochs[: len(mean_val)],
                mean_val,
                "--",
                label="val mean",
                color="tab:orange",
            )
            plt.fill_between(
                epochs[: len(mean_val)],
                mean_val - sem_val,
                mean_val + sem_val,
                alpha=0.3,
                color="tab:orange",
                label="val ± SEM",
            )
            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.title(f"{run_name} – Aggregated Training vs Validation {metric}")
            plt.legend()
            save_close(f"{run_name}_Agg_{metric}_curve.png")
        except Exception as e:
            print(f"Error creating aggregated {metric} curve for {run_name}: {e}")
            plt.close()

    # ---------- 5) aggregated test metrics summary ----------
    try:
        test_keys = ["loss", "CWA", "SWA", "CmpWA"]
        means = []
        sems = []
        for k in test_keys:
            vals = [r["test_metrics"][k] for r in run_list if k in r["test_metrics"]]
            if len(vals) == 0:
                means.append(np.nan)
                sems.append(0.0)
            else:
                vals = np.asarray(vals)
                means.append(vals.mean())
                sems.append(vals.std(ddof=1) / np.sqrt(len(vals)))

        plt.figure()
        x = np.arange(len(test_keys))
        bars = plt.bar(x, means, yerr=sems, capsize=5, color="skyblue")
        for xi, m in zip(x, means):
            if not np.isnan(m):
                plt.text(xi, m, f"{m:.3f}", ha="center", va="bottom")
        plt.xticks(x, [k.upper() for k in test_keys])
        plt.title(f"{run_name} – Aggregated Test Set Performance")
        save_close(f"{run_name}_Agg_test_summary.png")
    except Exception as e:
        print(f"Error creating aggregated test summary for {run_name}: {e}")
        plt.close()

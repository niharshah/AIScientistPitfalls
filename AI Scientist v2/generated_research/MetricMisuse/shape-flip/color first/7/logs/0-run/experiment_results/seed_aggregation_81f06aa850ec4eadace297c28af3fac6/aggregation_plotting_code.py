import matplotlib.pyplot as plt
import numpy as np
import os
from math import sqrt

# ---------------- setup ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- experiment paths ----------------
experiment_data_path_list = [
    "experiments/2025-08-30_20-55-38_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_913aa7ef26bf467a8d635495518e6f0c_proc_1494833/experiment_data.npy",
    "experiments/2025-08-30_20-55-38_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_59704523a67c41c29b3fcf11617a5078_proc_1494832/experiment_data.npy",
    "None/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_experiment_data.append(data)
    except Exception as e:
        print(f"Error loading experiment data from {p}: {e}")

n_runs = len(all_experiment_data)
if n_runs == 0:
    print("No experiment data could be loaded.")
    exit()


# ---------------- helper for CoWA ----------------
def colour_of(token: str) -> str:
    return token[1:] if len(token) > 1 else ""


def shape_of(token: str) -> str:
    return token[0]


def complexity_weight(seq: str) -> int:
    return len({colour_of(t) for t in seq.split() if t}) + len(
        {shape_of(t) for t in seq.split() if t}
    )


# ---------------- aggregation ----------------
# Collect union of dataset names across runs
all_ds_names = set()
for run in all_experiment_data:
    all_ds_names.update(run.keys())

for ds_name in all_ds_names:
    # Gather per-run series (may be missing in some runs)
    train_loss_runs, val_loss_runs = [], []
    train_acc_runs, val_acc_runs = [], []
    val_cowa_runs = []

    final_test_acc_list = []
    final_cowa_list = []

    for run in all_experiment_data:
        if ds_name not in run:
            continue
        ds = run[ds_name]

        # ----- series -----
        train_loss_runs.append(np.asarray(ds["losses"]["train"], dtype=float))
        val_loss_runs.append(np.asarray(ds["losses"]["val"], dtype=float))

        train_acc_runs.append(
            np.asarray([m["acc"] for m in ds["metrics"]["train"]], dtype=float)
        )
        val_acc_runs.append(
            np.asarray([m["acc"] for m in ds["metrics"]["val"]], dtype=float)
        )

        val_cowa_runs.append(
            np.asarray(
                [m.get("CompWA", m.get("cowa", np.nan)) for m in ds["metrics"]["val"]],
                dtype=float,
            )
        )

        # ----- final metrics -----
        try:
            preds = np.array(ds["predictions"])
            gts = np.array(ds["ground_truth"])
            seqs = np.array(ds["sequences"])
            test_acc = (preds == gts).mean()
            weights = np.array([complexity_weight(s) for s in seqs])
            cowa = (weights * (preds == gts)).sum() / weights.sum()
            final_test_acc_list.append(test_acc)
            final_cowa_list.append(cowa)
        except Exception as e:
            print(f"Error computing test metrics for run (dataset {ds_name}): {e}")

    # If nothing collected skip
    if len(train_loss_runs) == 0:
        continue

    # Helper to stack and compute mean & SE (truncate to shortest length)
    def stack_and_stats(series_list):
        min_len = min(len(s) for s in series_list)
        arr = np.vstack([s[:min_len] for s in series_list])
        mean = arr.mean(axis=0)
        se = (
            arr.std(axis=0, ddof=1) / sqrt(arr.shape[0])
            if arr.shape[0] > 1
            else np.zeros_like(mean)
        )
        return np.arange(1, min_len + 1), mean, se

    # ---------- aggregated loss curve ----------
    try:
        epochs, mean_train_loss, se_train_loss = stack_and_stats(train_loss_runs)
        _, mean_val_loss, se_val_loss = stack_and_stats(val_loss_runs)

        plt.figure()
        plt.plot(epochs, mean_train_loss, label="Train Mean")
        plt.fill_between(
            epochs,
            mean_train_loss - se_train_loss,
            mean_train_loss + se_train_loss,
            alpha=0.3,
            label="Train ± SE",
        )
        plt.plot(epochs, mean_val_loss, label="Validation Mean")
        plt.fill_between(
            epochs,
            mean_val_loss - se_val_loss,
            mean_val_loss + se_val_loss,
            alpha=0.3,
            label="Val ± SE",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(
            f"{ds_name} Aggregated Loss\nMean with ±1 SE over {len(train_loss_runs)} runs"
        )
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_aggregated_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {ds_name}: {e}")
        plt.close()

    # ---------- aggregated accuracy curve ----------
    try:
        epochs, mean_train_acc, se_train_acc = stack_and_stats(train_acc_runs)
        _, mean_val_acc, se_val_acc = stack_and_stats(val_acc_runs)

        plt.figure()
        plt.plot(epochs, mean_train_acc, label="Train Mean")
        plt.fill_between(
            epochs,
            mean_train_acc - se_train_acc,
            mean_train_acc + se_train_acc,
            alpha=0.3,
            label="Train ± SE",
        )
        plt.plot(epochs, mean_val_acc, label="Validation Mean")
        plt.fill_between(
            epochs,
            mean_val_acc - se_val_acc,
            mean_val_acc + se_val_acc,
            alpha=0.3,
            label="Val ± SE",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(
            f"{ds_name} Aggregated Accuracy\nMean with ±1 SE over {len(train_acc_runs)} runs"
        )
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"{ds_name}_aggregated_accuracy_curve.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated accuracy plot for {ds_name}: {e}")
        plt.close()

    # ---------- aggregated CoWA curve ----------
    try:
        # Some runs may have NaNs; we keep columns where at least one finite value exists
        finite_val_cowa_runs = [np.nan_to_num(s, nan=np.nan) for s in val_cowa_runs]
        epochs, mean_val_cowa, se_val_cowa = stack_and_stats(finite_val_cowa_runs)

        plt.figure()
        plt.plot(epochs, mean_val_cowa, label="Validation CoWA Mean")
        plt.fill_between(
            epochs,
            mean_val_cowa - se_val_cowa,
            mean_val_cowa + se_val_cowa,
            alpha=0.3,
            label="Val ± SE",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Complexity-Weighted Accuracy")
        plt.title(
            f"{ds_name} Aggregated CoWA\nMean with ±1 SE over {len(val_cowa_runs)} runs"
        )
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_aggregated_cowa_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated CoWA plot for {ds_name}: {e}")
        plt.close()

    # ---------- final test metrics bar ----------
    try:
        if final_test_acc_list:
            means = [np.mean(final_test_acc_list), np.mean(final_cowa_list)]
            ses = [
                (
                    np.std(final_test_acc_list, ddof=1) / sqrt(len(final_test_acc_list))
                    if len(final_test_acc_list) > 1
                    else 0
                ),
                (
                    np.std(final_cowa_list, ddof=1) / sqrt(len(final_cowa_list))
                    if len(final_cowa_list) > 1
                    else 0
                ),
            ]

            x = np.arange(2)
            labels = ["Test Accuracy", "Test CoWA"]

            plt.figure()
            plt.bar(x, means, yerr=ses, capsize=5, alpha=0.7)
            plt.xticks(x, labels)
            plt.ylabel("Score")
            plt.title(
                f"{ds_name} Final Test Metrics\nMean ± SE over {len(final_test_acc_list)} runs"
            )
            plt.savefig(
                os.path.join(working_dir, f"{ds_name}_aggregated_final_metrics.png")
            )
            plt.close()

            print(
                f"{ds_name}: Test Accuracy {means[0]:.3f} ± {ses[0]:.3f} | "
                f"Test CoWA {means[1]:.3f} ± {ses[1]:.3f}"
            )
    except Exception as e:
        print(f"Error creating final metrics bar plot for {ds_name}: {e}")
        plt.close()

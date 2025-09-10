import matplotlib.pyplot as plt
import numpy as np
import os
from math import sqrt

# mandatory working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# 1) LOAD ALL EXPERIMENTS ------------------------------------------
# ------------------------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-17_17-27-17_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_9d7a8df0e13f4d18bbd49b6d1de7b6a0_proc_3309947/experiment_data.npy",
    "experiments/2025-08-17_17-27-17_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_69853037235147cda2b40ae8feba2f31_proc_3309945/experiment_data.npy",
    "experiments/2025-08-17_17-27-17_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_727c8a684f6d46b0acca508376cab7a8_proc_3309948/experiment_data.npy",
]

all_runs = []
for p in experiment_data_path_list:
    try:
        root = os.getenv("AI_SCIENTIST_ROOT", "")
        full_path = os.path.join(root, p)
        data = np.load(full_path, allow_pickle=True).item()
        all_runs.append(data)
    except Exception as e:
        print(f"Error loading {p}: {e}")

if not all_runs:
    print("No experiment files could be loaded – aborting plotting.")
    quit()

# ------------------------------------------------------------------
# 2) GATHER & ALIGN METRICS ----------------------------------------
# ------------------------------------------------------------------
model_key = "NoAttnBiLSTM"
dataset_key = "SPR_BENCH"


def collect_series(key_chain):
    series_list = []
    for run in all_runs:
        try:
            obj = run
            for k in key_chain:
                obj = obj[k]
            series_list.append(np.asarray(obj, dtype=float))
        except Exception:
            continue
    if not series_list:
        return None, None, None  # nothing found
    min_len = min(len(s) for s in series_list)
    series_arr = np.vstack([s[:min_len] for s in series_list])
    epochs = np.arange(1, min_len + 1)
    mean = series_arr.mean(axis=0)
    sem = (
        series_arr.std(axis=0, ddof=1) / np.sqrt(series_arr.shape[0])
        if series_arr.shape[0] > 1
        else np.zeros_like(mean)
    )
    return epochs, mean, sem


# Losses
ep_loss, mean_tr_loss, sem_tr_loss = collect_series(
    [model_key, dataset_key, "losses", "train"]
)
_, mean_val_loss, sem_val_loss = collect_series(
    [model_key, dataset_key, "losses", "val"]
)

# F1
ep_f1, mean_tr_f1, sem_tr_f1 = collect_series(
    [model_key, dataset_key, "metrics", "train_f1"]
)
_, mean_val_f1, sem_val_f1 = collect_series(
    [model_key, dataset_key, "metrics", "val_f1"]
)

# Rule-Extraction Accuracy (single numbers per run)
rea_dev_vals, rea_test_vals = [], []
for run in all_runs:
    try:
        metrics = run[model_key][dataset_key]["metrics"]
        rea_dev_vals.append(float(metrics["REA_dev"]))
        rea_test_vals.append(float(metrics["REA_test"]))
    except Exception:
        continue
rea_dev_vals, rea_test_vals = np.asarray(rea_dev_vals), np.asarray(rea_test_vals)

# ------------------------------------------------------------------
# 3) PLOTTING -------------------------------------------------------
# ------------------------------------------------------------------

# 3.1 Loss curves ---------------------------------------------------
try:
    if ep_loss is not None:
        plt.figure()
        plt.plot(ep_loss, mean_tr_loss, label="Mean Train Loss", color="tab:blue")
        plt.fill_between(
            ep_loss,
            mean_tr_loss - sem_tr_loss,
            mean_tr_loss + sem_tr_loss,
            color="tab:blue",
            alpha=0.3,
            label="SEM Train",
        )
        plt.plot(ep_loss, mean_val_loss, label="Mean Val Loss", color="tab:orange")
        plt.fill_between(
            ep_loss,
            mean_val_loss - sem_val_loss,
            mean_val_loss + sem_val_loss,
            color="tab:orange",
            alpha=0.3,
            label="SEM Val",
        )
        plt.title("SPR_BENCH – Aggregated Loss Curves\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_aggregated_loss_curves.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

# 3.2 F1 curves -----------------------------------------------------
try:
    if ep_f1 is not None:
        plt.figure()
        plt.plot(ep_f1, mean_tr_f1, label="Mean Train F1", color="tab:green")
        plt.fill_between(
            ep_f1,
            mean_tr_f1 - sem_tr_f1,
            mean_tr_f1 + sem_tr_f1,
            color="tab:green",
            alpha=0.3,
            label="SEM Train",
        )
        plt.plot(ep_f1, mean_val_f1, label="Mean Val F1", color="tab:red")
        plt.fill_between(
            ep_f1,
            mean_val_f1 - sem_val_f1,
            mean_val_f1 + sem_val_f1,
            color="tab:red",
            alpha=0.3,
            label="SEM Val",
        )
        plt.title(
            "SPR_BENCH – Aggregated Macro-F1 Curves\nLeft: Train, Right: Validation"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_aggregated_f1_curves.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated F1 plot: {e}")
    plt.close()

# 3.3 Rule-Extraction Accuracy -------------------------------------
try:
    if rea_dev_vals.size and rea_test_vals.size:
        mean_dev, mean_test = rea_dev_vals.mean(), rea_test_vals.mean()
        sem_dev = (
            rea_dev_vals.std(ddof=1) / sqrt(len(rea_dev_vals))
            if len(rea_dev_vals) > 1
            else 0.0
        )
        sem_test = (
            rea_test_vals.std(ddof=1) / sqrt(len(rea_test_vals))
            if len(rea_test_vals) > 1
            else 0.0
        )

        plt.figure()
        x = np.arange(2)
        means = [mean_dev, mean_test]
        sems = [sem_dev, sem_test]
        bars = plt.bar(
            x, means, yerr=sems, capsize=5, color=["skyblue", "lightgreen"], alpha=0.8
        )
        plt.xticks(x, ["Dev", "Test"])
        plt.title(
            "SPR_BENCH – Aggregated Rule-Extraction Accuracy\nLeft: Dev, Right: Test"
        )
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        for bar, val in zip(bars, means):
            plt.text(
                bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center"
            )
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_aggregated_REA_accuracy.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated REA plot: {e}")
    plt.close()

print("Aggregated plot generation complete.")

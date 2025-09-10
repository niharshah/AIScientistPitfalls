import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------- #
# paths & data loading
# ---------------------------------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data_path_list = [
    "experiments/2025-08-14_23-40-39_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_f37bfe8e8dc44f87858b399263dbaf44_proc_2769061/experiment_data.npy",
    "experiments/2025-08-14_23-40-39_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_0aaa5bd34c39488a8ae9c25a1c544eb6_proc_2769064/experiment_data.npy",
    "None/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        full_p = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        if not os.path.isfile(full_p):
            raise FileNotFoundError(f"{full_p} does not exist")
        exp_data = np.load(full_p, allow_pickle=True).item()
        all_experiment_data.append(exp_data)
    except Exception as e:
        print(f"Error loading {p}: {e}")

if len(all_experiment_data) == 0:
    raise RuntimeError("No experiment data could be loaded.")

# ---------------------------------------------------- #
# aggregate SPR_BENCH
# ---------------------------------------------------- #
dataset_key = "SPR_BENCH"

runs_metrics = []
runs_preds_dev, runs_gts_dev = [], []
runs_preds_test, runs_gts_test = [], []

for exp in all_experiment_data:
    if dataset_key not in exp:
        continue
    d = exp[dataset_key]
    runs_metrics.append(d["metrics"])
    runs_preds_dev.append(np.array(d["predictions"]["dev"]))
    runs_gts_dev.append(np.array(d["ground_truth"]["dev"]))
    runs_preds_test.append(np.array(d["predictions"]["test"]))
    runs_gts_test.append(np.array(d["ground_truth"]["test"]))

num_runs = len(runs_metrics)
if num_runs == 0:
    raise RuntimeError(f"No runs contained key {dataset_key}")

# Helper to match epoch length across runs
min_epochs = min(len(m["train_loss"]) for m in runs_metrics)
metric_names = runs_metrics[0].keys()

# stack metrics: dict -> (num_runs, min_epochs) array
stacked = {
    k: np.vstack([m[k][:min_epochs] for m in runs_metrics]) for k in metric_names
}
epochs = np.arange(1, min_epochs + 1)


def mean_and_sem(arr):
    mean = arr.mean(axis=0)
    sem = (
        arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        if arr.shape[0] > 1
        else np.zeros_like(mean)
    )
    return mean, sem


# ---------------------------------------------------- #
# Figure 1: aggregated loss curves
# ---------------------------------------------------- #
try:
    train_mean, train_sem = mean_and_sem(stacked["train_loss"])
    val_mean, val_sem = mean_and_sem(stacked["val_loss"])

    plt.figure()
    plt.plot(epochs, train_mean, label="Train mean", color="blue")
    plt.fill_between(
        epochs,
        train_mean - train_sem,
        train_mean + train_sem,
        alpha=0.3,
        color="blue",
        label="Train SEM",
    )
    plt.plot(epochs, val_mean, label="Val mean", color="orange")
    plt.fill_between(
        epochs,
        val_mean - val_sem,
        val_mean + val_sem,
        alpha=0.3,
        color="orange",
        label="Val SEM",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH Aggregated Loss Curves\nMean ± SEM across runs")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_agg_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss curves: {e}")
    plt.close()

# ---------------------------------------------------- #
# Figure 2: aggregated validation metrics
# ---------------------------------------------------- #
try:
    swa_mean, swa_sem = mean_and_sem(stacked["val_swa"])
    cwa_mean, cwa_sem = mean_and_sem(stacked["val_cwa"])
    bps_mean, bps_sem = mean_and_sem(stacked["val_bps"])

    plt.figure()
    for mean, sem, name, color in [
        (swa_mean, swa_sem, "SWA", "green"),
        (cwa_mean, cwa_sem, "CWA", "purple"),
        (bps_mean, bps_sem, "BPS", "red"),
    ]:
        plt.plot(epochs, mean, label=f"{name} mean", color=color)
        plt.fill_between(
            epochs, mean - sem, mean + sem, alpha=0.3, color=color, label=f"{name} SEM"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("SPR_BENCH Aggregated Validation Metrics\nMean ± SEM across runs")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_agg_validation_metrics.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated validation metrics: {e}")
    plt.close()


# ---------------------------------------------------- #
# Helper for per-class accuracy
# ---------------------------------------------------- #
def per_class_acc(y_true, y_pred, num_classes):
    acc = np.zeros(num_classes)
    counts = np.zeros(num_classes)
    for t, p in zip(y_true, y_pred):
        counts[t] += 1
        if t == p:
            acc[t] += 1
    acc = np.divide(acc, counts, out=np.zeros_like(acc), where=counts > 0)
    return acc


# ---------------------------------------------------- #
# Figure 3: aggregated per-class accuracy
# ---------------------------------------------------- #
try:
    num_classes = int(
        max(max(g.max() for g in runs_gts_test), max(g.max() for g in runs_gts_dev)) + 1
    )
    acc_dev_runs = np.vstack(
        [
            per_class_acc(gt, pr, num_classes)
            for gt, pr in zip(runs_gts_dev, runs_preds_dev)
        ]
    )
    acc_test_runs = np.vstack(
        [
            per_class_acc(gt, pr, num_classes)
            for gt, pr in zip(runs_gts_test, runs_preds_test)
        ]
    )

    dev_mean, dev_sem = mean_and_sem(acc_dev_runs)
    test_mean, test_sem = mean_and_sem(acc_test_runs)

    x = np.arange(num_classes)
    width = 0.35
    plt.figure(figsize=(max(6, num_classes * 0.6), 4))
    plt.bar(x - width / 2, dev_mean, width, yerr=dev_sem, label="Dev mean±SEM")
    plt.bar(x + width / 2, test_mean, width, yerr=test_sem, label="Test mean±SEM")
    plt.xlabel("Class ID")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH Aggregated Per-Class Accuracy\nMean ± SEM across runs")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_agg_per_class_accuracy.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated per-class accuracy plot: {e}")
    plt.close()

# ---------------------------------------------------- #
# Print overall accuracy summary
# ---------------------------------------------------- #
overall_dev = np.array(
    [
        (pr == gt).mean() if gt.size else 0.0
        for pr, gt in zip(runs_preds_dev, runs_gts_dev)
    ]
)
overall_test = np.array(
    [
        (pr == gt).mean() if gt.size else 0.0
        for pr, gt in zip(runs_preds_test, runs_gts_test)
    ]
)

print(f"Runs considered               : {num_runs}")
print(
    f"Overall Dev Accuracy  mean±std: {overall_dev.mean():.4f} ± {overall_dev.std(ddof=1):.4f}"
)
print(
    f"Overall Test Accuracy mean±std: {overall_test.mean():.4f} ± {overall_test.std(ddof=1):.4f}"
)

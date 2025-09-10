import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------- setup ---------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------- load all experiment data -------------
experiment_data_path_list = [
    "experiments/2025-08-14_01-55-43_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_db3ee448efc848bd99d490e389f805c0_proc_2604134/experiment_data.npy",
    "experiments/2025-08-14_01-55-43_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_de1ee5e5bf8c4e2a9d9fb32c77fe3497_proc_2604133/experiment_data.npy",
    "experiments/2025-08-14_01-55-43_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_48051cc4013d4674bc15cfe76bf8a81e_proc_2604132/experiment_data.npy",
]

all_experiment_data = []
for path in experiment_data_path_list:
    try:
        data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), path), allow_pickle=True
        ).item()
        if "SPR_BENCH" in data:
            all_experiment_data.append(data["SPR_BENCH"])
    except Exception as e:
        print(f"Error loading experiment data: {e}")

num_runs = len(all_experiment_data)
if num_runs == 0:
    print("No SPR_BENCH data found in any run.")
    exit()


# -------- helper to stack / aggregate lists ----------
def aggregate(metric_key, subkey=None):
    """Return mean and sem arrays across runs for a given metric list."""
    series_list = []
    for d in all_experiment_data:
        if subkey:  # for metrics dicts
            series = [m.get(subkey, np.nan) for m in d["metrics"].get(metric_key, [])]
        else:  # for losses dicts
            series = d["losses"].get(metric_key, [])
        series_list.append(np.asarray(series, dtype=float))

    # keep only runs where we have at least 1 value
    series_list = [s for s in series_list if len(s) > 0]
    if not series_list:
        return None, None

    min_len = min(len(s) for s in series_list)
    trimmed = np.stack(
        [s[:min_len] for s in series_list], axis=0
    )  # shape (runs, epochs)
    mean = trimmed.mean(axis=0)
    sem = (
        trimmed.std(axis=0, ddof=1) / np.sqrt(trimmed.shape[0])
        if trimmed.shape[0] > 1
        else np.zeros_like(mean)
    )
    return mean, sem


# ---------------- aggregate epoch-wise ----------------
loss_mean, loss_sem = aggregate("train")  # training loss
val_loss_mean, val_loss_sem = aggregate("val")
acc_mean, acc_sem = aggregate("train", "acc")
val_acc_mean, val_acc_sem = aggregate("val", "acc")
swa_mean, swa_sem = aggregate("train", "swa")
val_swa_mean, val_swa_sem = aggregate("val", "swa")

# ------------------ aggregate test -------------------
test_accs, test_swas = [], []
for d in all_experiment_data:
    t = d["metrics"].get("test", {})
    if "acc" in t and "swa" in t:
        test_accs.append(t["acc"])
        test_swas.append(t["swa"])
test_accs = np.asarray(test_accs, dtype=float)
test_swas = np.asarray(test_swas, dtype=float)

# --------------------- PLOTS -------------------------
# 1) Aggregated Loss Curves
try:
    if loss_mean is not None and val_loss_mean is not None:
        plt.figure()
        epochs = np.arange(1, len(loss_mean) + 1)
        plt.plot(epochs, loss_mean, label="Train Mean")
        plt.fill_between(epochs, loss_mean - loss_sem, loss_mean + loss_sem, alpha=0.3)
        plt.plot(epochs, val_loss_mean, label="Val Mean")
        plt.fill_between(
            epochs,
            val_loss_mean - val_loss_sem,
            val_loss_mean + val_loss_sem,
            alpha=0.3,
        )
        plt.title("SPR_BENCH – Aggregated Loss Curves (Mean ± SEM)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_loss_curves_aggregated.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating aggregated loss curve: {e}")
    plt.close()

# 2) Aggregated Accuracy Curves
try:
    if acc_mean is not None and val_acc_mean is not None:
        plt.figure()
        epochs = np.arange(1, len(acc_mean) + 1)
        plt.plot(epochs, acc_mean, label="Train Mean")
        plt.fill_between(epochs, acc_mean - acc_sem, acc_mean + acc_sem, alpha=0.3)
        plt.plot(epochs, val_acc_mean, label="Val Mean")
        plt.fill_between(
            epochs, val_acc_mean - val_acc_sem, val_acc_mean + val_acc_sem, alpha=0.3
        )
        plt.title("SPR_BENCH – Aggregated Accuracy Curves (Mean ± SEM)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_accuracy_curves_aggregated.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating aggregated accuracy curve: {e}")
    plt.close()

# 3) Aggregated SWA Curves
try:
    if swa_mean is not None and val_swa_mean is not None:
        plt.figure()
        epochs = np.arange(1, len(swa_mean) + 1)
        plt.plot(epochs, swa_mean, label="Train Mean")
        plt.fill_between(epochs, swa_mean - swa_sem, swa_mean + swa_sem, alpha=0.3)
        plt.plot(epochs, val_swa_mean, label="Val Mean")
        plt.fill_between(
            epochs, val_swa_mean - val_swa_sem, val_swa_mean + val_swa_sem, alpha=0.3
        )
        plt.title("SPR_BENCH – Aggregated Shape-Weighted Accuracy (Mean ± SEM)")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_swa_curves_aggregated.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating aggregated SWA curve: {e}")
    plt.close()

# 4) Aggregated Test Metrics
try:
    if test_accs.size > 0 and test_swas.size > 0:
        plt.figure()
        metrics = ["Accuracy", "SWA"]
        means = [test_accs.mean(), test_swas.mean()]
        sems = [
            (
                test_accs.std(ddof=1) / np.sqrt(len(test_accs))
                if len(test_accs) > 1
                else 0
            ),
            (
                test_swas.std(ddof=1) / np.sqrt(len(test_swas))
                if len(test_swas) > 1
                else 0
            ),
        ]
        plt.bar(metrics, means, yerr=sems, color=["steelblue", "tan"], capsize=5)
        plt.title("SPR_BENCH – Aggregated Test Metrics (Mean ± SEM)")
        for i, v in enumerate(means):
            plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
        fname = os.path.join(working_dir, "spr_bench_test_metrics_aggregated.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating aggregated test metrics bar: {e}")
    plt.close()

# ---------------- print final metrics ----------------
if test_accs.size > 0 and test_swas.size > 0:
    print(f"Aggregated Test Accuracy: {test_accs.mean():.3f} ± {sems[0]:.3f}")
    print(f"Aggregated Test SWA     : {test_swas.mean():.3f} ± {sems[1]:.3f}")

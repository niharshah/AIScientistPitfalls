import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load every experiment ----------
experiment_data_path_list = [
    "experiments/2025-08-17_02-43-53_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_7f9710a8637448008cf34f1f3c7e7d07_proc_3198576/experiment_data.npy",
    "experiments/2025-08-17_02-43-53_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_80a558d3c01a47fa92aae47a5a391454_proc_3198574/experiment_data.npy",
    "experiments/2025-08-17_02-43-53_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_6602bb735cd54b7f9012ff0692a08325_proc_3198573/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        full_p = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        ed = np.load(full_p, allow_pickle=True).item()
        all_experiment_data.append(ed)
    except Exception as e:
        print(f"Error loading {p}: {e}")

# Only keep runs that actually have SPR_BENCH
runs = []
for ed in all_experiment_data:
    if "SPR_BENCH" in ed:
        runs.append(ed["SPR_BENCH"])


# Helper
def fetch(run_dict, cat, key):
    return np.asarray(run_dict.get(cat, {}).get(key, []))


# Aggregate a metric across runs (returns None if <2 valid runs)
def aggregate(metric_cat, metric_key):
    series = [fetch(r, metric_cat, metric_key) for r in runs]
    series = [s for s in series if s.size]  # keep non-empty
    if len(series) < 2:
        return None, None, None  # not enough data to aggregate
    min_len = min(len(s) for s in series)
    if min_len == 0:
        return None, None, None
    mat = np.vstack([s[:min_len] for s in series])
    mean = mat.mean(axis=0)
    se = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
    return mean, se, min_len


# ---------- Plot 1: Train / Val Accuracy with SE ----------
try:
    mean_tr, se_tr, n = aggregate("metrics", "train_acc")
    mean_val, se_val, _ = aggregate("metrics", "val_acc")
    if mean_tr is not None and mean_val is not None:
        epochs = np.arange(1, n + 1)
        plt.figure()
        plt.plot(epochs, mean_tr, label="Train Acc (mean)", color="tab:blue")
        plt.fill_between(
            epochs,
            mean_tr - se_tr,
            mean_tr + se_tr,
            color="tab:blue",
            alpha=0.25,
            label="Train ± SE",
        )
        plt.plot(epochs, mean_val, label="Val Acc (mean)", color="tab:orange")
        plt.fill_between(
            epochs,
            mean_val - se_val,
            mean_val + se_val,
            color="tab:orange",
            alpha=0.25,
            label="Val ± SE",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(
            "SPR_BENCH: Aggregated Train vs Validation Accuracy\n(mean ± standard error)"
        )
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, "SPR_BENCH_accuracy_curves_aggregate.png")
        )
        plt.close()
except Exception as e:
    print(f"Error creating aggregated accuracy plot: {e}")
    plt.close()

# ---------- Plot 2: Train / Val Loss with SE ----------
try:
    mean_tr_l, se_tr_l, n = aggregate("losses", "train")
    mean_val_l, se_val_l, _ = aggregate("losses", "val")
    if mean_tr_l is not None and mean_val_l is not None:
        epochs = np.arange(1, n + 1)
        plt.figure()
        plt.plot(epochs, mean_tr_l, label="Train Loss (mean)", color="tab:blue")
        plt.fill_between(
            epochs,
            mean_tr_l - se_tr_l,
            mean_tr_l + se_tr_l,
            color="tab:blue",
            alpha=0.25,
            label="Train ± SE",
        )
        plt.plot(epochs, mean_val_l, label="Val Loss (mean)", color="tab:orange")
        plt.fill_between(
            epochs,
            mean_val_l - se_val_l,
            mean_val_l + se_val_l,
            color="tab:orange",
            alpha=0.25,
            label="Val ± SE",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(
            "SPR_BENCH: Aggregated Train vs Validation Loss\n(mean ± standard error)"
        )
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves_aggregate.png"))
        plt.close()
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

# ---------- Plot 3: RBA vs Val Accuracy with SE ----------
try:
    mean_rba, se_rba, n = aggregate("metrics", "RBA")
    mean_val_acc, se_val_acc, _ = aggregate("metrics", "val_acc")
    if mean_rba is not None and mean_val_acc is not None:
        epochs = np.arange(1, n + 1)
        plt.figure()
        plt.plot(epochs, mean_val_acc, label="Val Acc (mean)", color="tab:green")
        plt.fill_between(
            epochs,
            mean_val_acc - se_val_acc,
            mean_val_acc + se_val_acc,
            color="tab:green",
            alpha=0.25,
            label="Val ± SE",
        )
        plt.plot(epochs, mean_rba, label="Rule-Based Acc (mean)", color="tab:red")
        plt.fill_between(
            epochs,
            mean_rba - se_rba,
            mean_rba + se_rba,
            color="tab:red",
            alpha=0.25,
            label="RBA ± SE",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(
            "SPR_BENCH: Aggregated Validation vs Rule-Based Accuracy\n(mean ± standard error)"
        )
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_rba_vs_val_aggregate.png"))
        plt.close()
except Exception as e:
    print(f"Error creating aggregated RBA plot: {e}")
    plt.close()

# ---------- Scalar evaluation: mean ± std of test accuracy ----------
accs = []
for r in runs:
    preds = np.asarray(r.get("predictions", []))
    gts = np.asarray(r.get("ground_truth", []))
    if preds.size and preds.shape == gts.shape:
        accs.append((preds == gts).mean())
if len(accs) > 0:
    print(
        f"Aggregated Test Accuracy over {len(accs)} runs: "
        f"{np.mean(accs):.3f} ± {np.std(accs, ddof=1):.3f}"
    )

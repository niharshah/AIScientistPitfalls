import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

# ---------------- paths & load ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- list of result files (relative to AI_SCIENTIST_ROOT) --------
experiment_data_path_list = [
    "experiments/2025-08-17_02-43-53_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_f15fbc114c7540f7a174e39f5472eec9_proc_3218404/experiment_data.npy",
    "experiments/2025-08-17_02-43-53_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_d5256260cf344c2f9e837e7a2f913293_proc_3218405/experiment_data.npy",
    "experiments/2025-08-17_02-43-53_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_5f3e12ea62624591a4a63d4ba703cf78_proc_3218403/experiment_data.npy",
]

all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full_p = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        blob = np.load(full_p, allow_pickle=True).item()
        all_experiment_data.append(blob)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# -------- aggregate runs per-dataset --------
agg = defaultdict(lambda: defaultdict(list))  # dataset → metric → list[runs]

for run_blob in all_experiment_data:
    for ds_name, ds_blob in run_blob.items():
        metrics = ds_blob.get("metrics", {})
        losses = ds_blob.get("losses", {})
        # accuracy curves
        for tag in ["train_acc", "val_acc"]:
            arr = np.asarray(metrics.get(tag, []))
            if arr.size:
                agg[ds_name][tag].append(arr)
        # loss curves
        for tag_raw, tag_store in [("train", "train_loss"), ("val", "val_loss")]:
            arr = np.asarray(losses.get(tag_raw, []))
            if arr.size:
                agg[ds_name][tag_store].append(arr)
        # rule fidelity
        rf = np.asarray(metrics.get("Rule_Fidelity", []))
        if rf.size:
            agg[ds_name]["Rule_Fidelity"].append(rf)
        # test accuracy (single number)
        preds = np.asarray(ds_blob.get("predictions", []))
        gts = np.asarray(ds_blob.get("ground_truth", []))
        if preds.size and gts.size and preds.shape == gts.shape:
            agg[ds_name]["test_acc"].append(float((preds == gts).mean()))


def plot_mean_sem(ax, series_list, label, color):
    """Plot mean with shaded SEM."""
    if not series_list:
        return
    min_len = min(len(s) for s in series_list)
    data = np.stack([s[:min_len] for s in series_list], axis=0)
    mean = data.mean(0)
    sem = (
        data.std(0, ddof=1) / np.sqrt(data.shape[0])
        if data.shape[0] > 1
        else np.zeros_like(mean)
    )
    epochs = np.arange(1, min_len + 1)
    ax.plot(epochs, mean, label=label, color=color)
    ax.fill_between(epochs, mean - sem, mean + sem, color=color, alpha=0.3)


# -------- iterate over aggregated datasets --------
for ds_name, metrics in agg.items():
    # 1) Accuracy curves (aggregate)
    try:
        if metrics.get("train_acc") and metrics.get("val_acc"):
            plt.figure()
            ax = plt.gca()
            plot_mean_sem(ax, metrics["train_acc"], "Train (mean ± SEM)", "tab:blue")
            plot_mean_sem(
                ax, metrics["val_acc"], "Validation (mean ± SEM)", "tab:orange"
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"{ds_name}: Aggregated Train vs Validation Accuracy")
            ax.legend()
            fname = os.path.join(working_dir, f"{ds_name}_agg_accuracy_curves.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated accuracy plot for {ds_name}: {e}")
        plt.close()

    # 2) Loss curves (aggregate)
    try:
        if metrics.get("train_loss") and metrics.get("val_loss"):
            plt.figure()
            ax = plt.gca()
            plot_mean_sem(ax, metrics["train_loss"], "Train (mean ± SEM)", "tab:green")
            plot_mean_sem(ax, metrics["val_loss"], "Validation (mean ± SEM)", "tab:red")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Cross-Entropy Loss")
            ax.set_title(f"{ds_name}: Aggregated Train vs Validation Loss")
            ax.legend()
            fname = os.path.join(working_dir, f"{ds_name}_agg_loss_curves.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {ds_name}: {e}")
        plt.close()

    # 3) Rule Fidelity
    try:
        if metrics.get("Rule_Fidelity"):
            plt.figure()
            ax = plt.gca()
            plot_mean_sem(
                ax, metrics["Rule_Fidelity"], "Rule Fidelity (mean ± SEM)", "tab:purple"
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Rule Fidelity")
            ax.set_title(f"{ds_name}: Aggregated Rule Fidelity over Epochs")
            ax.legend()
            fname = os.path.join(working_dir, f"{ds_name}_agg_rule_fidelity.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated rule fidelity plot for {ds_name}: {e}")
        plt.close()

    # ---- print aggregated test accuracy ----
    test_accs = metrics.get("test_acc", [])
    if test_accs:
        mean_acc = np.mean(test_accs)
        std_acc = np.std(test_accs, ddof=1) if len(test_accs) > 1 else 0.0
        print(
            f"{ds_name} Test Accuracy: {mean_acc:.3f} ± {std_acc:.3f} (n={len(test_accs)})"
        )

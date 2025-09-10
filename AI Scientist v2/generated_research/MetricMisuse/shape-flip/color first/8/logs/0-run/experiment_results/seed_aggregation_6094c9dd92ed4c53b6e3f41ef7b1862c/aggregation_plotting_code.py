import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- list all experiment_data paths ----------
experiment_data_path_list = [
    "None/experiment_data.npy",
    "experiments/2025-08-30_21-49-50_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_739457ffbfdd47968185b51791bbe34a_proc_1515243/experiment_data.npy",
    "None/experiment_data.npy",
]

# ---------- load data ----------
all_experiment_data = []
for path in experiment_data_path_list:
    try:
        # If AI_SCIENTIST_ROOT is set prepend it, else use path as is
        root = os.getenv("AI_SCIENTIST_ROOT", "")
        full_path = os.path.join(root, path) if root else path
        edata = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(edata)
    except Exception as e:
        print(f"Error loading experiment data from {path}: {e}")


# ---------- aggregate helper ----------
def stack_and_aggregate(list_of_arrays):
    """Return mean and stderr along axis 0 after matching min length."""
    min_len = min(len(a) for a in list_of_arrays)
    trimmed = np.stack([a[:min_len] for a in list_of_arrays], axis=0)  # (runs, epochs)
    mean = trimmed.mean(axis=0)
    stderr = trimmed.std(axis=0) / np.sqrt(trimmed.shape[0])
    return mean, stderr


# ---------- union of dataset names ----------
dataset_names = set()
for run_data in all_experiment_data:
    dataset_names.update(run_data.keys())

results_summary = {}

# ---------- iterate over datasets ----------
for ds_name in sorted(dataset_names):
    # gather metric lists per run
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for run_data in all_experiment_data:
        if ds_name not in run_data:
            continue
        ds_content = run_data[ds_name]
        losses = ds_content.get("losses", {})
        metrics = ds_content.get("metrics", {})
        if losses.get("train"):
            train_losses.append(np.array(losses["train"]))
        if losses.get("val"):
            val_losses.append(np.array(losses["val"]))
        if metrics.get("train"):
            train_accs.append(np.array(metrics["train"]))
        if metrics.get("val"):
            val_accs.append(np.array(metrics["val"]))

    # --------- aggregated loss curves ---------
    try:
        if train_losses or val_losses:
            plt.figure()
            if train_losses:
                m, se = stack_and_aggregate(train_losses)
                x = np.arange(len(m))
                plt.plot(x, m, label="Train mean")
                plt.fill_between(x, m - se, m + se, alpha=0.3, label="Train ±1 s.e.")
            if val_losses:
                m, se = stack_and_aggregate(val_losses)
                x = np.arange(len(m))
                plt.plot(x, m, label="Val mean")
                plt.fill_between(x, m - se, m + se, alpha=0.3, label="Val ±1 s.e.")
            plt.title(f"{ds_name} Aggregated Loss Curve\nMean ±1 Standard Error")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            fname = f"{ds_name}_aggregated_loss_curve.png".replace(" ", "_")
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss curve for {ds_name}: {e}")
        plt.close()

    # --------- aggregated accuracy curves ---------
    try:
        if train_accs or val_accs:
            plt.figure()
            if train_accs:
                m, se = stack_and_aggregate(train_accs)
                x = np.arange(len(m))
                plt.plot(x, m, label="Train mean")
                plt.fill_between(x, m - se, m + se, alpha=0.3, label="Train ±1 s.e.")
            if val_accs:
                m, se = stack_and_aggregate(val_accs)
                x = np.arange(len(m))
                plt.plot(x, m, label="Val mean")
                plt.fill_between(x, m - se, m + se, alpha=0.3, label="Val ±1 s.e.")
                results_summary[ds_name] = (m[-1], se[-1])
            plt.title(f"{ds_name} Aggregated Accuracy Curve\nMean ±1 Standard Error")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            fname = f"{ds_name}_aggregated_accuracy_curve.png".replace(" ", "_")
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated accuracy curve for {ds_name}: {e}")
        plt.close()

# ---------- print summary ----------
if results_summary:
    print("\nFinal Validation Accuracy (mean ± s.e.):")
    for ds_name, (mean_val, se_val) in results_summary.items():
        print(f"  {ds_name}: {mean_val:.4f} ± {se_val:.4f}")

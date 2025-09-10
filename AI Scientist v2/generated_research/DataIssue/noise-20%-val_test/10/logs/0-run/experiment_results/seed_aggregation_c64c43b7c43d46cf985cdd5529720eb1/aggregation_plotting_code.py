import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------------------------------------------------
# basic setup
# -------------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------------------------------
# load all experiment_data.npy files
# -------------------------------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-17_17-27-17_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_5cfc7e2f8e4b458db6e329c019e0d795_proc_3299637/experiment_data.npy",
    "experiments/2025-08-17_17-27-17_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_7eb3c72c40aa4e9ebafcd07e96aaa3ee_proc_3299634/experiment_data.npy",
    "experiments/2025-08-17_17-27-17_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_258efd606de24ec8b92d03b0b6538aa6_proc_3299635/experiment_data.npy",
]

all_experiment_data = []
try:
    for rel_path in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), rel_path)
        exp_dict = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_dict)
except Exception as e:
    print(f"Error loading experiment data: {e}")

if not all_experiment_data:
    print("No experiment data loaded – exiting early.")
    exit()


# -------------------------------------------------------------------------
# helper utilities
# -------------------------------------------------------------------------
def _stack_and_truncate(curve_list):
    """
    curve_list: list of 1-D float arrays of possibly different length
    returns: (stacked_array [n_runs, min_len], x_axis)
    """
    if not curve_list:
        return None, None
    min_len = min(len(c) for c in curve_list)
    stacked = np.vstack([c[:min_len] for c in curve_list])
    x_axis = np.arange(min_len)
    return stacked, x_axis


def _mean_and_stderr(stacked):
    mean = stacked.mean(axis=0)
    stderr = stacked.std(axis=0) / np.sqrt(stacked.shape[0])
    return mean, stderr


def safe_f1(preds, gts):
    try:
        from sklearn.metrics import f1_score

        return f1_score(gts, preds, average="macro")
    except Exception as e:
        print(f"Could not compute F1: {e}")
        return None


# -------------------------------------------------------------------------
# aggregate by dataset name
# -------------------------------------------------------------------------
dataset_names = set()
for run_dict in all_experiment_data:
    dataset_names.update(run_dict.keys())

for dset_name in dataset_names:
    # gather curves across runs ------------------------------------------------
    train_loss_curves, val_loss_curves = [], []
    train_f1_curves, val_f1_curves = [], []
    test_f1_scores = []

    for run_dict in all_experiment_data:
        d = run_dict.get(dset_name, {})
        # Loss
        losses = d.get("losses", {})
        if losses.get("train"):
            train_loss_curves.append(np.asarray(losses["train"], dtype=float))
        if losses.get("val"):
            val_loss_curves.append(np.asarray(losses["val"], dtype=float))
        # F1
        metrics = d.get("metrics", {})
        if metrics.get("train_f1"):
            train_f1_curves.append(np.asarray(metrics["train_f1"], dtype=float))
        if metrics.get("val_f1"):
            val_f1_curves.append(np.asarray(metrics["val_f1"], dtype=float))
        # Test F1
        preds, gts = np.asarray(d.get("predictions", [])), np.asarray(
            d.get("ground_truth", [])
        )
        if preds.size and gts.size:
            f1 = safe_f1(preds, gts)
            if f1 is not None:
                test_f1_scores.append(f1)

    # ---------------------- aggregated LOSS plot ----------------------------
    try:
        if train_loss_curves and val_loss_curves:
            tr_stack, x = _stack_and_truncate(train_loss_curves)
            val_stack, _ = _stack_and_truncate(val_loss_curves)
            if tr_stack is not None and val_stack is not None:
                tr_mean, tr_se = _mean_and_stderr(tr_stack)
                val_mean, val_se = _mean_and_stderr(val_stack)

                plt.figure()
                plt.plot(x, tr_mean, label="Train Loss (mean)")
                plt.fill_between(
                    x,
                    tr_mean - tr_se,
                    tr_mean + tr_se,
                    alpha=0.3,
                    label="Train Loss ±SE",
                )
                plt.plot(x, val_mean, label="Val Loss (mean)")
                plt.fill_between(
                    x,
                    val_mean - val_se,
                    val_mean + val_se,
                    alpha=0.3,
                    label="Val Loss ±SE",
                )
                plt.xlabel("Epoch")
                plt.ylabel("Cross-Entropy Loss")
                title_extra = ""
                if test_f1_scores:
                    title_extra = f" | Mean Test Macro-F1={np.mean(test_f1_scores):.3f}"
                plt.title(f"{dset_name}: Aggregated Loss Curves{title_extra}")
                plt.legend()
                fname = os.path.join(
                    working_dir, f"{dset_name}_aggregate_loss_curve.png"
                )
                plt.savefig(fname)
                plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dset_name}: {e}")
        plt.close()

    # ---------------------- aggregated F1 plot ------------------------------
    try:
        if train_f1_curves and val_f1_curves:
            tr_stack, x = _stack_and_truncate(train_f1_curves)
            val_stack, _ = _stack_and_truncate(val_f1_curves)
            if tr_stack is not None and val_stack is not None:
                tr_mean, tr_se = _mean_and_stderr(tr_stack)
                val_mean, val_se = _mean_and_stderr(val_stack)

                plt.figure()
                plt.plot(x, tr_mean, label="Train Macro-F1 (mean)")
                plt.fill_between(
                    x,
                    tr_mean - tr_se,
                    tr_mean + tr_se,
                    alpha=0.3,
                    label="Train Macro-F1 ±SE",
                )
                plt.plot(x, val_mean, label="Val Macro-F1 (mean)")
                plt.fill_between(
                    x,
                    val_mean - val_se,
                    val_mean + val_se,
                    alpha=0.3,
                    label="Val Macro-F1 ±SE",
                )
                plt.xlabel("Epoch")
                plt.ylabel("Macro-F1")
                best_val_idx = np.argmax(val_mean)
                best_val = val_mean[best_val_idx]
                title_extra = f"Best mean Val F1={best_val:.3f} at epoch {best_val_idx}"
                plt.title(f"{dset_name}: Aggregated Macro-F1 Curves | {title_extra}")
                plt.legend()
                fname = os.path.join(working_dir, f"{dset_name}_aggregate_f1_curve.png")
                plt.savefig(fname)
                plt.close()
    except Exception as e:
        print(f"Error creating aggregated F1 plot for {dset_name}: {e}")
        plt.close()

    # ---------------------- print aggregate statistics ----------------------
    if test_f1_scores:
        print(
            f"{dset_name}: mean test Macro-F1 over {len(test_f1_scores)} runs = {np.mean(test_f1_scores):.4f} "
            f"± {np.std(test_f1_scores)/np.sqrt(len(test_f1_scores)):.4f}"
        )

import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
# House-keeping
# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# Load all experiment_data.npy files
# ------------------------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-17_02-43-44_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_ff43174fef924aabab0d52c49c6759c2_proc_3209270/experiment_data.npy",
    "experiments/2025-08-17_02-43-44_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_ce8db0bcf92b423eba0b15daf97bf474_proc_3209268/experiment_data.npy",
    "experiments/2025-08-17_02-43-44_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_e4179b72bc3f47139ab09ebcae50c51c_proc_3209269/experiment_data.npy",
]
all_experiment_data = []
try:
    for p in experiment_data_path_list:
        exp = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_experiment_data.append(exp)
except Exception as e:
    print(f"Error loading experiment data: {e}")


# ------------------------------------------------------------------
# Helper: aggregate curves (truncate to min length)
# ------------------------------------------------------------------
def aggregate_curves(list_of_lists):
    list_of_lists = [np.asarray(x) for x in list_of_lists if len(x)]
    if not list_of_lists:
        return None, None, None
    min_len = min(len(x) for x in list_of_lists)
    stack = np.stack([x[:min_len] for x in list_of_lists], axis=0)
    mean = stack.mean(axis=0)
    se = stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0])
    return mean, se, np.arange(1, min_len + 1)


# ------------------------------------------------------------------
# Iterate over all dataset keys present across runs
# ------------------------------------------------------------------
ds_keys = set()
for exp in all_experiment_data:
    ds_keys.update(exp.keys())

for ds_key in ds_keys:
    # collect per-run arrays
    runs_train_acc, runs_val_acc = [], []
    runs_train_loss, runs_val_loss = [], []
    runs_val_rfs = []
    test_accs, test_rfs = [], []

    for exp in all_experiment_data:
        d = exp.get(ds_key, {})
        metrics = d.get("metrics", {})
        losses = d.get("losses", {})
        if metrics.get("train_acc"):
            runs_train_acc.append(metrics["train_acc"])
        if metrics.get("val_acc"):
            runs_val_acc.append(metrics["val_acc"])
        if losses.get("train"):
            runs_train_loss.append(losses["train"])
        if metrics.get("val_loss"):
            runs_val_loss.append(metrics["val_loss"])
        if metrics.get("val_rfs"):
            runs_val_rfs.append(metrics["val_rfs"])
        if d.get("test_acc") is not None:
            test_accs.append(d["test_acc"])
        if d.get("test_rfs") is not None:
            test_rfs.append(d["test_rfs"])

    # ------------------------ Accuracy curves ------------------------
    try:
        mean_tr, se_tr, epochs = aggregate_curves(runs_train_acc)
        mean_val, se_val, _ = aggregate_curves(runs_val_acc)
        if mean_tr is not None and mean_val is not None:
            plt.figure(figsize=(6, 4))
            plt.plot(epochs, mean_tr, label="Train (mean)", color="C0")
            plt.fill_between(
                epochs,
                mean_tr - se_tr,
                mean_tr + se_tr,
                color="C0",
                alpha=0.25,
                label="Train ± SE",
            )
            plt.plot(epochs, mean_val, label="Val (mean)", color="C1", linestyle="--")
            plt.fill_between(
                epochs,
                mean_val - se_val,
                mean_val + se_val,
                color="C1",
                alpha=0.25,
                label="Val ± SE",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(
                f"{ds_key}: Mean Training & Validation Accuracy\n(Shaded = ±SE over {len(runs_train_acc)} runs)"
            )
            plt.legend()
            plt.savefig(
                os.path.join(working_dir, f"{ds_key}_agg_acc_curves.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated accuracy for {ds_key}: {e}")
        plt.close()

    # ------------------------ Loss curves ------------------------
    try:
        mean_tr_l, se_tr_l, epochs_l = aggregate_curves(runs_train_loss)
        mean_val_l, se_val_l, _ = aggregate_curves(runs_val_loss)
        if mean_tr_l is not None and mean_val_l is not None:
            plt.figure(figsize=(6, 4))
            plt.plot(epochs_l, mean_tr_l, label="Train (mean)", color="C2")
            plt.fill_between(
                epochs_l,
                mean_tr_l - se_tr_l,
                mean_tr_l + se_tr_l,
                color="C2",
                alpha=0.25,
                label="Train ± SE",
            )
            plt.plot(
                epochs_l, mean_val_l, label="Val (mean)", color="C3", linestyle="--"
            )
            plt.fill_between(
                epochs_l,
                mean_val_l - se_val_l,
                mean_val_l + se_val_l,
                color="C3",
                alpha=0.25,
                label="Val ± SE",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(
                f"{ds_key}: Mean Training & Validation Loss\n(Shaded = ±SE over {len(runs_train_loss)} runs)"
            )
            plt.legend()
            plt.savefig(
                os.path.join(working_dir, f"{ds_key}_agg_loss_curves.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss for {ds_key}: {e}")
        plt.close()

    # ------------------------ Rule-Fidelity curves ------------------------
    try:
        mean_rf, se_rf, epochs_rf = aggregate_curves(runs_val_rfs)
        if mean_rf is not None:
            plt.figure(figsize=(6, 4))
            plt.plot(epochs_rf, mean_rf, color="purple", label="Val RFS (mean)")
            plt.fill_between(
                epochs_rf,
                mean_rf - se_rf,
                mean_rf + se_rf,
                color="purple",
                alpha=0.25,
                label="Val RFS ± SE",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Rule Fidelity")
            plt.title(
                f"{ds_key}: Mean Validation Rule Fidelity\n(Shaded = ±SE over {len(runs_val_rfs)} runs)"
            )
            plt.ylim(0, 1)
            plt.legend()
            plt.savefig(
                os.path.join(working_dir, f"{ds_key}_agg_val_rfs_curve.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated RFS for {ds_key}: {e}")
        plt.close()

    # ------------------------ Test metrics bar chart ------------------------
    try:
        if test_accs and test_rfs:
            bar_means = [np.mean(test_accs), np.mean(test_rfs)]
            bar_se = [
                np.std(test_accs, ddof=1) / np.sqrt(len(test_accs)),
                np.std(test_rfs, ddof=1) / np.sqrt(len(test_rfs)),
            ]
            x = np.arange(2)
            plt.figure(figsize=(4, 3))
            plt.bar(
                x,
                bar_means,
                yerr=bar_se,
                color=["green", "orange"],
                capsize=5,
                tick_label=["Test Acc", "Rule Fidelity"],
            )
            plt.ylim(0, 1)
            plt.title(
                f"{ds_key}: Test Metrics Across Runs\n(error bars = ±SE, n={len(test_accs)})"
            )
            plt.savefig(
                os.path.join(working_dir, f"{ds_key}_agg_test_metrics.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated test metrics for {ds_key}: {e}")
        plt.close()

    # ------------------------ Console summary ------------------------
    if test_accs:
        print(
            f"{ds_key} | TEST ACCURACY  mean±std: {np.mean(test_accs):.4f} ± {np.std(test_accs, ddof=1):.4f}"
        )
    if test_rfs:
        print(
            f"{ds_key} | TEST RULEFID. mean±std: {np.mean(test_rfs):.4f} ± {np.std(test_rfs, ddof=1):.4f}"
        )

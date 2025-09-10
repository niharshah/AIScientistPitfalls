import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# Paths supplied by the task
experiment_data_path_list = [
    "experiments/2025-08-17_02-43-50_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_45edcc5c212841c1a8acc8f279a01bfb_proc_3214201/experiment_data.npy",
    "experiments/2025-08-17_02-43-50_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_a1a76b2710d044738afb54322f1cb5e9_proc_3214204/experiment_data.npy",
    "experiments/2025-08-17_02-43-50_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_2012bb67ffb042fa845df0f47560c58e_proc_3214203/experiment_data.npy",
]

all_noise_records = []  # will store list of dicts keyed by noise level

# -------------------- Load all experiments --------------------
try:
    for experiment_data_path in experiment_data_path_list:
        exp_full_path = os.path.join(
            os.getenv("AI_SCIENTIST_ROOT", ""), experiment_data_path
        )
        edata = np.load(exp_full_path, allow_pickle=True).item()
        if "label_noise_robustness" in edata:
            all_noise_records.append(edata["label_noise_robustness"])
        else:
            print(
                f"Warning: 'label_noise_robustness' not found in {experiment_data_path}"
            )
except Exception as e:
    print(f"Error loading experiment data: {e}")

# -------------------- Aggregate by noise level --------------------
agg = {}  # noise_level -> dict of lists
for noise_dict in all_noise_records:
    for _, rec in noise_dict.items():
        n_lvl = rec["noise_level"]
        metrics = rec.get("metrics", {})
        losses = rec.get("losses", {})
        agg.setdefault(n_lvl, {"train": [], "val": [], "test": [], "val_loss": []})
        agg[n_lvl]["train"].append(metrics.get("train", [np.nan])[0])
        agg[n_lvl]["val"].append(metrics.get("val", [np.nan])[0])
        agg[n_lvl]["test"].append(metrics.get("test", [np.nan])[0])
        agg[n_lvl]["val_loss"].append(losses.get("val", [np.nan])[0])

# convert to sorted lists
noise_levels = sorted(agg.keys())
mean_train, se_train = [], []
mean_val, se_val = [], []
mean_test, se_test = [], []
mean_vloss, se_vloss = [], []

for n in noise_levels:
    for key, mean_list, se_list in [
        ("train", mean_train, se_train),
        ("val", mean_val, se_val),
        ("test", mean_test, se_test),
        ("val_loss", mean_vloss, se_vloss),
    ]:
        arr = np.array(agg[n][key], dtype=float)
        arr = arr[~np.isnan(arr)]  # drop nans
        if arr.size == 0:
            mean_val_ = np.nan
            se_val_ = np.nan
        else:
            mean_val_ = arr.mean()
            se_val_ = arr.std(ddof=1) / np.sqrt(arr.size)
        mean_list.append(mean_val_)
        se_list.append(se_val_)

# Print aggregated table
print("Noise  TrainAcc(±SE)  ValAcc(±SE)  TestAcc(±SE)  ValLoss(±SE)")
for i, n in enumerate(noise_levels):
    print(
        f"{n:4.2f}  {mean_train[i]:6.3f}±{se_train[i]:.3f} "
        f"{mean_val[i]:6.3f}±{se_val[i]:.3f} "
        f"{mean_test[i]:6.3f}±{se_test[i]:.3f} "
        f"{mean_vloss[i]:7.3f}±{se_vloss[i]:.3f}"
    )

# -------------------- Plot 1: Accuracy vs Noise with error bars --------------------
try:
    plt.figure()
    plt.errorbar(
        noise_levels, mean_train, yerr=se_train, fmt="-o", capsize=4, label="Train"
    )
    plt.errorbar(
        noise_levels, mean_val, yerr=se_val, fmt="-s", capsize=4, label="Validation"
    )
    plt.errorbar(
        noise_levels, mean_test, yerr=se_test, fmt="-^", capsize=4, label="Test"
    )
    plt.xlabel("Label noise fraction")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH – Accuracy vs Label Noise (mean ± SE)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_accuracy_vs_noise_mean_se.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot with error bars: {e}")
    plt.close()

# -------------------- Plot 2: Validation Loss vs Noise with error bars --------------------
try:
    plt.figure()
    plt.errorbar(
        noise_levels,
        mean_vloss,
        yerr=se_vloss,
        fmt="-o",
        color="orange",
        capsize=4,
        label="Val Loss",
    )
    plt.xlabel("Label noise fraction")
    plt.ylabel("Validation log-loss")
    plt.title("SPR_BENCH – Validation Loss vs Label Noise (mean ± SE)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_loss_vs_noise_mean_se.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating val loss plot with error bars: {e}")
    plt.close()

# -------------------- Optional: plot averaged epoch curves if present --------------------
try:
    # Attempt to collect per-epoch validation accuracy curves
    epoch_dict = {}  # noise_level -> list of np.array curves
    for noise_dict in all_noise_records:
        for _, rec in noise_dict.items():
            n_lvl = rec["noise_level"]
            history = rec.get("metrics_history", {})
            val_curve = history.get("val")  # assume shape (epochs,)
            if val_curve is not None:
                epoch_dict.setdefault(n_lvl, []).append(
                    np.asarray(val_curve, dtype=float)
                )

    if epoch_dict:
        for n_lvl, curves in epoch_dict.items():
            # align by shortest length
            min_len = min(len(c) for c in curves)
            curves = [c[:min_len] for c in curves]  # truncate
            arr = np.stack(curves, axis=0)
            mean_curve = arr.mean(axis=0)
            se_curve = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])

            # sample at most 5 time points for plotting
            epochs = np.arange(min_len)
            step = max(1, int(np.ceil(min_len / 5)))
            sampled_idx = epochs[::step]

            plt.figure()
            plt.errorbar(
                sampled_idx,
                mean_curve[sampled_idx],
                yerr=se_curve[sampled_idx],
                fmt="-o",
                capsize=3,
            )
            plt.xlabel("Epoch")
            plt.ylabel("Validation Accuracy")
            plt.title(f"SPR_BENCH – Val Acc Curve (Noise={n_lvl:.2f}) mean ± SE")
            fname = os.path.join(
                working_dir, f"SPR_BENCH_val_curve_noise_{n_lvl:.2f}.png"
            )
            plt.savefig(fname)
            plt.close()
except Exception as e:
    print(f"Error creating epoch curves: {e}")
    plt.close()

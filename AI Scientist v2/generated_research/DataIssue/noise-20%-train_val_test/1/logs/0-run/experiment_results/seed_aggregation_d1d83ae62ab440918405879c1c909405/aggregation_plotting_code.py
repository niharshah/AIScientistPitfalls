import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------------------------------------------------------- #
# basic setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------------------------------------------------- #
# paths to results (provided by the system)
experiment_data_path_list = [
    "experiments/2025-08-17_00-43-58_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_be549cedf21f4fd08a55c4b587c4973e_proc_3159063/experiment_data.npy",
    "experiments/2025-08-17_00-43-58_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_ef5da56a675846ef9a5ca7aae9a29cfd_proc_3159061/experiment_data.npy",
    "experiments/2025-08-17_00-43-58_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_9da198f02965469ca21acf01847e8b4a_proc_3159062/experiment_data.npy",
]

# detect root folder (default to cwd if env var not set)
root = os.getenv("AI_SCIENTIST_ROOT", os.getcwd())

all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full_path = os.path.join(root, p)
        if not os.path.isfile(full_path):
            print(f"Missing file: {full_path}")
            continue
        data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

if len(all_experiment_data) == 0:
    print("No experiment data loaded — exiting.")
    exit()

# --------------------------------------------------------------------- #
# collect SPR_BENCH data across runs
runs_bench = []
for data in all_experiment_data:
    bench = data.get("dropout_tuning", {}).get("SPR_BENCH", {})
    if bench:
        runs_bench.append(bench)

if len(runs_bench) == 0:
    print("No SPR_BENCH section found in any run — exiting.")
    exit()

# assume all runs share identical dropout settings
dropouts = sorted(list(runs_bench[0].keys()), key=float)

# sanity check: epoch length consistency
try:
    EPOCHS = len(runs_bench[0][dropouts[0]]["metrics"]["train_acc"])
    for run in runs_bench:
        for dp in dropouts:
            assert len(run[dp]["metrics"]["train_acc"]) == EPOCHS
except AssertionError:
    print("Mismatch in epoch counts across runs – unable to aggregate.")
    exit()

num_runs = len(runs_bench)

# containers
mean_curves = {
    dp: {"train": [], "val": [], "train_loss": [], "val_loss": []} for dp in dropouts
}
se_curves = {
    dp: {"train": [], "val": [], "train_loss": [], "val_loss": []} for dp in dropouts
}
test_acc_stats = {dp: {"mean": None, "se": None} for dp in dropouts}

# --------------------------------------------------------------------- #
# compute statistics
for dp in dropouts:
    # gather epoch-wise arrays of shape (num_runs, epochs)
    tr_arr = np.array([run[dp]["metrics"]["train_acc"] for run in runs_bench])
    val_arr = np.array([run[dp]["metrics"]["val_acc"] for run in runs_bench])
    tr_loss_arr = np.array([run[dp]["losses"]["train_loss"] for run in runs_bench])
    val_loss_arr = np.array([run[dp]["losses"]["val_loss"] for run in runs_bench])

    # mean & standard error
    mean_curves[dp]["train"] = tr_arr.mean(axis=0)
    mean_curves[dp]["val"] = val_arr.mean(axis=0)
    mean_curves[dp]["train_loss"] = tr_loss_arr.mean(axis=0)
    mean_curves[dp]["val_loss"] = val_loss_arr.mean(axis=0)

    se_curves[dp]["train"] = tr_arr.std(axis=0, ddof=1) / np.sqrt(num_runs)
    se_curves[dp]["val"] = val_arr.std(axis=0, ddof=1) / np.sqrt(num_runs)
    se_curves[dp]["train_loss"] = tr_loss_arr.std(axis=0, ddof=1) / np.sqrt(num_runs)
    se_curves[dp]["val_loss"] = val_loss_arr.std(axis=0, ddof=1) / np.sqrt(num_runs)

    # test accuracy statistics
    test_vals = np.array([run[dp]["test_acc"] for run in runs_bench])
    test_acc_stats[dp]["mean"] = test_vals.mean()
    test_acc_stats[dp]["se"] = test_vals.std(ddof=1) / np.sqrt(num_runs)

# print summary table
print("\n=== Aggregated Test Accuracies (mean ± SE) ===")
for dp in dropouts:
    m = test_acc_stats[dp]["mean"] * 100
    s = test_acc_stats[dp]["se"] * 100
    print(f"Dropout {dp}: {m:.2f}% ± {s:.2f}%")

# --------------------------------------------------------------------- #
# Plot 1: accuracy curves with SE envelope
try:
    plt.figure()
    epochs_axis = np.arange(1, EPOCHS + 1)
    for dp in dropouts:
        m_tr = mean_curves[dp]["train"]
        se_tr = se_curves[dp]["train"]
        m_val = mean_curves[dp]["val"]
        se_val = se_curves[dp]["val"]

        # plot mean lines
        plt.plot(epochs_axis, m_tr, label=f"{dp} train", linewidth=1.5)
        plt.plot(epochs_axis, m_val, label=f"{dp} val", linestyle="--", linewidth=1.5)

        # shaded SE
        plt.fill_between(epochs_axis, m_tr - se_tr, m_tr + se_tr, alpha=0.2)
        plt.fill_between(epochs_axis, m_val - se_val, m_val + se_val, alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        "SPR_BENCH: Mean Training/Validation Accuracy ± SE\n(Aggregated over runs)"
    )
    plt.legend(fontsize=8)
    fname = os.path.join(working_dir, "SPR_BENCH_mean_accuracy_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated accuracy plot: {e}")
    plt.close()

# --------------------------------------------------------------------- #
# Plot 2: loss curves with SE envelope
try:
    plt.figure()
    for dp in dropouts:
        m_tr = mean_curves[dp]["train_loss"]
        se_tr = se_curves[dp]["train_loss"]
        m_val = mean_curves[dp]["val_loss"]
        se_val = se_curves[dp]["val_loss"]

        plt.plot(epochs_axis, m_tr, label=f"{dp} train", linewidth=1.5)
        plt.plot(epochs_axis, m_val, label=f"{dp} val", linestyle="--", linewidth=1.5)

        plt.fill_between(epochs_axis, m_tr - se_tr, m_tr + se_tr, alpha=0.2)
        plt.fill_between(epochs_axis, m_val - se_val, m_val + se_val, alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH: Mean Training/Validation Loss ± SE\n(Aggregated over runs)")
    plt.legend(fontsize=8)
    fname = os.path.join(working_dir, "SPR_BENCH_mean_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

# --------------------------------------------------------------------- #
# Plot 3: bar chart of test accuracy with SE error bars
try:
    plt.figure()
    x = np.arange(len(dropouts))
    y_mean = [test_acc_stats[dp]["mean"] * 100 for dp in dropouts]
    y_se = [test_acc_stats[dp]["se"] * 100 for dp in dropouts]
    plt.bar(x, y_mean, yerr=y_se, capsize=5, color="skyblue", alpha=0.8)
    plt.xticks(x, dropouts)
    plt.ylabel("Test Accuracy (%)")
    plt.title(
        "SPR_BENCH: Mean Test Accuracy ± SE by Dropout Rate\n(Aggregated over runs)"
    )
    fname = os.path.join(working_dir, "SPR_BENCH_mean_test_accuracy_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated test accuracy bar chart: {e}")
    plt.close()

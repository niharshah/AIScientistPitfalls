import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------- basic setup --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# experiment data paths supplied by the system
experiment_data_path_list = [
    "experiments/2025-08-14_12-19-19_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_5fc5a7317b6141e58c0d6e57751fff73_proc_2637259/experiment_data.npy",
    "experiments/2025-08-14_12-19-19_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_89d00f0ef72e428682ae0a099bcd6025_proc_2637261/experiment_data.npy",
    "experiments/2025-08-14_12-19-19_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_7ebfe2c31bcf4f309478b232ceb5e23c_proc_2637262/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_experiment_data.append(data)
    except Exception as e:
        print(f"Error loading experiment data from {p}: {e}")


# -------------------- helpers --------------------
def stacked_metric(list_of_arrays):
    """Stack into (runs, epochs) after cutting to min length."""
    if len(list_of_arrays) == 0:
        return None, None
    min_len = min(len(a) for a in list_of_arrays)
    arr = np.stack([a[:min_len] for a in list_of_arrays], axis=0)  # (runs, epochs)
    mean = arr.mean(axis=0)
    sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
    return mean, sem


# Gather set of learning rates present in any run
learning_rates = sorted(
    {lr for exp in all_experiment_data for lr in exp.get("learning_rate", {})},
    key=float,
)

# -------------------- Plot 1: aggregated loss curves --------------------
try:
    plt.figure()
    for lr in learning_rates:
        train_runs, val_runs = [], []
        for exp in all_experiment_data:
            lr_dict = exp.get("learning_rate", {}).get(lr)
            if lr_dict is None:  # this exp did not run that lr
                continue
            train_runs.append(np.asarray(lr_dict["losses"]["train"]))
            val_runs.append(np.asarray(lr_dict["losses"]["val"]))

        m_tr, se_tr = stacked_metric(train_runs)
        m_val, se_val = stacked_metric(val_runs)
        if m_tr is None or m_val is None:
            continue

        epochs = np.arange(len(m_tr))
        plt.plot(epochs, m_tr, label=f"train lr={lr}")
        plt.fill_between(epochs, m_tr - se_tr, m_tr + se_tr, alpha=0.2)

        plt.plot(epochs, m_val, linestyle="--", label=f"val lr={lr}")
        plt.fill_between(epochs, m_val - se_val, m_val + se_val, alpha=0.2)

    plt.title("SPR_Bench Loss Curves (mean ± SEM)\nTrain vs Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy")
    plt.legend(fontsize=7)
    plt.savefig(os.path.join(working_dir, "SPR_Bench_loss_curves_agg.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss curves: {e}")
    plt.close()

# -------------------- Plot 2: aggregated accuracy curves --------------------
try:
    plt.figure()
    for lr in learning_rates:
        tr_acc_runs, val_acc_runs = [], []
        for exp in all_experiment_data:
            lr_dict = exp.get("learning_rate", {}).get(lr)
            if lr_dict is None:
                continue
            tr_acc_runs.append(np.asarray(lr_dict["metrics"]["train"]))
            val_acc_runs.append(np.asarray(lr_dict["metrics"]["val"]))

        m_tr, se_tr = stacked_metric(tr_acc_runs)
        m_val, se_val = stacked_metric(val_acc_runs)
        if m_tr is None or m_val is None:
            continue

        epochs = np.arange(len(m_tr))
        plt.plot(epochs, m_tr, label=f"train lr={lr}")
        plt.fill_between(epochs, m_tr - se_tr, m_tr + se_tr, alpha=0.2)

        plt.plot(epochs, m_val, linestyle="--", label=f"val lr={lr}")
        plt.fill_between(epochs, m_val - se_val, m_val + se_val, alpha=0.2)

    plt.title("SPR_Bench Accuracy Curves (mean ± SEM)\nTrain vs Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(fontsize=7)
    plt.savefig(os.path.join(working_dir, "SPR_Bench_accuracy_curves_agg.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated accuracy curves: {e}")
    plt.close()

# -------------------- Plot 3: aggregated RGS curves (validation) --------------------
try:
    plt.figure()
    for lr in learning_rates:
        val_rgs_runs = []
        for exp in all_experiment_data:
            lr_dict = exp.get("learning_rate", {}).get(lr)
            if lr_dict is None:
                continue
            if "val_rgs" in lr_dict["metrics"]:
                val_rgs_runs.append(np.asarray(lr_dict["metrics"]["val_rgs"]))
        m_val, se_val = stacked_metric(val_rgs_runs)
        if m_val is None:
            continue
        epochs = np.arange(len(m_val))
        plt.plot(epochs, m_val, label=f"val RGS lr={lr}")
        plt.fill_between(epochs, m_val - se_val, m_val + se_val, alpha=0.2)

    plt.title("SPR_Bench Validation RGS Curves (mean ± SEM)")
    plt.xlabel("Epoch")
    plt.ylabel("RGS")
    plt.legend(fontsize=7)
    plt.savefig(os.path.join(working_dir, "SPR_Bench_RGS_curves_agg.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated RGS curves: {e}")
    plt.close()

# -------------------- Plot 4: Final test accuracy bar (mean ± SEM) --------------------
try:
    means, sems, labels = [], [], []
    for lr in learning_rates:
        acc_runs = []
        for exp in all_experiment_data:
            lr_dict = exp.get("learning_rate", {}).get(lr)
            if lr_dict and "acc" in lr_dict["test_metrics"]:
                acc_runs.append(lr_dict["test_metrics"]["acc"])
        if len(acc_runs) == 0:
            continue
        means.append(np.mean(acc_runs))
        sems.append(np.std(acc_runs, ddof=1) / np.sqrt(len(acc_runs)))
        labels.append(lr)

    x = np.arange(len(means))
    plt.figure()
    plt.bar(x, means, yerr=sems, capsize=5, color="skyblue")
    plt.xticks(x, labels)
    plt.title("SPR_Bench Final Test Accuracy per Learning Rate (mean ± SEM)")
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(working_dir, "SPR_Bench_test_accuracy_agg.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated test accuracy bar plot: {e}")
    plt.close()

# -------------------- Plot 5: Final test RGS, SWA, CWA bars (mean ± SEM) --------------------
try:
    rgs_m, rgs_se = [], []
    swa_m, swa_se = [], []
    cwa_m, cwa_se = [], []
    labels = []
    for lr in learning_rates:
        rgs_runs, swa_runs, cwa_runs = [], [], []
        for exp in all_experiment_data:
            lr_dict = exp.get("learning_rate", {}).get(lr)
            if lr_dict is None:
                continue
            tm = lr_dict["test_metrics"]
            if "rgs" in tm:
                rgs_runs.append(tm["rgs"])
            if "swa" in tm:
                swa_runs.append(tm["swa"])
            if "cwa" in tm:
                cwa_runs.append(tm["cwa"])
        if len(rgs_runs) == 0:  # skip lr values without any runs
            continue
        labels.append(lr)

        # helper to compute mean & sem safely
        def ms(arr):
            return np.mean(arr), (
                (np.std(arr, ddof=1) / np.sqrt(len(arr)))
                if len(arr) > 1
                else (np.mean(arr), 0.0)
            )

        rgsm, rgss = ms(rgs_runs)
        rgs_m.append(rgsm)
        rgs_se.append(rgss)
        swam, swas = ms(swa_runs)
        swa_m.append(swam)
        swa_se.append(swas)
        cwam, cwas = ms(cwa_runs)
        cwa_m.append(cwam)
        cwa_se.append(cwas)

    width = 0.25
    x = np.arange(len(labels))
    plt.figure()
    plt.bar(x - width, rgs_m, yerr=rgs_se, capsize=4, width=width, label="RGS")
    plt.bar(x, swa_m, yerr=swa_se, capsize=4, width=width, label="SWA")
    plt.bar(x + width, cwa_m, yerr=cwa_se, capsize=4, width=width, label="CWA")
    plt.xticks(x, labels)
    plt.title("SPR_Bench Test Metrics (mean ± SEM)\nRGS vs SWA vs CWA")
    plt.xlabel("Learning Rate")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_Bench_test_RGS_SWA_CWA_agg.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated grouped test metrics plot: {e}")
    plt.close()

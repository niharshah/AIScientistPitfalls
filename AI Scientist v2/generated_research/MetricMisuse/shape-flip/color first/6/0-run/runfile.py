import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# -----------------------------------------------------------------
# paths & constants
# -----------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# List of experiment_data.npy files provided in the prompt
experiment_data_path_list = [
    "experiments/2025-08-30_20-55-34_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_3dd52966729c4f759a89ed8646a88873_proc_1497710/experiment_data.npy",
    "experiments/2025-08-30_20-55-34_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_ba909c0085f74f21bd52f86fd8b401f2_proc_1497712/experiment_data.npy",
    "experiments/2025-08-30_20-55-34_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_f85674e6f952456f8cf0e17c7016d2d5_proc_1497711/experiment_data.npy",
]

# -----------------------------------------------------------------
# Load and aggregate data
# -----------------------------------------------------------------
all_runs = []
try:
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp = np.load(full_path, allow_pickle=True).item()
        ed = exp["NodeFeatureProjectionAblation"]["SPR_BENCH"]
        all_runs.append(ed)
except Exception as e:
    print(f"Error loading experiment data: {e}")

if len(all_runs) == 0:
    raise RuntimeError("No experiment runs loaded – cannot plot.")

# Assume all runs share the same epoch vector
epochs = all_runs[0]["epochs"]
ds_name = "SPR_BENCH"
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
n_runs = len(all_runs)


# -----------------------------------------------------------------
# Helper: stack arrays and compute mean+SEM
# -----------------------------------------------------------------
def stack_and_stats(key_chain):
    """Traverse key_chain (list of keys) inside each run, stack, return mean & SEM."""
    collected = []
    for r in all_runs:
        d = r
        for k in key_chain:
            d = d[k]
        collected.append(np.asarray(d))
    arr = np.vstack(collected)  # shape (n_runs, n_epochs)
    mean = arr.mean(axis=0)
    sem = arr.std(axis=0, ddof=1) / np.sqrt(n_runs)
    return mean, sem


# -----------------------------------------------------------------
# Plotting utility
# -----------------------------------------------------------------
def plot_curve(mean_tr, sem_tr, mean_val, sem_val, ylabel, fname_suffix):
    try:
        plt.figure()
        # Train
        plt.plot(epochs, mean_tr, label="train mean", color="tab:blue")
        plt.fill_between(
            epochs,
            mean_tr - sem_tr,
            mean_tr + sem_tr,
            color="tab:blue",
            alpha=0.3,
            label="train ±1 SEM",
        )
        # Validation
        plt.plot(epochs, mean_val, label="val mean", color="tab:orange")
        plt.fill_between(
            epochs,
            mean_val - sem_val,
            mean_val + sem_val,
            color="tab:orange",
            alpha=0.3,
            label="val ±1 SEM",
        )

        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.title(f"{ds_name} – {ylabel} (mean ± SEM over {n_runs} runs)")
        plt.legend()
        save_path = os.path.join(working_dir, f"{ds_name}_{fname_suffix}_{ts}.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating plot {ylabel}: {e}")
        plt.close()


# -----------------------------------------------------------------
# Create plots (max 5 figures – we only make 4)
# -----------------------------------------------------------------
# 1) Loss
m_tr, s_tr = stack_and_stats(["losses", "train"])
m_val, s_val = stack_and_stats(["losses", "val"])
plot_curve(m_tr, s_tr, m_val, s_val, "Loss", "loss_curve")

# 2) Color-Weighted Accuracy
m_tr, s_tr = stack_and_stats(["metrics", "train", "CWA"])
m_val, s_val = stack_and_stats(["metrics", "val", "CWA"])
plot_curve(m_tr, s_tr, m_val, s_val, "Color-Weighted Accuracy", "cwa_curve")

# 3) Shape-Weighted Accuracy
m_tr, s_tr = stack_and_stats(["metrics", "train", "SWA"])
m_val, s_val = stack_and_stats(["metrics", "val", "SWA"])
plot_curve(m_tr, s_tr, m_val, s_val, "Shape-Weighted Accuracy", "swa_curve")

# 4) Complexity-Weighted Accuracy
m_tr, s_tr = stack_and_stats(["metrics", "train", "CplxWA"])
m_val, s_val = stack_and_stats(["metrics", "val", "CplxWA"])
plot_curve(m_tr, s_tr, m_val, s_val, "Complexity-Weighted Accuracy", "cplxwa_curve")

# -----------------------------------------------------------------
# Aggregate and print test metrics
# -----------------------------------------------------------------
try:
    test_cwa = []
    test_swa = []
    test_cplxwa = []
    for r in all_runs:
        t = r["metrics"]["test"]
        test_cwa.append(t["CWA"])
        test_swa.append(t["SWA"])
        test_cplxwa.append(t["CplxWA"])
    print(
        f"Test metrics over {n_runs} runs "
        f"(mean ± std):  CWA={np.mean(test_cwa):.3f}±{np.std(test_cwa,ddof=1):.3f}, "
        f"SWA={np.mean(test_swa):.3f}±{np.std(test_swa,ddof=1):.3f}, "
        f"CplxWA={np.mean(test_cplxwa):.3f}±{np.std(test_cplxwa,ddof=1):.3f}"
    )
except Exception as e:
    print(f"Error computing test metrics: {e}")

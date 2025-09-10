import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
# Basic setup
# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# Load every experiment_data.npy that was provided
# ------------------------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-30_17-49-30_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_a54bca8557344951b55e3158bd8ba95e_proc_1441297/experiment_data.npy",
    "experiments/2025-08-30_17-49-30_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_aabb56a02b9d4130823593c412b4808c_proc_1441298/experiment_data.npy",
    "experiments/2025-08-30_17-49-30_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_c9282642aa774286ba9a258a1ceda2a1_proc_1441296/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        exp = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_experiment_data.append(exp)
    except Exception as e:
        print(f"Error loading {p}: {e}")

if not all_experiment_data:
    print("No experiment data could be loaded – aborting plot generation.")
    exit()

# ------------------------------------------------------------------
# Aggregate data across files for each learning-rate
# ------------------------------------------------------------------
agg = {}  # {lr : dict of lists of arrays}
for exp in all_experiment_data:
    try:
        runs = exp["learning_rate"]["SPR"]
    except KeyError:
        print("Missing expected keys in experiment_data file – skipping.")
        continue
    for lr, run in runs.items():
        bucket = agg.setdefault(
            lr,
            {
                "train_loss": [],
                "val_loss": [],
                "val_acc": [],
                "val_caa": [],
                "val_cwa": [],
                "val_swa": [],
            },
        )
        bucket["train_loss"].append(np.asarray(run["losses"]["train"], dtype=float))
        bucket["val_loss"].append(np.asarray(run["losses"]["val"], dtype=float))
        # Pull validation metrics
        v_metrics = run["metrics"]["val"]
        bucket["val_acc"].append(np.asarray([m["acc"] for m in v_metrics], dtype=float))
        bucket["val_caa"].append(np.asarray([m["caa"] for m in v_metrics], dtype=float))
        bucket["val_cwa"].append(np.asarray([m["cwa"] for m in v_metrics], dtype=float))
        bucket["val_swa"].append(np.asarray([m["swa"] for m in v_metrics], dtype=float))


# Helper that stacks arrays trimming to the minimal length
def stack_and_get_mean_sem(list_of_arrays):
    min_len = min(arr.shape[0] for arr in list_of_arrays)
    trimmed = np.stack([arr[:min_len] for arr in list_of_arrays], axis=0)
    mean = trimmed.mean(axis=0)
    sem = trimmed.std(axis=0, ddof=1) / np.sqrt(trimmed.shape[0])
    return mean, sem, min_len


# Color cycle to remain consistent
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# ------------------------------------------------------------------
# FIGURE 1  – Train & Val Loss with SEM
# ------------------------------------------------------------------
try:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
    for idx, (lr, data) in enumerate(
        sorted(agg.items(), key=lambda x: float(x[0].replace("e-", "e-0")))
    ):
        m_train, sem_train, ep_train = stack_and_get_mean_sem(data["train_loss"])
        m_val, sem_val, ep_val = stack_and_get_mean_sem(data["val_loss"])
        epochs = np.arange(1, min(ep_train, ep_val) + 1)
        c = colors[idx % len(colors)]

        ax[0].plot(epochs, m_train, color=c, label=f"lr={lr}")
        ax[0].fill_between(
            epochs,
            m_train - sem_train[: len(epochs)],
            m_train + sem_train[: len(epochs)],
            color=c,
            alpha=0.3,
        )
        ax[1].plot(epochs, m_val, color=c, label=f"lr={lr}")
        ax[1].fill_between(
            epochs,
            m_val - sem_val[: len(epochs)],
            m_val + sem_val[: len(epochs)],
            color=c,
            alpha=0.3,
        )

    ax[0].set_title("Left: Train Loss (mean ± SEM)")
    ax[1].set_title("Right: Validation Loss (mean ± SEM)")
    for a in ax:
        a.set_xlabel("Epoch")
        a.set_ylabel("Loss")
        a.legend()
    fig.suptitle("SPR – Aggregated Loss Curves Across Learning Rates")
    fname = os.path.join(working_dir, "SPR_aggregated_loss_curves.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# FIGURE 2  – Accuracy & CAA with SEM
# ------------------------------------------------------------------
try:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
    for idx, (lr, data) in enumerate(
        sorted(agg.items(), key=lambda x: float(x[0].replace("e-", "e-0")))
    ):
        m_acc, sem_acc, ep_acc = stack_and_get_mean_sem(data["val_acc"])
        m_caa, sem_caa, ep_caa = stack_and_get_mean_sem(data["val_caa"])
        epochs = np.arange(1, min(ep_acc, ep_caa) + 1)
        c = colors[idx % len(colors)]

        ax[0].plot(epochs, m_acc, color=c, label=f"lr={lr}")
        ax[0].fill_between(
            epochs,
            m_acc - sem_acc[: len(epochs)],
            m_acc + sem_acc[: len(epochs)],
            color=c,
            alpha=0.3,
        )
        ax[1].plot(epochs, m_caa, color=c, label=f"lr={lr}")
        ax[1].fill_between(
            epochs,
            m_caa - sem_caa[: len(epochs)],
            m_caa + sem_caa[: len(epochs)],
            color=c,
            alpha=0.3,
        )

    ax[0].set_title("Left: Accuracy (mean ± SEM)")
    ax[1].set_title("Right: Complexity-Adjusted Accuracy (mean ± SEM)")
    for a in ax:
        a.set_xlabel("Epoch")
        a.set_ylabel("Score")
        a.legend()
    fig.suptitle("SPR – Aggregated Accuracy Metrics")
    fname = os.path.join(working_dir, "SPR_aggregated_accuracy_caa.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating aggregated accuracy/CAA plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# FIGURE 3  – CWA & SWA with SEM
# ------------------------------------------------------------------
try:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
    for idx, (lr, data) in enumerate(
        sorted(agg.items(), key=lambda x: float(x[0].replace("e-", "e-0")))
    ):
        m_cwa, sem_cwa, ep_cwa = stack_and_get_mean_sem(data["val_cwa"])
        m_swa, sem_swa, ep_swa = stack_and_get_mean_sem(data["val_swa"])
        epochs = np.arange(1, min(ep_cwa, ep_swa) + 1)
        c = colors[idx % len(colors)]

        ax[0].plot(epochs, m_cwa, color=c, label=f"lr={lr}")
        ax[0].fill_between(
            epochs,
            m_cwa - sem_cwa[: len(epochs)],
            m_cwa + sem_cwa[: len(epochs)],
            color=c,
            alpha=0.3,
        )
        ax[1].plot(epochs, m_swa, color=c, label=f"lr={lr}")
        ax[1].fill_between(
            epochs,
            m_swa - sem_swa[: len(epochs)],
            m_swa + sem_swa[: len(epochs)],
            color=c,
            alpha=0.3,
        )

    ax[0].set_title("Left: Color-Weighted Accuracy (mean ± SEM)")
    ax[1].set_title("Right: Shape-Weighted Accuracy (mean ± SEM)")
    for a in ax:
        a.set_xlabel("Epoch")
        a.set_ylabel("Score")
        a.legend()
    fig.suptitle("SPR – Aggregated CWA vs SWA")
    fname = os.path.join(working_dir, "SPR_aggregated_cwa_swa.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating aggregated CWA/SWA plot: {e}")
    plt.close()

import matplotlib.pyplot as plt
import numpy as np
import os

# mandatory working dir -------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# load all experiment files
experiment_data_path_list = [
    "experiments/2025-08-15_18-22-30_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_40b4cb8206ce41a4986412c765a09ee4_proc_2971860/experiment_data.npy",
    "experiments/2025-08-15_18-22-30_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_c1db7fd482e74ca394b87e3a52737487_proc_2971858/experiment_data.npy",
    "experiments/2025-08-15_18-22-30_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_bcbaee4828544b68aa53f58380a414b2_proc_2971859/experiment_data.npy",
]

all_runs = []
try:
    root = os.getenv("AI_SCIENTIST_ROOT", "")
    for p in experiment_data_path_list:
        fp = os.path.join(root, p)
        edict = np.load(fp, allow_pickle=True).item()
        # we only use data that exists – expect 'supervised_only'/'SPR_BENCH'
        if "supervised_only" in edict and "SPR_BENCH" in edict["supervised_only"]:
            all_runs.append(edict["supervised_only"]["SPR_BENCH"])
except Exception as e:
    print(f"Error loading experiment data: {e}")

# -----------------------------------------------------------------------------
if len(all_runs) >= 1:
    # gather arrays – first trim to min length so shapes match
    min_len = min(len(r["losses"]["train"]) for r in all_runs)

    def stack(key_chain):
        """Helper to gather series across runs, trimmed to min_len"""
        arrs = []
        for r in all_runs:
            data = r
            for k in key_chain:
                data = data[k]
            arrs.append(np.asarray(data)[:min_len])
        return np.stack(arrs, axis=0)

    losses_train = stack(["losses", "train"])
    losses_val = stack(["losses", "val"])
    swa = stack(["metrics", "val_SWA"])
    cwa = stack(["metrics", "val_CWA"])
    scwa = stack(["metrics", "val_SCWA"])

    epochs = np.arange(1, min_len + 1)

    # compute mean & sem -------------------------------------------------------
    def mean_sem(arr):
        return arr.mean(axis=0), arr.std(axis=0, ddof=0) / np.sqrt(arr.shape[0])

    m_train, sem_train = mean_sem(losses_train)
    m_val, sem_val = mean_sem(losses_val)
    m_swa, sem_swa = mean_sem(swa)
    m_cwa, sem_cwa = mean_sem(cwa)
    m_scwa, sem_scwa = mean_sem(scwa)

    # ------------------------------- plot 1 -----------------------------------
    try:
        plt.figure()
        plt.plot(epochs, m_train, label="Train Loss (mean)")
        plt.fill_between(
            epochs,
            m_train - sem_train,
            m_train + sem_train,
            alpha=0.3,
            label="Train SEM",
        )
        plt.plot(epochs, m_val, label="Val Loss (mean)")
        plt.fill_between(
            epochs, m_val - sem_val, m_val + sem_val, alpha=0.3, label="Val SEM"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Mean Training/Validation Loss with SEM")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_mean_loss_with_sem.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot: {e}")
        plt.close()

    # ------------------------------- plot 2 -----------------------------------
    try:
        plt.figure()
        plt.plot(epochs, m_swa, label="SWA (mean)")
        plt.fill_between(
            epochs, m_swa - sem_swa, m_swa + sem_swa, alpha=0.3, label="SWA SEM"
        )
        plt.plot(epochs, m_cwa, label="CWA (mean)")
        plt.fill_between(
            epochs, m_cwa - sem_cwa, m_cwa + sem_cwa, alpha=0.3, label="CWA SEM"
        )
        plt.plot(epochs, m_scwa, label="SCWA (mean)")
        plt.fill_between(
            epochs, m_scwa - sem_scwa, m_scwa + sem_scwa, alpha=0.3, label="SCWA SEM"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR_BENCH Mean Validation Metrics with SEM")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_mean_metrics_with_sem.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated metrics plot: {e}")
        plt.close()

    # -------------------------- print summary ---------------------------------
    best_epoch = int(np.argmax(m_scwa))
    print(f"Aggregated best epoch (mean SCWA): {best_epoch+1}")
    print(f"Mean SCWA at best epoch: {m_scwa[best_epoch]:.4f}")
else:
    print("No runs could be loaded – nothing to plot.")

import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- basic setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- collect all experiment_data ----------
experiment_data_path_list = [
    "experiments/2025-08-17_18-47-55_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_cf566ac52e1c4ad8b3d83c5116348a2d_proc_3333224/experiment_data.npy",
    "experiments/2025-08-17_18-47-55_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_ca19a710fb1b457fb466370eac43c579_proc_3333223/experiment_data.npy",
    "experiments/2025-08-17_18-47-55_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_bd1f7b2ac7f648a7814196222794578e_proc_3333222/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_experiment_data.append(data)
    except Exception as e:
        print(f"Error loading {p}: {e}")

n_runs = len(all_experiment_data)
if n_runs == 0:
    raise SystemExit("No experiment data could be loaded.")


# ---------- helper to gather arrays ----------
def stack_metric(metric_key):
    """
    metric_key examples:
        ('losses', 'train')
        ('metrics', 'val_MCC')
    Returns stacked ndarray of size (n_runs, min_common_epochs)
    """
    series_list = []
    for exp in all_experiment_data:
        spr_data = exp["LEARNING_RATE"]["SPR_BENCH"]
        best_lr = spr_data["best_lr"]
        rec = spr_data[f"{best_lr:.0e}"]
        # safely navigate nested keys
        ref = rec
        for k in metric_key:
            ref = ref[k]
        series_list.append(np.asarray(ref))
    # align to shortest run length
    min_len = min(len(s) for s in series_list)
    arr = np.stack([s[:min_len] for s in series_list], axis=0)  # (runs, epochs)
    return arr


# ---------- aggregated loss curves ----------
try:
    train_loss = stack_metric(("losses", "train"))
    val_loss = stack_metric(("losses", "val"))
    epochs = np.arange(train_loss.shape[1])

    train_mean = train_loss.mean(axis=0)
    val_mean = val_loss.mean(axis=0)
    train_sem = train_loss.std(axis=0, ddof=1) / np.sqrt(n_runs)
    val_sem = val_loss.std(axis=0, ddof=1) / np.sqrt(n_runs)

    plt.figure()
    plt.plot(epochs, train_mean, label="Train Mean", color="tab:blue")
    plt.fill_between(
        epochs,
        train_mean - train_sem,
        train_mean + train_sem,
        alpha=0.3,
        color="tab:blue",
        label="Train SEM",
    )
    plt.plot(epochs, val_mean, label="Val Mean", color="tab:orange")
    plt.fill_between(
        epochs,
        val_mean - val_sem,
        val_mean + val_sem,
        alpha=0.3,
        color="tab:orange",
        label="Val SEM",
    )

    plt.title("SPR_BENCH Aggregated Loss Curves\nBands show ±1 SEM across runs")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.legend()
    fname = "spr_bench_aggregated_loss_mean_sem.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating aggregated loss curve: {e}")
    plt.close()

# ---------- aggregated MCC curves ----------
try:
    train_mcc = stack_metric(("metrics", "train_MCC"))
    val_mcc = stack_metric(("metrics", "val_MCC"))
    epochs = np.arange(train_mcc.shape[1])

    train_mean = train_mcc.mean(axis=0)
    val_mean = val_mcc.mean(axis=0)
    train_sem = train_mcc.std(axis=0, ddof=1) / np.sqrt(n_runs)
    val_sem = val_mcc.std(axis=0, ddof=1) / np.sqrt(n_runs)

    plt.figure()
    plt.plot(epochs, train_mean, label="Train Mean", color="tab:green")
    plt.fill_between(
        epochs,
        train_mean - train_sem,
        train_mean + train_sem,
        alpha=0.3,
        color="tab:green",
        label="Train SEM",
    )
    plt.plot(epochs, val_mean, label="Val Mean", color="tab:red")
    plt.fill_between(
        epochs,
        val_mean - val_sem,
        val_mean + val_sem,
        alpha=0.3,
        color="tab:red",
        label="Val SEM",
    )

    plt.title("SPR_BENCH Aggregated MCC Curves\nBands show ±1 SEM across runs")
    plt.xlabel("Epoch")
    plt.ylabel("MCC")
    plt.legend()
    fname = "spr_bench_aggregated_mcc_mean_sem.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating aggregated MCC curve: {e}")
    plt.close()

# ---------- aggregated LR sweep summary ----------
try:
    # Gather all unique LR keys
    lr_to_values = {}
    for exp in all_experiment_data:
        spr_data = exp["LEARNING_RATE"]["SPR_BENCH"]
        for lr_key, rec in spr_data.items():
            if lr_key.endswith("e"):  # skip aux keys
                continue
            peak_mcc = max(rec["metrics"]["val_MCC"])
            lr_to_values.setdefault(lr_key, []).append(peak_mcc)

    lrs = sorted(lr_to_values.keys(), key=lambda x: float(x))
    means = np.array([np.mean(lr_to_values[lr]) for lr in lrs])
    sems = np.array(
        [
            np.std(lr_to_values[lr], ddof=1) / np.sqrt(len(lr_to_values[lr]))
            for lr in lrs
        ]
    )

    plt.figure()
    plt.bar(
        lrs, means, yerr=sems, capsize=5, color="skyblue", alpha=0.8, label="Mean ± SEM"
    )
    plt.title("SPR_BENCH LR Sweep (Aggregated)\nPeak Validation MCC per LR ± SEM")
    plt.xlabel("Learning Rate")
    plt.ylabel("Peak Val MCC")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    fname = "spr_bench_lr_sweep_aggregated_mcc.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating aggregated LR sweep plot: {e}")
    plt.close()

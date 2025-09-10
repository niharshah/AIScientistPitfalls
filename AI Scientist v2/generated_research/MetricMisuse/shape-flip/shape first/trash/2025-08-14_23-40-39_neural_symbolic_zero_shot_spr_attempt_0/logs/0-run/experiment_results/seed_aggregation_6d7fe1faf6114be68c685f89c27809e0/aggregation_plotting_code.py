import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- experiment paths (relative to AI_SCIENTIST_ROOT) ----------
experiment_data_path_list = [
    "experiments/2025-08-14_23-40-39_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_bbfce9a0a30641a69dcf31e5406ff473_proc_2777400/experiment_data.npy",
    "experiments/2025-08-14_23-40-39_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_81bb60ea815d44b39dfe91e09f05b9ff_proc_2777399/experiment_data.npy",
    "experiments/2025-08-14_23-40-39_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_ad0d820383f046b58669024f2f488c87_proc_2777401/experiment_data.npy",
]

# ---------- load ----------
all_experiment_data = []
try:
    root = os.getenv("AI_SCIENTIST_ROOT", "")
    for p in experiment_data_path_list:
        full = os.path.join(root, p)
        d = np.load(full, allow_pickle=True).item()
        all_experiment_data.append(d)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []


# ---------- aggregate & plot ----------
def stack_metric(runs, dataset, metric_name):
    """Return stacked metric array of shape (n_runs, n_epochs_aligned)."""
    series = []
    for r in runs:
        try:
            arr = np.asarray(
                r["epochs_tuning"][dataset]["metrics"][metric_name], dtype=float
            )
            series.append(arr)
        except Exception:
            pass
    if not series:
        return None
    min_len = min(len(s) for s in series)
    series = np.stack([s[:min_len] for s in series], axis=0)
    return series


for ds in all_experiment_data[0]["epochs_tuning"].keys() if all_experiment_data else []:
    # collect arrays
    train_loss = stack_metric(all_experiment_data, ds, "train_loss")
    val_loss = stack_metric(all_experiment_data, ds, "val_loss")
    val_swa = stack_metric(all_experiment_data, ds, "val_swa")
    val_cwa = stack_metric(all_experiment_data, ds, "val_cwa")
    val_bps = stack_metric(all_experiment_data, ds, "val_bps")

    # derive epoch axis after alignment
    if train_loss is None:
        continue
    epochs = np.arange(1, train_loss.shape[1] + 1)
    n_runs = train_loss.shape[0]

    # ---------- 1. aggregated loss curves ----------
    try:
        plt.figure()
        mean_tr = train_loss.mean(0)
        se_tr = train_loss.std(0, ddof=1) / np.sqrt(n_runs)
        mean_va = val_loss.mean(0)
        se_va = val_loss.std(0, ddof=1) / np.sqrt(n_runs)

        plt.plot(epochs, mean_tr, label="Train Loss (mean)")
        plt.fill_between(epochs, mean_tr - se_tr, mean_tr + se_tr, alpha=0.3)
        plt.plot(epochs, mean_va, label="Val Loss (mean)")
        plt.fill_between(epochs, mean_va - se_va, mean_va + se_va, alpha=0.3)

        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{ds}: Aggregated Training vs Validation Loss\n(shaded = ±1 s.e.)")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds}_loss_curves_aggregate.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {ds}: {e}")
        plt.close()

    # ---------- 2. aggregated accuracy/BPS curves ----------
    try:
        if val_swa is None:
            raise ValueError("Required metrics missing.")
        plt.figure()
        for metric_arr, name, color in zip(
            [val_swa, val_cwa, val_bps],
            ["Val SWA", "Val CWA", "Val BPS"],
            ["tab:blue", "tab:orange", "tab:green"],
        ):
            if metric_arr is None:
                continue
            m = metric_arr.mean(0)
            se = metric_arr.std(0, ddof=1) / np.sqrt(n_runs)
            plt.plot(epochs, m, label=f"{name} (mean)", color=color)
            plt.fill_between(epochs, m - se, m + se, alpha=0.3, color=color)
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(f"{ds}: Aggregated Validation Metrics\n(shaded = ±1 s.e.)")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds}_accuracy_curves_aggregate.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated accuracy plot for {ds}: {e}")
        plt.close()

    # ---------- 3. print final epoch summary ----------
    last_idx = -1
    try:
        for metric_arr, name in [
            (val_loss, "Val Loss"),
            (val_swa, "Val SWA"),
            (val_cwa, "Val CWA"),
            (val_bps, "Val BPS"),
        ]:
            if metric_arr is None:
                continue
            final_vals = metric_arr[:, last_idx]
            print(
                f"{ds} | {name} @ final epoch: mean={final_vals.mean():.4f}, "
                f"std={final_vals.std(ddof=1):.4f}"
            )
    except Exception as e:
        print(f"Error printing final metrics for {ds}: {e}")

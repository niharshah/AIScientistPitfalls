import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- basic setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load multiple experiment_data files ----------
experiment_data_path_list = [
    "experiments/2025-08-17_22-28-20_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_55e0cbe980f946eba96e761779d63bfd_proc_3445457/experiment_data.npy",
    "experiments/2025-08-17_22-28-20_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_159c8cc43ce144c3914215ae77f987e5_proc_3445459/experiment_data.npy",
    "experiments/2025-08-17_22-28-20_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_e8d7b33f2e73432b92dd2f19a72ab4ac_proc_3445460/experiment_data.npy",
]
all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp)
except Exception as e:
    print(f"Error loading experiment data: {e}")


# ---------- aggregate across runs ----------
def _stack_and_crop(list_of_arrays):
    """Stack 1-D arrays to shape (runs, epochs), cropped to min length."""
    if not list_of_arrays:
        return np.empty((0, 0))
    min_len = min(len(a) for a in list_of_arrays)
    stacked = np.vstack([a[:min_len] for a in list_of_arrays])
    return stacked


for dname in {d for exp in all_experiment_data for d in exp.keys()}:
    # gather per-run curves
    train_losses, val_losses, macro_f1s, cwas = [], [], [], []
    for exp in all_experiment_data:
        logs = exp.get(dname, None)
        if logs is None:
            continue
        train_losses.append(np.asarray(logs["losses"].get("train", []), dtype=float))
        val_losses.append(np.asarray(logs["losses"].get("val", []), dtype=float))
        v_metrics = logs["metrics"].get("val", [])
        macro_f1s.append(
            np.asarray([m["macro_f1"] for m in v_metrics], dtype=float)
            if v_metrics
            else np.array([])
        )
        cwas.append(
            np.asarray([m["cwa"] for m in v_metrics], dtype=float)
            if v_metrics
            else np.array([])
        )

    # convert to (runs, epochs) matrices cropped to common length
    train_mat = _stack_and_crop([a for a in train_losses if a.size])
    val_mat = _stack_and_crop([a for a in val_losses if a.size])
    f1_mat = _stack_and_crop([a for a in macro_f1s if a.size])
    cwa_mat = _stack_and_crop([a for a in cwas if a.size])

    epochs = np.arange(1, train_mat.shape[1] + 1) if train_mat.size else None
    n_runs = train_mat.shape[0] if train_mat.size else 0

    # ---------- aggregated loss curves ----------
    try:
        if train_mat.size and val_mat.size:
            plt.figure()
            # train
            train_mean = train_mat.mean(axis=0)
            train_se = train_mat.std(axis=0, ddof=1) / np.sqrt(n_runs)
            plt.plot(epochs, train_mean, label="train mean", color="tab:blue")
            plt.fill_between(
                epochs,
                train_mean - train_se,
                train_mean + train_se,
                alpha=0.3,
                color="tab:blue",
                label="train ± SE",
            )
            # val
            val_mean = val_mat.mean(axis=0)
            val_se = val_mat.std(axis=0, ddof=1) / np.sqrt(n_runs)
            plt.plot(
                epochs, val_mean, label="val mean", color="tab:orange", linestyle="--"
            )
            plt.fill_between(
                epochs,
                val_mean - val_se,
                val_mean + val_se,
                alpha=0.3,
                color="tab:orange",
                label="val ± SE",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{dname}: Aggregated Training/Validation Loss (n={n_runs} runs)")
            plt.legend()
            plt.savefig(
                os.path.join(working_dir, f"{dname}_loss_curves_aggregated.png")
            )
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss curves for {dname}: {e}")
        plt.close()

    # ---------- aggregated macro-F1 ----------
    try:
        if f1_mat.size:
            epochs_f1 = np.arange(1, f1_mat.shape[1] + 1)
            mean_f1 = f1_mat.mean(axis=0)
            se_f1 = f1_mat.std(axis=0, ddof=1) / np.sqrt(f1_mat.shape[0])
            plt.figure()
            plt.plot(epochs_f1, mean_f1, color="tab:green", label="macro-F1 mean")
            plt.fill_between(
                epochs_f1,
                mean_f1 - se_f1,
                mean_f1 + se_f1,
                alpha=0.3,
                color="tab:green",
                label="macro-F1 ± SE",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Macro-F1")
            plt.title(
                f"{dname}: Aggregated Validation Macro-F1 (n={f1_mat.shape[0]} runs)"
            )
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dname}_macro_f1_aggregated.png"))
            plt.close()

            # quick console summary
            best_epoch = np.argmax(mean_f1)
            print(
                f"{dname}: best mean Macro-F1={mean_f1[best_epoch]:.3f} at epoch {best_epoch+1}"
            )
    except Exception as e:
        print(f"Error creating aggregated macro-F1 for {dname}: {e}")
        plt.close()

    # ---------- aggregated CWA ----------
    try:
        if cwa_mat.size:
            epochs_cwa = np.arange(1, cwa_mat.shape[1] + 1)
            mean_cwa = cwa_mat.mean(axis=0)
            se_cwa = cwa_mat.std(axis=0, ddof=1) / np.sqrt(cwa_mat.shape[0])
            plt.figure()
            plt.plot(epochs_cwa, mean_cwa, color="tab:red", label="CWA mean")
            plt.fill_between(
                epochs_cwa,
                mean_cwa - se_cwa,
                mean_cwa + se_cwa,
                alpha=0.3,
                color="tab:red",
                label="CWA ± SE",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Complexity-Weighted Accuracy")
            plt.title(f"{dname}: Aggregated Validation CWA (n={cwa_mat.shape[0]} runs)")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dname}_cwa_aggregated.png"))
            plt.close()

            # quick console summary
            best_epoch = np.argmax(mean_cwa)
            print(
                f"{dname}: best mean CWA={mean_cwa[best_epoch]:.3f} at epoch {best_epoch+1}"
            )
    except Exception as e:
        print(f"Error creating aggregated CWA for {dname}: {e}")
        plt.close()

import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- data paths ----------
experiment_data_path_list = [
    "experiments/2025-08-14_15-47-52_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_0cd23e688fea4c73af0393ae71795777_proc_2682351/experiment_data.npy",
    "experiments/2025-08-14_15-47-52_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_8a7218a9ce7c4655998d42addbe50ee5_proc_2682349/experiment_data.npy",
    "experiments/2025-08-14_15-47-52_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_5990210ae4aa418391e48b037f06bf36_proc_2682350/experiment_data.npy",
]

# ---------- load all experiments ----------
all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        edata = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(edata)
except Exception as e:
    print(f"Error loading experiment data: {e}")


# ---------- helper to aggregate ----------
def aggregate_metric(list_of_arrays):
    """return mean, se, min_len"""
    list_of_arrays = [np.asarray(a, dtype=float) for a in list_of_arrays if len(a)]
    if not list_of_arrays:
        return None, None, 0
    min_len = min(len(a) for a in list_of_arrays)
    trimmed = np.stack([a[:min_len] for a in list_of_arrays], axis=0)
    mean = trimmed.mean(axis=0)
    se = trimmed.std(axis=0, ddof=1) / np.sqrt(trimmed.shape[0])
    return mean, se, min_len


# ---------- iterate over dataset names ----------
if all_experiment_data:
    # collect union of dataset names appearing in any run
    dset_names = set()
    for ed in all_experiment_data:
        dset_names.update(ed.keys())

    for dset in dset_names:
        # collect per-run arrays
        train_losses, val_losses = [], []
        train_swa, val_swa = [], []
        accuracies = []

        for ed in all_experiment_data:
            rec = ed.get(dset, {})
            train_losses.append(rec.get("losses", {}).get("train", []))
            val_losses.append(rec.get("losses", {}).get("val", []))
            train_swa.append(rec.get("metrics", {}).get("train_swa", []))
            val_swa.append(rec.get("metrics", {}).get("val_swa", []))

            preds = np.asarray(rec.get("predictions", []))
            gts = np.asarray(rec.get("ground_truth", []))
            if preds.size and gts.size and len(preds) == len(gts):
                accuracies.append(float(np.mean(preds == gts)))

        # ------- aggregate curves -------
        tr_mean, tr_se, tr_len = aggregate_metric(train_losses)
        va_mean, va_se, va_len = aggregate_metric(val_losses)
        ts_mean, ts_se, ts_len = aggregate_metric(train_swa)
        vs_mean, vs_se, vs_len = aggregate_metric(val_swa)

        # ------- plot LOSS curves -------
        try:
            if tr_len and va_len:
                epochs = np.arange(1, min(tr_len, va_len) + 1)
                plt.figure()
                plt.plot(
                    epochs, tr_mean[: len(epochs)], label="Train mean", color="tab:blue"
                )
                plt.fill_between(
                    epochs,
                    tr_mean[: len(epochs)] - tr_se[: len(epochs)],
                    tr_mean[: len(epochs)] + tr_se[: len(epochs)],
                    color="tab:blue",
                    alpha=0.3,
                    label="Train ± SE",
                )
                plt.plot(
                    epochs, va_mean[: len(epochs)], label="Val mean", color="tab:orange"
                )
                plt.fill_between(
                    epochs,
                    va_mean[: len(epochs)] - va_se[: len(epochs)],
                    va_mean[: len(epochs)] + va_se[: len(epochs)],
                    color="tab:orange",
                    alpha=0.3,
                    label="Val ± SE",
                )
                plt.xlabel("Epoch")
                plt.ylabel("Cross-Entropy Loss")
                plt.title(f"{dset}: Aggregate Train vs Validation Loss")
                plt.legend()
                fname = os.path.join(working_dir, f"{dset}_aggregate_loss.png")
                plt.savefig(fname)
                plt.close()
        except Exception as e:
            print(f"Error creating aggregate loss plot for {dset}: {e}")
            plt.close()

        # ------- plot SWA curves -------
        try:
            if ts_len and vs_len:
                epochs = np.arange(1, min(ts_len, vs_len) + 1)
                plt.figure()
                plt.plot(
                    epochs,
                    ts_mean[: len(epochs)],
                    label="Train SWA mean",
                    color="tab:green",
                )
                plt.fill_between(
                    epochs,
                    ts_mean[: len(epochs)] - ts_se[: len(epochs)],
                    ts_mean[: len(epochs)] + ts_se[: len(epochs)],
                    color="tab:green",
                    alpha=0.3,
                    label="Train SWA ± SE",
                )
                plt.plot(
                    epochs,
                    vs_mean[: len(epochs)],
                    label="Val SWA mean",
                    color="tab:red",
                )
                plt.fill_between(
                    epochs,
                    vs_mean[: len(epochs)] - vs_se[: len(epochs)],
                    vs_mean[: len(epochs)] + vs_se[: len(epochs)],
                    color="tab:red",
                    alpha=0.3,
                    label="Val SWA ± SE",
                )
                plt.xlabel("Epoch")
                plt.ylabel("Shape-Weighted Accuracy")
                plt.title(f"{dset}: Aggregate Train vs Validation SWA")
                plt.legend()
                fname = os.path.join(working_dir, f"{dset}_aggregate_swa.png")
                plt.savefig(fname)
                plt.close()
        except Exception as e:
            print(f"Error creating aggregate SWA plot for {dset}: {e}")
            plt.close()

        # ------- plot accuracy bar -------
        try:
            if accuracies:
                acc_arr = np.asarray(accuracies, dtype=float)
                mean_acc = acc_arr.mean()
                se_acc = acc_arr.std(ddof=1) / np.sqrt(len(acc_arr))
                plt.figure()
                plt.bar([0], [mean_acc], yerr=[se_acc], capsize=5)
                plt.ylim(0, 1)
                plt.xticks([0], ["Accuracy"])
                plt.title(f"{dset}: Aggregate Test Accuracy (Mean ± SE)")
                fname = os.path.join(working_dir, f"{dset}_aggregate_accuracy.png")
                plt.savefig(fname)
                plt.close()
                print(f"{dset}: mean_test_acc={mean_acc:.4f} ±{se_acc:.4f}")
        except Exception as e:
            print(f"Error creating aggregate accuracy plot for {dset}: {e}")
            plt.close()

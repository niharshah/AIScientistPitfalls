import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# --------------------------------------------------------------------------- #
# helper functions
# --------------------------------------------------------------------------- #
def pad_and_stack(list_of_1d_arrays, pad_val=np.nan):
    """Pad 1-D arrays to the same length with NaN and stack (runs, time)."""
    if not list_of_1d_arrays:
        return None
    max_len = max(len(a) for a in list_of_1d_arrays)
    stacked = np.full((len(list_of_1d_arrays), max_len), pad_val, dtype=float)
    for i, arr in enumerate(list_of_1d_arrays):
        stacked[i, : len(arr)] = arr
    return stacked


def mean_stderr(stacked):
    """Return mean and stderr ignoring NaNs along axis 0."""
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0)
    n = np.sum(~np.isnan(stacked), axis=0)
    stderr = np.where(n > 0, std / np.sqrt(n), np.nan)
    return mean, stderr


# --------------------------------------------------------------------------- #
# load all experiment files
# --------------------------------------------------------------------------- #
experiment_data_path_list = [
    "experiments/2025-08-14_15-47-52_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_b4cc84622fc646bab34c2cb750b6820c_proc_2675006/experiment_data.npy",
    "experiments/2025-08-14_15-47-52_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_e8422addb0674cee9b10e71ab35d9518_proc_2675004/experiment_data.npy",
    "experiments/2025-08-14_15-47-52_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_580ea1caf5ce4012afb6de6ceb21f534_proc_2675003/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        root = os.getenv("AI_SCIENTIST_ROOT", "")
        full_p = os.path.join(root, p)
        data = np.load(full_p, allow_pickle=True).item()
        all_experiment_data.append(data)
    except Exception as e:
        print(f"Error loading {p}: {e}")

# --------------------------------------------------------------------------- #
# aggregate by dataset
# --------------------------------------------------------------------------- #
datasets = {}
for run_idx, run_data in enumerate(all_experiment_data):
    for dset_name, dset_val in run_data.items():
        ds = datasets.setdefault(
            dset_name,
            {"losses": {"train": [], "val": []}, "metrics": {}, "accuracies": []},
        )
        # losses
        losses = dset_val.get("losses", {})
        for phase in ["train", "val"]:
            if phase in losses and len(losses[phase]):
                ds["losses"][phase].append(np.array(losses[phase], dtype=float))
        # metrics
        for mname, mvals in dset_val.get("metrics", {}).items():
            if len(mvals):
                ds["metrics"].setdefault(mname, []).append(np.array(mvals, dtype=float))
        # accuracy
        preds = dset_val.get("predictions")
        gts = dset_val.get("ground_truth")
        if preds is not None and gts is not None and len(preds) == len(gts):
            acc = (np.asarray(preds) == np.asarray(gts)).mean()
            ds["accuracies"].append(acc)

# --------------------------------------------------------------------------- #
# create aggregated plots
# --------------------------------------------------------------------------- #
for dset_name, agg in datasets.items():

    # 1) aggregated loss curves --------------------------------------------- #
    try:
        any_loss = any(len(v) for v in agg["losses"].values())
        if any_loss:
            plt.figure()
            for phase, runs in agg["losses"].items():
                if not runs:
                    continue
                stacked = pad_and_stack(runs)
                mean, stderr = mean_stderr(stacked)
                epochs = np.arange(len(mean))
                plt.plot(epochs, mean, label=f"{phase} mean")
                plt.fill_between(
                    epochs,
                    mean - stderr,
                    mean + stderr,
                    alpha=0.3,
                    label=f"{phase} ± stderr",
                )
            plt.title(f"{dset_name}: Aggregated Training vs Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset_name}_aggregated_loss_curves.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error plotting aggregated loss for {dset_name}: {e}")
        plt.close()

    # 2) aggregated metric curves ------------------------------------------- #
    try:
        if agg["metrics"]:
            for mname, runs in agg["metrics"].items():
                plt.figure()
                stacked = pad_and_stack(runs)
                mean, stderr = mean_stderr(stacked)
                epochs = np.arange(len(mean))
                plt.plot(epochs, mean, label="mean")
                plt.fill_between(
                    epochs, mean - stderr, mean + stderr, alpha=0.3, label="± stderr"
                )
                plt.title(f"{dset_name}: Aggregated {mname} Curve")
                plt.xlabel("Epoch")
                plt.ylabel(mname)
                plt.legend()
                fname = os.path.join(
                    working_dir, f"{dset_name}_aggregated_{mname}_curve.png"
                )
                plt.savefig(fname)
                plt.close()
    except Exception as e:
        print(f"Error plotting aggregated metrics for {dset_name}: {e}")
        plt.close()

    # 3) aggregated accuracy ------------------------------------------------- #
    try:
        accs = agg["accuracies"]
        if accs:
            accs = np.array(accs, dtype=float)
            mean_acc = accs.mean()
            stderr_acc = accs.std(ddof=0) / np.sqrt(len(accs))
            print(
                f"{dset_name}: accuracy mean ± stderr = {mean_acc:.4f} ± {stderr_acc:.4f}"
            )

            plt.figure()
            plt.bar(
                [0],
                [mean_acc],
                yerr=[stderr_acc],
                alpha=0.7,
                capsize=10,
                label=f"mean ± stderr (n={len(accs)})",
            )
            plt.xticks([0], [dset_name])
            plt.ylim(0, 1)
            plt.ylabel("Accuracy")
            plt.title(f"{dset_name}: Aggregated Accuracy Across Runs")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset_name}_aggregated_accuracy.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error plotting aggregated accuracy for {dset_name}: {e}")
        plt.close()

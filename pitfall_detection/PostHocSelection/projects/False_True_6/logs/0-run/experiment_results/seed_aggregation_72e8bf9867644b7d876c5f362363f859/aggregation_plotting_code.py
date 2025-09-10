import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
# basic setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# load every experiment_data.npy that was supplied
experiment_data_path_list = [
    "experiments/2025-08-14_17-37-20_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_9b9e6880e5b44a73a095908a86e968e4_proc_2699130/experiment_data.npy",
    "experiments/2025-08-14_17-37-20_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_57c4bfbf5b304cb99f20fca8220539e1_proc_2699132/experiment_data.npy",
    "None/experiment_data.npy",
]

all_experiment_data = []
for path in experiment_data_path_list:
    try:
        abs_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), path)
        exp_dict = np.load(abs_path, allow_pickle=True).item()
        all_experiment_data.append(exp_dict)
    except Exception as e:
        print(f"Error loading {path}: {e}")


# ------------------------------------------------------------------
# helper: gather arrays of equal length
def stack_and_crop(list_of_lists):
    """Crop every sequence to the minimum length and stack into 2-D array."""
    if not list_of_lists:
        return np.array([])
    min_len = min(len(seq) for seq in list_of_lists)
    if min_len == 0:
        return np.array([])
    return np.vstack([np.array(seq[:min_len]) for seq in list_of_lists])


def epoch_subsample(arr, max_points=100):
    """Downsample long curves for readability while keeping first/last point."""
    if arr.size == 0:
        return np.array([]), np.array([])
    if arr.shape[-1] <= max_points:
        idx = np.arange(arr.shape[-1])
    else:
        idx = np.round(np.linspace(0, arr.shape[-1] - 1, max_points)).astype(int)
    return idx + 1, arr[..., idx]


# ------------------------------------------------------------------
# union of all dataset names appearing in any run
dataset_names = set()
for exp in all_experiment_data:
    dataset_names.update(exp.keys())

for dset in dataset_names:

    # ------------------------------------------------------------------
    # 1) aggregated loss curves
    try:
        train_seq, val_seq = [], []
        for exp in all_experiment_data:
            if dset not in exp:
                continue
            losses = exp[dset].get("losses", {})
            if "train" in losses:
                train_seq.append(losses["train"])
            if "val" in losses:
                val_seq.append(losses["val"])

        train_mat = stack_and_crop(train_seq)
        val_mat = stack_and_crop(val_seq)

        if train_mat.size and val_mat.size:
            train_mean, train_sem = train_mat.mean(0), train_mat.std(
                0, ddof=1
            ) / np.sqrt(train_mat.shape[0])
            val_mean, val_sem = val_mat.mean(0), val_mat.std(0, ddof=1) / np.sqrt(
                val_mat.shape[0]
            )

            ep_train, train_plot = epoch_subsample(train_mean)
            _, train_sems = epoch_subsample(train_sem)
            ep_val, val_plot = epoch_subsample(val_mean)
            _, val_sems = epoch_subsample(val_sem)

            plt.figure()
            plt.plot(ep_train, train_plot, label="Train Mean", color="tab:blue")
            plt.fill_between(
                ep_train,
                train_plot - train_sems,
                train_plot + train_sems,
                color="tab:blue",
                alpha=0.2,
                label="Train ± SEM",
            )
            plt.plot(ep_val, val_plot, label="Val Mean", color="tab:orange")
            plt.fill_between(
                ep_val,
                val_plot - val_sems,
                val_plot + val_sems,
                color="tab:orange",
                alpha=0.2,
                label="Val ± SEM",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(
                f"{dset} Aggregated Loss Curves\nMean ± SEM over {train_mat.shape[0]} runs"
            )
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{dset.lower()}_aggregated_loss.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dset}: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # 2) aggregated validation metrics
    try:
        metrics_names = ("acc", "swa", "cwa", "nrgs")
        metric_arrays = {m: [] for m in metrics_names}

        for exp in all_experiment_data:
            if dset not in exp:
                continue
            val_metrics = exp[dset].get("metrics", {}).get("val", [])
            if not val_metrics:
                continue
            # build per-metric sequence
            for m in metrics_names:
                seq = [ep.get(m) for ep in val_metrics if ep.get(m) is not None]
                if seq:
                    metric_arrays[m].append(seq)

        # plot each metric separately so figure isn't overcrowded
        for m, seqs in metric_arrays.items():
            mat = stack_and_crop(seqs)
            if not mat.size:
                continue
            mean, sem = mat.mean(0), mat.std(0, ddof=1) / np.sqrt(mat.shape[0])
            epochs, mean = epoch_subsample(mean)
            _, sem = epoch_subsample(sem)

            plt.figure()
            plt.plot(epochs, mean, label=f"{m.upper()} Mean", color="tab:green")
            plt.fill_between(
                epochs,
                mean - sem,
                mean + sem,
                color="tab:green",
                alpha=0.2,
                label="± SEM",
            )
            plt.xlabel("Epoch")
            plt.ylabel(m.upper())
            plt.ylim(0, 1)
            plt.title(
                f"{dset} Validation {m.upper()} (Mean ± SEM, {mat.shape[0]} runs)"
            )
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{dset.lower()}_aggregated_{m}.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated metric plot for {dset}: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # 3) aggregated confusion matrix
    try:
        agg_cm = None
        labels_set = set()
        for exp in all_experiment_data:
            if dset not in exp:
                continue
            preds = np.array(exp[dset].get("predictions", []))
            trues = np.array(exp[dset].get("ground_truth", []))
            if preds.size == 0 or trues.size == 0:
                continue
            labels_set.update(np.unique(np.concatenate([preds, trues])))

        if labels_set:
            labels = np.sort(np.array(list(labels_set)))
            label_index = {lbl: i for i, lbl in enumerate(labels)}
            agg_cm = np.zeros((labels.size, labels.size), dtype=int)

            for exp in all_experiment_data:
                if dset not in exp:
                    continue
                preds = np.array(exp[dset].get("predictions", []))
                trues = np.array(exp[dset].get("ground_truth", []))
                for t, p in zip(trues, preds):
                    agg_cm[label_index[t], label_index[p]] += 1

            plt.figure()
            im = plt.imshow(agg_cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.xticks(range(labels.size), labels)
            plt.yticks(range(labels.size), labels)
            plt.title(
                f"{dset} Aggregated Confusion Matrix\nLeft: Ground Truth, Right: Generated Samples"
            )
            # annotate cells
            for i in range(labels.size):
                for j in range(labels.size):
                    plt.text(
                        j,
                        i,
                        agg_cm[i, j],
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=8,
                    )
            plt.tight_layout()
            fname = os.path.join(
                working_dir, f"{dset.lower()}_aggregated_confusion_matrix.png"
            )
            plt.savefig(fname)
            print(f"Saved {fname}")
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated confusion matrix for {dset}: {e}")
        plt.close()

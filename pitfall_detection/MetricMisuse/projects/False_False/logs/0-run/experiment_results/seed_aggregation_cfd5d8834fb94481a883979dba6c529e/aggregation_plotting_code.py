import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------- basic setup -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- load all experiment_data dicts -----------------
experiment_data_path_list = [
    "experiments/2025-08-15_18-22-30_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_66e67345d9a8412a9cfbac8c0c044a67_proc_2960009/experiment_data.npy",
    "experiments/2025-08-15_18-22-30_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_2c8c226ca6f64bc0964989d8bef1a88f_proc_2960008/experiment_data.npy",
    "experiments/2025-08-15_18-22-30_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_cfd7fc617253463b9f48242d1bd2102e_proc_2960006/experiment_data.npy",
]
all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        all_experiment_data.append(np.load(full, allow_pickle=True).item())
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []


# ----------------- aggregate per-dataset information --------------
def confusion(y_true, y_pred, num_cls):
    cm = np.zeros((num_cls, num_cls), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


# collect union of dataset names
dataset_names = set()
for exp in all_experiment_data:
    dataset_names |= set(exp.keys())

for dset in dataset_names:
    # gather runs that contain this dataset
    runs_for_dataset = [exp[dset] for exp in all_experiment_data if dset in exp]
    if not runs_for_dataset:
        continue

    # ------------------- aggregate loss curves -------------------
    try:
        train_losses = [
            np.asarray(r["losses"]["train"]) for r in runs_for_dataset if "losses" in r
        ]
        val_losses = [
            np.asarray(r["losses"]["val"]) for r in runs_for_dataset if "losses" in r
        ]
        # make sure we have at least one run with both curves
        if train_losses and val_losses:
            min_len = min(map(len, train_losses + val_losses))
            train_mat = np.stack([tl[:min_len] for tl in train_losses])
            val_mat = np.stack([vl[:min_len] for vl in val_losses])
            epochs = np.arange(1, min_len + 1)

            train_mean, train_se = train_mat.mean(axis=0), train_mat.std(
                axis=0
            ) / np.sqrt(train_mat.shape[0])
            val_mean, val_se = val_mat.mean(axis=0), val_mat.std(axis=0) / np.sqrt(
                val_mat.shape[0]
            )

            plt.figure()
            plt.plot(epochs, train_mean, label="Train Mean", color="tab:blue")
            plt.fill_between(
                epochs,
                train_mean - train_se,
                train_mean + train_se,
                alpha=0.3,
                color="tab:blue",
                label="Train ±SE",
            )
            plt.plot(epochs, val_mean, label="Val Mean", color="tab:orange")
            plt.fill_between(
                epochs,
                val_mean - val_se,
                val_mean + val_se,
                alpha=0.3,
                color="tab:orange",
                label="Val ±SE",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(
                f"{dset}: Aggregated Train/Val Loss\n(Mean ± Standard Error over {train_mat.shape[0]} runs)"
            )
            plt.legend()
            fname = os.path.join(working_dir, f"{dset}_aggregate_loss_curve.png")
            plt.savefig(fname)
            plt.close()
            print("Saved", fname)
    except Exception as e:
        print(f"Error creating aggregate loss curve for {dset}: {e}")
        plt.close()

    # ------------------- aggregate macro-F1 curves ---------------
    try:
        tF1 = [
            np.asarray(r["metrics"]["train_macroF1"])
            for r in runs_for_dataset
            if "metrics" in r and "train_macroF1" in r["metrics"]
        ]
        vF1 = [
            np.asarray(r["metrics"]["val_macroF1"])
            for r in runs_for_dataset
            if "metrics" in r and "val_macroF1" in r["metrics"]
        ]
        if tF1 and vF1:
            min_len = min(map(len, tF1 + vF1))
            tF1_mat = np.stack([a[:min_len] for a in tF1])
            vF1_mat = np.stack([a[:min_len] for a in vF1])
            epochs = np.arange(1, min_len + 1)

            t_mean, t_se = tF1_mat.mean(axis=0), tF1_mat.std(axis=0) / np.sqrt(
                tF1_mat.shape[0]
            )
            v_mean, v_se = vF1_mat.mean(axis=0), vF1_mat.std(axis=0) / np.sqrt(
                vF1_mat.shape[0]
            )

            plt.figure()
            plt.plot(epochs, t_mean, label="Train Mean", color="tab:green")
            plt.fill_between(
                epochs,
                t_mean - t_se,
                t_mean + t_se,
                alpha=0.3,
                color="tab:green",
                label="Train ±SE",
            )
            plt.plot(epochs, v_mean, label="Val Mean", color="tab:red")
            plt.fill_between(
                epochs,
                v_mean - v_se,
                v_mean + v_se,
                alpha=0.3,
                color="tab:red",
                label="Val ±SE",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Macro-F1")
            plt.title(
                f"{dset}: Aggregated Train/Val Macro-F1\n(Mean ± Standard Error over {tF1_mat.shape[0]} runs)"
            )
            plt.legend()
            fname = os.path.join(working_dir, f"{dset}_aggregate_macroF1_curve.png")
            plt.savefig(fname)
            plt.close()
            print("Saved", fname)
    except Exception as e:
        print(f"Error creating aggregate F1 curve for {dset}: {e}")
        plt.close()

    # ------------------- aggregate confusion matrix --------------
    try:
        cms = []
        num_cls = 0
        for r in runs_for_dataset:
            y_true = np.array(r.get("ground_truth", []))
            y_pred = np.array(r.get("predictions", []))
            if y_true.size == 0:
                continue
            n_cls = int(max(y_true.max(), y_pred.max()) + 1)
            num_cls = max(num_cls, n_cls)
            cms.append(confusion(y_true, y_pred, n_cls))
        if cms and num_cls:
            agg_cm = np.zeros((num_cls, num_cls), dtype=int)
            for cm in cms:
                # expand if needed
                if cm.shape[0] < num_cls:
                    pad = np.zeros((num_cls, num_cls), dtype=int)
                    pad[: cm.shape[0], : cm.shape[1]] = cm
                    cm = pad
                agg_cm += cm
            cm_perc = (
                agg_cm / agg_cm.sum(axis=1, keepdims=True).clip(min=1) * 100
            )  # row-normalised
            plt.figure()
            im = plt.imshow(cm_perc, cmap="Blues")
            plt.colorbar(im, label="% of class")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{dset}: Aggregated Confusion Matrix (Validation)")
            fname = os.path.join(working_dir, f"{dset}_aggregate_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
            print("Saved", fname)
    except Exception as e:
        print(f"Error creating aggregate confusion matrix for {dset}: {e}")
        plt.close()

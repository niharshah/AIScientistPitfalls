import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
# basic set-up
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# list of experiment result files supplied by the user
experiment_data_path_list = [
    "experiments/2025-08-30_20-55-34_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_516db75062474672b51dbec9267360e6_proc_1488361/experiment_data.npy",
    "experiments/2025-08-30_20-55-34_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_002ca1edf07b4b6283f7f75f6bded02e_proc_1488363/experiment_data.npy",
    "experiments/2025-08-30_20-55-34_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_6100461ca1444403b3a391a1be6863c6_proc_1488360/experiment_data.npy",
]

# ------------------------------------------------------------------
# load all experiments
all_experiment_data = []
try:
    for p in experiment_data_path_list:
        abs_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp_dict = np.load(abs_path, allow_pickle=True).item()
        all_experiment_data.append(exp_dict)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# ------------------------------------------------------------------
# collect all dataset names present in any run
dataset_names = set()
for exp in all_experiment_data:
    dataset_names.update(exp.keys())


# ------------------------------------------------------------------
def stack_and_trim(list_of_arrays):
    """Stack 1-D arrays after trimming all to the minimum length."""
    if not list_of_arrays:
        return np.empty((0, 0))
    min_len = min(len(a) for a in list_of_arrays)
    trimmed = np.stack([a[:min_len] for a in list_of_arrays], axis=0)
    return trimmed


# ------------------------------------------------------------------
# iterate per dataset and create plots
for dname in dataset_names:
    # gather per-run arrays ----------------------------------------------------
    epochs_runs, train_runs, val_runs, compwa_runs = [], [], [], []
    preds_runs, gts_runs = [], []

    for exp in all_experiment_data:
        if dname not in exp:
            continue
        dct = exp[dname]
        epochs_runs.append(np.asarray(dct.get("epochs", [])))
        train_runs.append(np.asarray(dct.get("losses", {}).get("train", [])))
        val_runs.append(np.asarray(dct.get("losses", {}).get("val", [])))
        compwa_runs.append(np.asarray(dct.get("metrics", {}).get("val_compwa", [])))
        preds_runs.append(np.asarray(dct.get("predictions", [])))
        gts_runs.append(np.asarray(dct.get("ground_truth", [])))

    # -------------------- aggregated loss curve ------------------------------
    try:
        train_mat = stack_and_trim(train_runs)
        val_mat = stack_and_trim(val_runs)
        epoch_mat = stack_and_trim(epochs_runs)
        if train_mat.size and val_mat.size and epoch_mat.size:
            epochs = epoch_mat[0]  # after trimming, identical across runs
            train_mean = train_mat.mean(axis=0)
            train_se = train_mat.std(axis=0, ddof=1) / np.sqrt(train_mat.shape[0])
            val_mean = val_mat.mean(axis=0)
            val_se = val_mat.std(axis=0, ddof=1) / np.sqrt(val_mat.shape[0])

            plt.figure()
            plt.plot(epochs, train_mean, color="tab:blue", label="Train Loss – mean")
            plt.fill_between(
                epochs,
                train_mean - train_se,
                train_mean + train_se,
                color="tab:blue",
                alpha=0.25,
                label="Train ±SEM",
            )
            plt.plot(epochs, val_mean, color="tab:orange", label="Val Loss – mean")
            plt.fill_between(
                epochs,
                val_mean - val_se,
                val_mean + val_se,
                color="tab:orange",
                alpha=0.25,
                label="Val ±SEM",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(
                f"{dname} – Aggregated Training/Validation Loss (N={train_mat.shape[0]})"
            )
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_aggregated_train_val_loss.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss curve for {dname}: {e}")
        plt.close()

    # -------------------- aggregated CompWA curve ----------------------------
    try:
        compwa_mat = stack_and_trim(compwa_runs)
        epoch_mat = stack_and_trim(epochs_runs)
        if compwa_mat.size and epoch_mat.size:
            epochs = epoch_mat[0]
            compwa_mean = compwa_mat.mean(axis=0)
            compwa_se = compwa_mat.std(axis=0, ddof=1) / np.sqrt(compwa_mat.shape[0])

            plt.figure()
            plt.plot(epochs, compwa_mean, color="tab:green", label="Val CompWA – mean")
            plt.fill_between(
                epochs,
                compwa_mean - compwa_se,
                compwa_mean + compwa_se,
                color="tab:green",
                alpha=0.25,
                label="±SEM",
            )
            plt.xlabel("Epoch")
            plt.ylabel("CompWA")
            plt.title(
                f"{dname} – Aggregated Validation CompWA (N={compwa_mat.shape[0]})"
            )
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_aggregated_compwa.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated CompWA curve for {dname}: {e}")
        plt.close()

    # -------------------- aggregated confusion matrix ------------------------
    try:
        # accumulate total confusion counts
        class_set = set()
        for g, gp in zip(gts_runs, preds_runs):
            class_set.update(g)
            class_set.update(gp)
        if class_set:
            classes = sorted(list(class_set))
            conf_total = np.zeros((len(classes), len(classes)), dtype=int)
            for g, p in zip(gts_runs, preds_runs):
                for true, pred in zip(g, p):
                    ti, pi = classes.index(true), classes.index(pred)
                    conf_total[ti, pi] += 1

            plt.figure()
            im = plt.imshow(conf_total, cmap="Blues")
            plt.colorbar(im)
            plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
            plt.yticks(range(len(classes)), classes)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(f"{dname} – Aggregated Confusion Matrix (All Runs)")
            fname = os.path.join(
                working_dir, f"{dname}_aggregated_confusion_matrix.png"
            )
            plt.savefig(fname, bbox_inches="tight")
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated confusion matrix for {dname}: {e}")
        plt.close()

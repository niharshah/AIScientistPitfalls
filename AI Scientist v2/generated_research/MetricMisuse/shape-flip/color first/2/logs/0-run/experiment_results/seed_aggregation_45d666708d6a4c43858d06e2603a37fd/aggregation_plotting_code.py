import matplotlib.pyplot as plt
import numpy as np
import os

# ----------- setup -----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------- paths -----------
experiment_data_path_list = [
    "experiments/2025-08-30_17-49-38_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_29bbd6bde4ab44fd9c48098b83b32ec9_proc_1437146/experiment_data.npy",
    "experiments/2025-08-30_17-49-38_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_3e594b895c8c494faff0cdea61726d42_proc_1437144/experiment_data.npy",
    "experiments/2025-08-30_17-49-38_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_899352698ddc417ebbc401f3ac717803_proc_1437145/experiment_data.npy",
]

all_experiment_data = []
try:
    root = os.getenv("AI_SCIENTIST_ROOT", "")
    for p in experiment_data_path_list:
        full_path = os.path.join(root, p) if root else p
        all_experiment_data.append(np.load(full_path, allow_pickle=True).item())
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []


# ----------- helper -----------
def unpack(pairs):
    """convert list of (ts,val) into list[val]"""
    return [v for ts, v in pairs if v is not None] if pairs else []


# ----------- plotting per dataset -----------
for dataset_name in all_experiment_data[0].keys() if all_experiment_data else []:
    # collect run-specific arrays
    train_losses, val_losses, val_dwas = [], [], []
    preds_all, gts_all = [], []
    min_epochs = np.inf

    for run in all_experiment_data:
        data = run.get(dataset_name, {})
        if not data:
            continue
        tl = np.array(unpack(data["losses"]["train"]))
        vl = np.array(unpack(data["losses"]["val"]))
        vd = np.array(unpack(data["metrics"]["val"]))
        min_epochs = min(min_epochs, len(tl), len(vl), len(vd))
        train_losses.append(tl)
        val_losses.append(vl)
        val_dwas.append(vd)
        preds_all.append(np.array(data["predictions"]))
        gts_all.append(np.array(data["ground_truth"]))

    if len(train_losses) == 0:
        continue  # nothing to plot

    # align to common epoch length
    min_epochs = int(min_epochs)
    train_mat = np.vstack([tl[:min_epochs] for tl in train_losses])
    val_mat = np.vstack([vl[:min_epochs] for vl in val_losses])
    dwa_mat = np.vstack([vd[:min_epochs] for vd in val_dwas])
    epochs = np.arange(1, min_epochs + 1)

    # compute mean & SEM
    def mean_sem(mat):
        mean = mat.mean(axis=0)
        sem = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
        return mean, sem

    mean_train, sem_train = mean_sem(train_mat)
    mean_val, sem_val = mean_sem(val_mat)
    mean_dwa, sem_dwa = mean_sem(dwa_mat)

    # -------- Loss plot --------
    try:
        plt.figure()
        plt.plot(epochs, mean_train, "-o", label="Mean Train Loss")
        plt.fill_between(
            epochs,
            mean_train - sem_train,
            mean_train + sem_train,
            alpha=0.3,
            label="Train SEM",
        )
        plt.plot(epochs, mean_val, "-o", label="Mean Val Loss")
        plt.fill_between(
            epochs, mean_val - sem_val, mean_val + sem_val, alpha=0.3, label="Val SEM"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(
            f"{dataset_name}: Mean Training vs Validation Loss\n(n={train_mat.shape[0]} runs)"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"{dataset_name}_loss_curves_aggregated.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dataset_name}: {e}")
        plt.close()

    # -------- DWA plot --------
    try:
        plt.figure()
        plt.plot(epochs, mean_dwa, "-o", color="green", label="Mean Val DWA")
        plt.fill_between(
            epochs,
            mean_dwa - sem_dwa,
            mean_dwa + sem_dwa,
            alpha=0.3,
            color="green",
            label="DWA SEM",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Dual-Weighted Accuracy")
        plt.title(f"{dataset_name}: Mean Validation DWA\n(n={dwa_mat.shape[0]} runs)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"{dataset_name}_DWA_curve_aggregated.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated DWA plot for {dataset_name}: {e}")
        plt.close()

    # -------- Confusion matrix --------
    try:
        # aggregate predictions & ground truths
        preds = np.concatenate(preds_all)
        gts = np.concatenate(gts_all)
        classes = sorted(set(gts) | set(preds))
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for t, p in zip(gts, preds):
            cm[classes.index(t), classes.index(p)] += 1

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(
            f"{dataset_name}: Aggregated Confusion Matrix\nLeft: Ground Truth, Bottom: Predicted"
        )
        plt.xticks(range(len(classes)), classes)
        plt.yticks(range(len(classes)), classes)
        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"{dataset_name}_confusion_matrix_aggregated.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated confusion matrix for {dataset_name}: {e}")
        plt.close()

import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------ #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------ #
# 1. Load every provided experiment_data.npy
try:
    experiment_data_path_list = [
        "experiments/2025-08-17_00-43-58_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_6b3f51b9f2e746ab9d7768697bea6fa6_proc_3173676/experiment_data.npy",
        "experiments/2025-08-17_00-43-58_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_71b8322e22404f3e8e_proc_3173675/experiment_data.npy",
        "experiments/2025-08-17_00-43-58_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_9cb7de2dd6be459cb4a4b7df37c09d0a_proc_3173677/experiment_data.npy",
    ]
    all_experiment_data = []
    for p in experiment_data_path_list:
        exp = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_experiment_data.append(exp)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

if not all_experiment_data:
    raise RuntimeError(
        "No experiment data could be loaded, aborting aggregation plots."
    )

# ------------------------------------------------------------------ #
# 2. Aggregate metrics
train_loss, val_loss = [], []
train_acc, val_acc = [], []
train_f1, val_f1 = [], []
all_preds, all_gts = [], []

for exp in all_experiment_data:
    spr = exp["SPR_BENCH"]
    metrics, losses = spr["metrics"], spr["losses"]
    train_loss.append(np.asarray(losses["train"]))
    val_loss.append(np.asarray(losses["val"]))

    train_acc.append(np.asarray(metrics["train_acc"]))
    val_acc.append(np.asarray(metrics["val_acc"]))

    train_f1.append(np.asarray(metrics["train_f1"]))
    val_f1.append(np.asarray(metrics["val_f1"]))

    all_preds.append(np.asarray(spr["predictions"]))
    all_gts.append(np.asarray(spr["ground_truth"]))


# Ensure equal length across runs for each metric
def stack_and_trim(list_of_arrays):
    min_len = min(arr.shape[0] for arr in list_of_arrays)
    arr = np.stack([a[:min_len] for a in list_of_arrays], axis=0)
    return arr


stacked = {
    "loss_train": stack_and_trim(train_loss),
    "loss_val": stack_and_trim(val_loss),
    "acc_train": stack_and_trim(train_acc),
    "acc_val": stack_and_trim(val_acc),
    "f1_train": stack_and_trim(train_f1),
    "f1_val": stack_and_trim(val_f1),
}


# ------------------------------------------------------------------ #
# 3. Helper to compute mean and SEM
def mean_sem(arr):
    mean = arr.mean(axis=0)
    sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
    return mean, sem


plots_info = [
    (
        "agg_loss_curve",
        ("loss_train", "loss_val"),
        "Loss",
        "Aggregated Train vs. Val Loss",
    ),
    (
        "agg_acc_curve",
        ("acc_train", "acc_val"),
        "Accuracy",
        "Aggregated Train vs. Val Accuracy",
    ),
    (
        "agg_f1_curve",
        ("f1_train", "f1_val"),
        "Macro-F1",
        "Aggregated Train vs. Val Macro-F1",
    ),
]

# ------------------------------------------------------------------ #
# 4. Create plots with shaded SEM
for fname, (train_key, val_key), ylabel, title in plots_info:
    try:
        plt.figure()
        # Train
        mean_t, sem_t = mean_sem(stacked[train_key])
        epochs = np.arange(len(mean_t))
        plt.plot(epochs, mean_t, label="Train Mean", color="tab:blue")
        plt.fill_between(
            epochs,
            mean_t - sem_t,
            mean_t + sem_t,
            alpha=0.3,
            color="tab:blue",
            label="Train ±SEM",
        )

        # Val
        mean_v, sem_v = mean_sem(stacked[val_key])
        plt.plot(epochs, mean_v, label="Val Mean", color="tab:orange")
        plt.fill_between(
            epochs,
            mean_v - sem_v,
            mean_v + sem_v,
            alpha=0.3,
            color="tab:orange",
            label="Val ±SEM",
        )

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{title} (Dataset: SPR_BENCH)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"spr_bench_{fname}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {fname}: {e}")
        plt.close()

# ------------------------------------------------------------------ #
# 5. Aggregated confusion matrix (mean counts)
try:
    # First, ensure all runs have same label space
    labels = np.unique(np.concatenate(all_gts))
    n_cls = len(labels)
    cm_accum = np.zeros((n_cls, n_cls), dtype=float)

    for preds, gts in zip(all_preds, all_gts):
        cm = np.zeros((n_cls, n_cls), dtype=float)
        for t, p in zip(gts, preds):
            cm[int(t), int(p)] += 1
        cm_accum += cm
    cm_mean = cm_accum / len(all_preds)

    plt.figure()
    im = plt.imshow(cm_mean, cmap="Blues")
    plt.colorbar(im)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("Mean Confusion Matrix (Dataset: SPR_BENCH)")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "spr_bench_mean_confusion_matrix.png"))
    plt.close()
except Exception as e:
    print(f"Error creating mean confusion matrix: {e}")
    plt.close()


# ------------------------------------------------------------------ #
# 6. Print aggregated test metrics
def macro_f1(pred, true, n_cls):
    f1s = []
    for c in range(n_cls):
        tp = ((pred == c) & (true == c)).sum()
        fp = ((pred == c) & (true != c)).sum()
        fn = ((pred != c) & (true == c)).sum()
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        prec, rec = tp / (tp + fp), tp / (tp + fn)
        f1s.append(0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


acc_list, f1_list = [], []
for preds, gts in zip(all_preds, all_gts):
    acc_list.append((preds == gts).mean())
    f1_list.append(macro_f1(preds, gts, n_cls))


def mean_sem_scalar(lst):
    arr = np.asarray(lst, dtype=float)
    mean = arr.mean()
    sem = arr.std(ddof=1) / np.sqrt(arr.shape[0])
    return mean, sem


mean_acc, sem_acc = mean_sem_scalar(acc_list)
mean_f1, sem_f1 = mean_sem_scalar(f1_list)

print(f"Aggregated Test Accuracy : {mean_acc*100:.2f}% ± {sem_acc*100:.2f}%")
print(f"Aggregated Test Macro-F1 : {mean_f1:.4f} ± {sem_f1:.4f}")

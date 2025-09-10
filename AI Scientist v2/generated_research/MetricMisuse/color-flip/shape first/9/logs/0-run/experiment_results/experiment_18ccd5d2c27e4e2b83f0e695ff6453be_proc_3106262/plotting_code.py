import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper to fetch arrays safely
def get_path(dic, path, default=None):
    for k in path:
        if k not in dic:
            return default
        dic = dic[k]
    return dic if dic else default


loss_train = get_path(
    experiment_data,
    ["MultiDatasetContrastivePretrain", "SPR_BENCH", "losses", "train"],
    [],
)
loss_val = get_path(
    experiment_data,
    ["MultiDatasetContrastivePretrain", "SPR_BENCH", "losses", "val"],
    [],
)
cwa_val = get_path(
    experiment_data,
    ["MultiDatasetContrastivePretrain", "SPR_BENCH", "metrics", "val"],
    [],
)
preds = get_path(
    experiment_data, ["MultiDatasetContrastivePretrain", "SPR_BENCH", "predictions"], []
)
gts = get_path(
    experiment_data,
    ["MultiDatasetContrastivePretrain", "SPR_BENCH", "ground_truth"],
    [],
)

# ---------- 1) Loss curves ----------
try:
    epochs = np.arange(1, len(loss_train) + 1)
    plt.figure()
    plt.plot(epochs, loss_train, label="Train Loss")
    plt.plot(epochs, loss_val, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ---------- 2) CWA curve ----------
try:
    if cwa_val:
        epochs = np.arange(1, len(cwa_val) + 1)
        plt.figure()
        plt.plot(epochs, cwa_val, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Comp-Weighted Accuracy")
        plt.title("SPR_BENCH Validation CWA per Epoch")
        fname = os.path.join(working_dir, "SPR_BENCH_CWA_curve.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating CWA plot: {e}")
    plt.close()

# ---------- 3) Confusion matrix ----------
try:
    if preds and gts:
        preds = np.array(preds)
        gts = np.array(gts)
        labels = np.unique(np.concatenate([gts, preds]))
        cm = np.zeros((labels.size, labels.size), dtype=int)
        for t, p in zip(gts, preds):
            cm[np.where(labels == t)[0][0], np.where(labels == p)[0][0]] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xticks(range(len(labels)), labels)
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH Confusion Matrix\nLeft: GT, Right: Predicted")
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=8,
                )
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

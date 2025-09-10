import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ds_name = "SPR_BENCH"
ds = experiment_data.get(ds_name, {})


# ---------- helper ----------
def safe_get(path, default=None):
    tmp = ds
    for k in path:
        tmp = tmp.get(k, {})
    return tmp if tmp else default


loss_train = safe_get(["losses", "train"], [])
loss_val = safe_get(["losses", "val"], [])
val_bwa = safe_get(["metrics", "val"], [])
preds = ds.get("predictions", [])
gts = ds.get("ground_truth", [])

epochs = np.arange(1, len(loss_train) + 1)

# ---------- 1. Loss curves ----------
try:
    plt.figure()
    if loss_train:
        plt.plot(epochs, loss_train, label="Train Loss")
    if loss_val:
        plt.plot(epochs, loss_val, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"{ds_name} Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, f"{ds_name}_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ---------- 2. Validation BWA ----------
try:
    if val_bwa and any(v is not None for v in val_bwa):
        plt.figure()
        plt.plot(
            epochs,
            [v if v is not None else np.nan for v in val_bwa],
            marker="o",
            label="Validation BWA",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Balanced Weighted Accuracy")
        plt.title(f"{ds_name} Validation Balanced-Weighted Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name}_validation_bwa.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating BWA plot: {e}")
    plt.close()

# ---------- 3. Confusion matrix ----------
try:
    if preds and gts:
        labels = sorted(set(gts) | set(preds))
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(gts, preds):
            cm[labels.index(t), labels.index(p)] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xticks(range(len(labels)), labels)
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(f"{ds_name} Confusion Matrix")
        fname = os.path.join(working_dir, f"{ds_name}_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------- final metric print ----------
if preds and gts:
    test_acc = np.mean(np.array(preds) == np.array(gts))
    print(f"{ds_name} Test Accuracy: {test_acc:.4f}")

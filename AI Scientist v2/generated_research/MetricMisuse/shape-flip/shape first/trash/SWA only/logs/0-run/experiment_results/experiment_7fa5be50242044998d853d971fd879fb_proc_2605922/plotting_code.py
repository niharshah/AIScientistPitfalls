import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ---------- helper ----------
def get_entry(expd, abl="TokenOrderShuffled", ds="SPR_BENCH"):
    return expd.get(abl, {}).get(ds, {})


ed = get_entry(experiment_data)
if not ed:
    print("No experiment data found for TokenOrderShuffled / SPR_BENCH.")
    exit()

loss_tr = ed["losses"]["train"]
loss_va = ed["losses"]["val"]
acc_tr = [m["acc"] for m in ed["metrics"]["train"]]
acc_va = [m["acc"] for m in ed["metrics"]["val"]]
swa_tr = [m["swa"] for m in ed["metrics"]["train"]]
swa_va = [m["swa"] for m in ed["metrics"]["val"]]
preds = ed.get("predictions", [])
gts = ed.get("ground_truth", [])
test_metrics = ed["metrics"]["test"]

# ---------- plots ----------

# 1) Loss curves
try:
    plt.figure()
    plt.plot(loss_tr, label="Train loss")
    plt.plot(loss_va, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves – SPR_BENCH\nLeft: Train, Right: Validation")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2) Accuracy curves
try:
    plt.figure()
    plt.plot(acc_tr, label="Train Acc")
    plt.plot(acc_va, label="Val Acc")
    plt.plot(swa_tr, label="Train SWA")
    plt.plot(swa_va, label="Val SWA")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves – SPR_BENCH\nPlain vs Shape-Weighted")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curve: {e}")
    plt.close()

# 3) Confusion matrix
try:
    if preds and gts:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(gts, preds, labels=[0, 1])
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.colorbar()
        plt.xticks([0, 1], ["0", "1"])
        plt.yticks([0, 1], ["0", "1"])
        plt.title(
            "Confusion Matrix – SPR_BENCH\nLeft: Ground Truth, Right: Predictions"
        )
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------- print metrics ----------
print(f"Test Accuracy: {test_metrics.get('acc', 'N/A'):.3f}")
print(f"Test Shape-Weighted Accuracy: {test_metrics.get('swa', 'N/A'):.3f}")

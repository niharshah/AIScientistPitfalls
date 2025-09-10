import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------
# load experiment data
# ------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ed = experiment_data.get("SPR_BENCH", {})
if not ed:
    print("No SPR_BENCH data found.")
    exit()

epochs = list(range(1, len(ed["losses"]["train"]) + 1))
train_loss = ed["losses"]["train"]
val_loss = ed["losses"]["val"]
train_acc = [m["acc"] for m in ed["metrics"]["train"]]
val_acc = [m["acc"] for m in ed["metrics"]["val"]]
val_cswa = [m["CSWA"] for m in ed["metrics"]["val"]]

# ------------------------------------------------------------
# 1) Loss curves
# ------------------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ------------------------------------------------------------
# 2) Accuracy curves
# ------------------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, train_acc, label="Train")
    plt.plot(epochs, val_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH: Training vs Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_accuracy_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curve: {e}")
    plt.close()

# ------------------------------------------------------------
# 3) CSWA score
# ------------------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, val_cswa, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("CSWA")
    plt.title("SPR_BENCH: Validation CSWA Over Epochs")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_CSWA_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CSWA curve: {e}")
    plt.close()

# ------------------------------------------------------------
# 4) Confusion matrix on test set
# ------------------------------------------------------------
try:
    preds = np.array(ed.get("predictions", []))
    gts = np.array(ed.get("ground_truth", []))
    n_cls = int(max(preds.max(), gts.max()) + 1) if preds.size else 0
    if n_cls > 0:
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for p, g in zip(preds, gts):
            cm[g, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("SPR_BENCH: Confusion Matrix (Test Set)")
        plt.xticks(range(n_cls))
        plt.yticks(range(n_cls))
        for i in range(n_cls):
            for j in range(n_cls):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

print(f"All plots saved to {working_dir}")

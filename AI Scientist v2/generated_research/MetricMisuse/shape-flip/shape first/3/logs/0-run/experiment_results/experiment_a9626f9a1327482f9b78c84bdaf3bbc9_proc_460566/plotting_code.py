import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------ setup ---------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit(0)

epochs = ed["epochs"]
train_loss = ed["losses"]["train"]
val_loss = ed["losses"]["val"]
train_swa = ed["metrics"]["train_swa"]
val_swa = ed["metrics"]["val_swa"]
train_rcaa = ed["metrics"]["train_rcaa"]
val_rcaa = ed["metrics"]["val_rcaa"]
y_true = ed.get("ground_truth", [])
y_pred = ed.get("predictions", [])
test_swa = ed.get("test_swa", None)
test_rcaa = ed.get("test_rcaa", None)

if test_swa is not None and test_rcaa is not None:
    print(f"Test SWA: {test_swa:.4f} | Test RCAA: {test_rcaa:.4f}")

# ------------------------------ plots ---------------------------------
# 1) Loss curve
try:
    plt.figure()
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.title("SPR_BENCH: Train vs Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2) SWA curve
try:
    plt.figure()
    plt.plot(epochs, train_swa, label="Train SWA")
    plt.plot(epochs, val_swa, label="Val SWA")
    plt.title("SPR_BENCH: Shape-Weighted Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("SWA")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_swa_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating SWA curve: {e}")
    plt.close()

# 3) RCAA curve
try:
    plt.figure()
    plt.plot(epochs, train_rcaa, label="Train RCAA")
    plt.plot(epochs, val_rcaa, label="Val RCAA")
    plt.title("SPR_BENCH: Rule-Complexity-Adjusted Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("RCAA")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_rcaa_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating RCAA curve: {e}")
    plt.close()

# 4) Confusion matrix (if labels exist)
try:
    if y_true and y_pred:
        labels = sorted(set(y_true) | set(y_pred))
        idx = {l: i for i, l in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[idx[t], idx[p]] += 1

        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, cmap="Blues")
        plt.title("SPR_BENCH: Confusion Matrix (Test Set)")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.xticks(range(len(labels)), labels, rotation=90, fontsize=6)
        plt.yticks(range(len(labels)), labels, fontsize=6)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

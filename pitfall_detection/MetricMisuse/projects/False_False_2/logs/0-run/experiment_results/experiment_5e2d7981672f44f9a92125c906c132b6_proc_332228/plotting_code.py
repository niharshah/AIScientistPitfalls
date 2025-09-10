import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths / load ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


def confusion_matrix(y_true, y_pred, n_cls):
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


key = "spr_bench"
log = experiment_data.get(key, {})

epochs = log.get("epochs", [])
loss_tr = log.get("losses", {}).get("train", [])
loss_dev = log.get("losses", {}).get("dev", [])
swa_tr = log.get("metrics", {}).get("train_SWA", [])
swa_dev = log.get("metrics", {}).get("dev_SWA", [])
y_true = np.asarray(log.get("ground_truth", []))
y_pred = np.asarray(log.get("predictions", []))
test_swa = log.get("test_SWA", None)

# 1) Loss curve ---------------------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, loss_tr, label="Train")
    plt.plot(epochs, loss_dev, label="Dev")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR-BENCH Loss Curve")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "spr_bench_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2) SWA curve ----------------------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, swa_tr, label="Train SWA")
    plt.plot(epochs, swa_dev, label="Dev SWA")
    plt.xlabel("Epoch")
    plt.ylabel("Shape-Weighted Accuracy")
    plt.title("SPR-BENCH SWA Curve")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "spr_bench_swa_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating SWA curve: {e}")
    plt.close()

# 3) Test SWA bar -------------------------------------------------------------
try:
    if test_swa is not None:
        plt.figure()
        plt.bar(["Test SWA"], [test_swa], color="tab:blue")
        plt.ylim(0, 1)
        plt.title("SPR-BENCH Test SWA")
        plt.text(0, test_swa + 0.02, f"{test_swa:.2f}", ha="center")
        plt.savefig(os.path.join(working_dir, "spr_bench_test_swa.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test SWA bar: {e}")
    plt.close()

# 4) Confusion matrix ---------------------------------------------------------
try:
    if y_true.size and y_pred.size:
        n_cls = int(max(y_true.max(), y_pred.max()) + 1)
        cm = confusion_matrix(y_true, y_pred, n_cls)
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.title("SPR-BENCH Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        for i in range(n_cls):
            for j in range(n_cls):
                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
        plt.savefig(os.path.join(working_dir, "spr_bench_confusion_matrix.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

print("Plotting complete; figures saved to", working_dir)

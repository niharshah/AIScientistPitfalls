import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------------- paths & data load ----------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

bag = experiment_data.get("BagOfGlyph", {}).get("SPR_BENCH", {})
losses = bag.get("losses", {})
metrics = bag.get("metrics", {})
test_metrics = bag.get("metrics", {}).get("test", {})
preds = np.array(bag.get("predictions", []))
tgts = np.array(bag.get("ground_truth", []))

# --------------------------- 1. loss curves -------------------------------- #
try:
    tr_loss = losses.get("train", [])
    val_loss = losses.get("val", [])
    epochs = np.arange(1, len(tr_loss) + 1)

    plt.figure()
    if tr_loss:
        plt.plot(epochs, tr_loss, label="Train Loss")
    if val_loss:
        plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs. Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# --------------------------- 2. metric curves ------------------------------ #
try:
    val_metrics = metrics.get("val", [])
    if val_metrics:
        cwa = [m["CWA"] for m in val_metrics]
        swa = [m["SWA"] for m in val_metrics]
        gcwa = [m["GCWA"] for m in val_metrics]
        epochs = np.arange(1, len(val_metrics) + 1)

        plt.figure()
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, gcwa, label="GCWA")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.title("SPR_BENCH: Validation Metrics over Epochs")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_metrics_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating metric curve: {e}")
    plt.close()

# --------------------------- 3. confusion matrix --------------------------- #
try:
    if preds.size and tgts.size:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(tgts, preds, labels=sorted(set(tgts)))
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("SPR_BENCH: Confusion Matrix (Test)")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# --------------------------- numerical summary ----------------------------- #
if test_metrics:
    print("Final Test Metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.3f}")

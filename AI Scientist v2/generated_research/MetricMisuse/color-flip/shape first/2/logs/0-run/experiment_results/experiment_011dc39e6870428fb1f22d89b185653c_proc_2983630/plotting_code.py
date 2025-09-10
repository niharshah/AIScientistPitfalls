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

spr = experiment_data.get("SPR_BENCH", {})


# ---------- helper ----------
def save_and_close(fig, fname):
    fig.savefig(os.path.join(working_dir, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")


# ---------- 1. loss curves ----------
try:
    tr_loss = spr.get("losses", {}).get("train", [])
    va_loss = spr.get("losses", {}).get("val", [])
    if tr_loss and va_loss:
        epochs = np.arange(1, len(tr_loss) + 1)
        fig = plt.figure()
        plt.plot(epochs, tr_loss, label="Train Loss")
        plt.plot(epochs, va_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curves")
        plt.legend()
        save_and_close(fig, "SPR_BENCH_loss_curves.png")
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ---------- 2. metric curves ----------
try:
    val_metrics = spr.get("metrics", {}).get("val", [])
    if val_metrics:
        swa = [m["swa"] for m in val_metrics]
        cwa = [m["cwa"] for m in val_metrics]
        hwa = [m["hwa"] for m in val_metrics]
        epochs = np.arange(1, len(swa) + 1)
        fig = plt.figure()
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, hwa, label="HWA")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH Weighted Accuracies")
        plt.legend()
        save_and_close(fig, "SPR_BENCH_weighted_accuracy_curves.png")
except Exception as e:
    print(f"Error creating metric curves: {e}")
    plt.close()

# ---------- 3. confusion matrix ----------
try:
    preds = np.array(spr.get("predictions", []), dtype=int)
    trues = np.array(spr.get("ground_truth", []), dtype=int)
    if preds.size and trues.size:
        num_labels = int(max(preds.max(), trues.max())) + 1
        cm = np.zeros((num_labels, num_labels), dtype=int)
        for t, p in zip(trues, preds):
            cm[t, p] += 1
        fig = plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(
            "SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Generated Predictions"
        )
        save_and_close(fig, "SPR_BENCH_confusion_matrix.png")
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

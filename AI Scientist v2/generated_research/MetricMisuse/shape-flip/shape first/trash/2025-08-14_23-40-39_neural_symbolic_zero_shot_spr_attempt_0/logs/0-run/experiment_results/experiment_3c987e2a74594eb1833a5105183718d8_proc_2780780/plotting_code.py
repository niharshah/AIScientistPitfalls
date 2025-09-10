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
    experiment_data = None

if experiment_data and "SPR_BENCH" in experiment_data:
    spr = experiment_data["SPR_BENCH"]
    metrics = spr["metrics"]
    ep = np.arange(1, len(metrics.get("train_loss", [])) + 1)

    # ---------- 1. loss curves ----------
    try:
        plt.figure()
        plt.plot(ep, metrics["train_loss"], label="Train Loss")
        plt.plot(ep, metrics["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ---------- 2. SWA curves ----------
    try:
        plt.figure()
        plt.plot(ep, metrics["val_swa"], label="Val SWA", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title("SPR_BENCH: Validation SWA")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_swa_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating SWA plot: {e}")
        plt.close()

    # ---------- 3. confusion matrix on test ----------
    try:
        preds = np.array(spr["predictions"]["test"])
        gts = np.array(spr["ground_truth"]["test"])
        classes = np.unique(np.concatenate([gts, preds]))
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title("SPR_BENCH: Test Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(classes)
        plt.yticks(classes)
        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        fname = os.path.join(working_dir, "SPR_BENCH_test_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # ---------- print final metrics ----------
    try:
        dev_swa = metrics["val_swa"][-1] if metrics["val_swa"] else np.nan
        total = cm.sum()
        test_acc = np.trace(cm) / total if total else 0.0
        print(f"Final DEV SWA: {round(dev_swa, 4)}")
        print(f"Final TEST accuracy: {round(test_acc, 4)}")
    except Exception as e:
        print(f"Error computing final metrics: {e}")

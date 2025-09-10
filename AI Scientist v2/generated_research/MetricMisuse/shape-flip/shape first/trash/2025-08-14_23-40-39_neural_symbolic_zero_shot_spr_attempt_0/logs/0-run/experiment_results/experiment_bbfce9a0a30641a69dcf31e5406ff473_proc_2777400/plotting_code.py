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

if experiment_data:
    spr = experiment_data["epochs_tuning"]["SPR_BENCH"]
    metrics = spr["metrics"]
    ep = np.arange(1, len(metrics["train_loss"]) + 1)

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

    # ---------- 2. weighted accuracy / BPS ----------
    try:
        plt.figure()
        plt.plot(ep, metrics["val_swa"], label="Val SWA")
        plt.plot(ep, metrics["val_cwa"], label="Val CWA")
        plt.plot(ep, metrics["val_bps"], label="Val BPS")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR_BENCH: Validation SWA, CWA, BPS")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
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
    dev_idx = -1
    print(
        "Final DEV metrics:",
        "loss",
        metrics["val_loss"][dev_idx],
        "SWA",
        metrics["val_swa"][dev_idx],
        "CWA",
        metrics["val_cwa"][dev_idx],
        "BPS",
        metrics["val_bps"][dev_idx],
    )

    # test metrics are not stored in metrics, so compute simple ones from confusion matrix
    total = cm.sum()
    correct = np.trace(cm)
    test_acc = correct / total if total else 0.0
    print(
        "Final TEST metrics:",
        "acc",
        round(test_acc, 4),
        "SWA",
        np.nan,  # not stored
        "CWA",
        np.nan,  # not stored
        "BPS",
        np.nan,
    )  # not stored

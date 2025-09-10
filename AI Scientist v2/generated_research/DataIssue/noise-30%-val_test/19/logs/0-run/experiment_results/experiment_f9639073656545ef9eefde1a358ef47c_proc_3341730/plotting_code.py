import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- load data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["no_padmask"]["spr"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ed = None

if ed:
    # convenient shorthands
    epochs = np.arange(1, len(ed["losses"]["train"]) + 1)
    tr_loss, va_loss = ed["losses"]["train"], ed["losses"]["val"]
    tr_mcc, va_mcc = ed["metrics"]["train_MCC"], ed["metrics"]["val_MCC"]
    test_mcc = ed.get("test_MCC")
    test_f1 = ed.get("test_F1")
    preds, gt = np.array(ed["predictions"]), np.array(ed["ground_truth"])

    # ------------- 1. Loss curve -------------
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, va_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title("SPR – Loss Curve (No-PadMask)")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "spr_no_padmask_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ------------- 2. MCC curve --------------
    try:
        plt.figure()
        plt.plot(epochs, tr_mcc, label="Train")
        plt.plot(epochs, va_mcc, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Matthews CorrCoef")
        plt.title("SPR – MCC Curve (No-PadMask)")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "spr_no_padmask_mcc_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating MCC curve: {e}")
        plt.close()

    # ------------- 3. Test metrics bar -------
    try:
        plt.figure()
        plt.bar(
            ["Test MCC", "Test F1"], [test_mcc, test_f1], color=["skyblue", "salmon"]
        )
        for i, v in enumerate([test_mcc, test_f1]):
            plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
        plt.ylim(0, 1)
        plt.title("SPR – Test Metrics (No-PadMask)")
        plt.tight_layout()
        fname = os.path.join(working_dir, "spr_no_padmask_test_metrics.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test metrics bar: {e}")
        plt.close()

    # ------------- 4. Confusion matrix -------
    try:
        from itertools import product

        cm = np.zeros((2, 2), dtype=int)
        for p, y in zip(preds, gt):
            cm[int(y), int(p)] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        classes = ["Neg", "Pos"]
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        for i, j in product(range(2), range(2)):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("SPR – Confusion Matrix (No-PadMask)")
        plt.tight_layout()
        fname = os.path.join(working_dir, "spr_no_padmask_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ------------- print final numbers -------
    print(f"Loaded metrics – Test MCC: {test_mcc:.3f}, Test macro-F1: {test_f1:.3f}")

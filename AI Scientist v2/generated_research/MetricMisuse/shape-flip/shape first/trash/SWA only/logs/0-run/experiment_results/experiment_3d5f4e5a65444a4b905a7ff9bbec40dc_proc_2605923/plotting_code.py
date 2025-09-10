import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load data --------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ED = experiment_data["Remove-Equality-Feature"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ED = None

if ED:
    # helpers
    epochs = list(range(1, len(ED["losses"]["train"]) + 1))
    train_losses, val_losses = ED["losses"]["train"], ED["losses"]["val"]
    train_acc = [m["acc"] for m in ED["metrics"]["train"]]
    val_acc = [m["acc"] for m in ED["metrics"]["val"]]
    train_swa = [m["swa"] for m in ED["metrics"]["train"]]
    val_swa = [m["swa"] for m in ED["metrics"]["val"]]
    y_true, y_pred = ED["ground_truth"], ED["predictions"]
    test_metrics = ED["metrics"]["test"]

    # 1. Loss curves
    try:
        plt.figure()
        plt.plot(epochs, train_losses, label="Train")
        plt.plot(epochs, val_losses, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH Loss Curves (Remove EQ)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # 2. Accuracy curves
    try:
        plt.figure()
        plt.plot(epochs, train_acc, label="Train")
        plt.plot(epochs, val_acc, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH Accuracy Curves (Remove EQ)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # 3. Shape-weighted accuracy curves
    try:
        plt.figure()
        plt.plot(epochs, train_swa, label="Train")
        plt.plot(epochs, val_swa, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title("SPR_BENCH SWA Curves (Remove EQ)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_swa_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating SWA plot: {e}")
        plt.close()

    # 4. Confusion matrix on test set
    try:
        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(
            "SPR_BENCH Confusion Matrix (Remove EQ)\nLeft: Ground Truth, Right: Generated Samples"
        )
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # ------------------ print metrics ------------------
    try:
        acc = test_metrics.get("acc", None)
        swa = test_metrics.get("swa", None)
        print(
            f"Final Test Accuracy: {acc:.3f}"
            if acc is not None
            else "No test accuracy found"
        )
        print(
            f"Final Test Shape-Weighted Accuracy: {swa:.3f}"
            if swa is not None
            else "No test SWA found"
        )
    except Exception as e:
        print(f"Error printing metrics: {e}")

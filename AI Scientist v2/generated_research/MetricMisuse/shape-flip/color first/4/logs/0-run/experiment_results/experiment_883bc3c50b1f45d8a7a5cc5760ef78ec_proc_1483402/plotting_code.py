import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ────────── LOAD DATA ──────────
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr = experiment_data["SPR"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr = None

# Proceed only if data loaded
if spr:
    epochs = range(1, len(spr["losses"]["train"]) + 1)

    # ────────── LOSS CURVE ──────────
    try:
        plt.figure()
        plt.plot(epochs, spr["losses"]["train"], label="Train")
        plt.plot(epochs, spr["losses"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR Dataset: Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ────────── METRIC CURVES ──────────
    try:
        val_metrics = spr["metrics"]["val"]
        cwa = [m["CWA"] for m in val_metrics]
        swa = [m["SWA"] for m in val_metrics]
        dwa = [m["DWA"] for m in val_metrics]
        plt.figure()
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, dwa, label="DWA")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR Dataset: Validation Weighted Accuracies")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_val_metric_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating metric curves: {e}")
        plt.close()

    # ────────── TEST METRIC BAR CHART ──────────
    try:
        test_m = spr["metrics"]["test"]
        plt.figure()
        plt.bar(
            ["CWA", "SWA", "DWA"],
            [test_m["CWA"], test_m["SWA"], test_m["DWA"]],
            color=["tab:blue", "tab:orange", "tab:green"],
        )
        plt.ylim(0, 1)
        plt.title("SPR Dataset: Final Test Metrics")
        plt.savefig(os.path.join(working_dir, "SPR_test_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating test metric bar chart: {e}")
        plt.close()

    # ────────── CONFUSION MATRIX ──────────
    try:
        y_true = np.array(spr["ground_truth"])
        y_pred = np.array(spr["predictions"])
        labels = sorted(set(y_true) | set(y_pred))
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xticks(range(len(labels)), labels)
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(
            "SPR Dataset: Confusion Matrix\nLeft: Ground Truth, Right: Predictions"
        )
        plt.savefig(os.path.join(working_dir, "SPR_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ────────── PRINT TEST METRICS ──────────
    print("Test Metrics (SPR):", spr["metrics"]["test"])

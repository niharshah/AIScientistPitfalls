import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["UniGRU_no_backward"]["SPR"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp:
    epochs = exp["epochs"]
    tr_losses = exp["losses"]["train"]
    val_losses = exp["losses"]["val"]
    val_metrics = exp["metrics"]["val"]  # list of dicts
    preds = np.array(exp["predictions"])
    gts = np.array(exp["ground_truth"])

    # -----------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, tr_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR Dataset – Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # -----------------------------------------------------
    try:
        swa = [d["SWA"] for d in val_metrics]
        cwa = [d["CWA"] for d in val_metrics]
        scaa = [d["SCAA"] for d in val_metrics]

        plt.figure()
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, scaa, label="SCAA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR Dataset – Validation Metrics Over Epochs")
        plt.legend()
        plt.ylim(0, 1)
        plt.savefig(os.path.join(working_dir, "SPR_validation_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating metrics plot: {e}")
        plt.close()

    # -----------------------------------------------------
    try:
        n_classes = int(max(gts.max(), preds.max()) + 1)
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(
            "SPR Dataset – Confusion Matrix (Final Epoch)\nLeft: Ground Truth, Right: Predictions"
        )
        plt.xticks(range(n_classes))
        plt.yticks(range(n_classes))
        for i in range(n_classes):
            for j in range(n_classes):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.savefig(os.path.join(working_dir, "SPR_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # -----------------------------------------------------
    last_metrics = val_metrics[-1]
    print(
        f"Final Validation Metrics – SWA: {last_metrics['SWA']:.3f}, "
        f"CWA: {last_metrics['CWA']:.3f}, SCAA: {last_metrics['SCAA']:.3f}"
    )

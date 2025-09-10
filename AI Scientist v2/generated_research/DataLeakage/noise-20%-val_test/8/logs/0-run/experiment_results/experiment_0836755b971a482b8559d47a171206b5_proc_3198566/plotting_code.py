import matplotlib.pyplot as plt
import numpy as np
import os

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Plotting for each dataset
for dset, ddata in experiment_data.items():
    # 1) Accuracy curves -------------------------------------------------------
    try:
        train_acc = ddata["metrics"]["train"]
        val_acc = ddata["metrics"]["val"]
        epochs = list(range(len(train_acc)))
        plt.figure()
        plt.plot(epochs, train_acc, label="Train Accuracy", marker="o")
        plt.plot(epochs, val_acc, label="Validation Accuracy", marker="x")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{dset} Accuracy Curve\nSubtitle: Training vs Validation Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset.lower()}_accuracy_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curve for {dset}: {e}")
        plt.close()

    # 2) Loss curves -----------------------------------------------------------
    try:
        train_loss = ddata["losses"]["train"]
        val_loss = ddata["losses"]["val"]
        epochs = list(range(len(train_loss)))
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss", marker="o")
        plt.plot(epochs, val_loss, label="Validation Loss", marker="x")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dset} Loss Curve\nSubtitle: Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset.lower()}_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {dset}: {e}")
        plt.close()

    # 3) Confusion matrix ------------------------------------------------------
    try:
        preds = np.array(ddata["predictions"])
        gts = np.array(ddata["ground_truth"])
        if preds.size and gts.size:
            num_classes = max(np.max(preds), np.max(gts)) + 1
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for p, g in zip(preds, gts):
                cm[g, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title(
                f"{dset} Confusion Matrix\nSubtitle: Left: Ground Truth, Right: Predictions"
            )
            for i in range(num_classes):
                for j in range(num_classes):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            fname = os.path.join(working_dir, f"{dset.lower()}_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dset}: {e}")
        plt.close()

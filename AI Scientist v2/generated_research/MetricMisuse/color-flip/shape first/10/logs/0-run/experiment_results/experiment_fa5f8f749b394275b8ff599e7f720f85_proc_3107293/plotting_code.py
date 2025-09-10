import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr_data = experiment_data["remove_contrastive_pretrain"]["SPR"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr_data = None

if spr_data:
    epochs = spr_data.get("epochs", [])
    train_loss = spr_data["losses"].get("train_sup", [])
    val_loss = spr_data["losses"].get("val_sup", [])
    val_acc = spr_data["metrics"].get("val_acc", [])
    val_acS = spr_data["metrics"].get("val_ACS", [])
    preds = np.array(spr_data.get("predictions", []))
    gts = np.array(spr_data.get("ground_truth", []))

    # Plot 1: Loss curves
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR: Supervised Loss Curves")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # Plot 2: Accuracy & ACS
    try:
        plt.figure()
        plt.plot(epochs, val_acc, label="Validation Accuracy")
        plt.plot(epochs, val_acS, label="Validation ACS")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR: Validation Accuracy & ACS")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_metrics_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating metrics plot: {e}")
        plt.close()

    # Plot 3: Confusion matrix (only if predictions & gts align)
    try:
        if preds.size and gts.size and preds.shape == gts.shape:
            num_classes = len(np.unique(gts))
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for p, g in zip(preds, gts):
                cm[g, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title("SPR: Confusion Matrix (Final Epoch)")
            for i in range(num_classes):
                for j in range(num_classes):
                    plt.text(
                        j,
                        i,
                        cm[i, j],
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=8,
                    )
            plt.savefig(os.path.join(working_dir, "SPR_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

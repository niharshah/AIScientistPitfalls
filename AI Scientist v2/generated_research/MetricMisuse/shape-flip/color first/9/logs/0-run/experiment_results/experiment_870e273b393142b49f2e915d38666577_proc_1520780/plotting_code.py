import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------------------------------------------
# Load data
# ----------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    run = experiment_data["NoColorEmbedding"]["SPR"]  # shorthand
    epochs = np.array(run["epochs"])
    train_loss = np.array(run["losses"]["train"])
    val_loss = np.array(run["losses"]["val"])
    # metrics are list[dict]
    train_hwa = np.array([m["HWA"] for m in run["metrics"]["train"]])
    val_hwa = np.array([m["HWA"] for m in run["metrics"]["val"]])

    y_true = np.array(run["ground_truth"])
    y_pred = np.array(run["predictions"])
    # overall accuracy
    acc = (y_true == y_pred).mean()
    print(f"Test accuracy = {acc*100:.2f}%")

    # ------------------------------------------------
    # 1. Loss curves
    # ------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR (NoColorEmbedding) – Loss Curves")
        plt.legend()
        save_path = os.path.join(working_dir, "SPR_NoColorEmbedding_loss_curves.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ------------------------------------------------
    # 2. HWA metric curves
    # ------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, train_hwa, label="Train HWA")
        plt.plot(epochs, val_hwa, label="Validation HWA")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.title("SPR (NoColorEmbedding) – HWA Metric Curves")
        plt.legend()
        save_path = os.path.join(working_dir, "SPR_NoColorEmbedding_hwa_curves.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating HWA plot: {e}")
        plt.close()

    # ------------------------------------------------
    # 3. Confusion matrix (test)
    # ------------------------------------------------
    try:
        classes = sorted(set(np.concatenate([y_true, y_pred])))
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xticks(classes)
        plt.yticks(classes)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR (NoColorEmbedding) – Confusion Matrix (Test)")
        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black",
                    fontsize=8,
                )
        save_path = os.path.join(
            working_dir, "SPR_NoColorEmbedding_confusion_matrix.png"
        )
        plt.savefig(save_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

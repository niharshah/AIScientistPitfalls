import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ED = experiment_data["no_aug_contrast"]["SPR"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ED = None


# ---------- helper ----------
def save_close(fig_path):
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------- 1. supervised losses ----------
try:
    if ED and ED["losses"]["train_sup"]:
        epochs = ED["epochs"]
        plt.figure()
        plt.plot(epochs, ED["losses"]["train_sup"], label="Train Loss")
        plt.plot(epochs, ED["losses"]["val_sup"], label="Val Loss")
        plt.title("SPR Supervised Training\nLeft: Train Loss, Right: Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        save_close(os.path.join(working_dir, "SPR_supervised_loss.png"))
except Exception as e:
    print(f"Error creating supervised loss plot: {e}")
    plt.close()

# ---------- 2. contrastive pre-training loss ----------
try:
    if ED and ED["losses"]["contrastive"]:
        c_epochs = np.arange(1, len(ED["losses"]["contrastive"]) + 1)
        plt.figure()
        plt.plot(c_epochs, ED["losses"]["contrastive"], marker="o")
        plt.title("SPR Contrastive Pre-training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("NT-Xent Loss")
        save_close(os.path.join(working_dir, "SPR_contrastive_loss.png"))
except Exception as e:
    print(f"Error creating contrastive loss plot: {e}")
    plt.close()

# ---------- 3. ACS over epochs ----------
try:
    if ED and ED["metrics"]["val_ACS"]:
        plt.figure()
        plt.plot(ED["epochs"], ED["metrics"]["val_ACS"], marker="s", color="green")
        plt.title("SPR Augmentation Consistency Score (ACS)")
        plt.xlabel("Epoch")
        plt.ylabel("ACS")
        plt.ylim(0, 1)
        save_close(os.path.join(working_dir, "SPR_ACS_curve.png"))
except Exception as e:
    print(f"Error creating ACS plot: {e}")
    plt.close()

# ---------- 4. confusion matrix ----------
try:
    preds = ED["predictions"]
    gts = ED["ground_truth"]
    if preds and gts:
        num_classes = len(set(gts))
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for p, t in zip(preds, gts):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.title(
            "SPR Confusion Matrix\nLeft: Ground Truth (rows), Right: Predictions (cols)"
        )
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.xticks(range(num_classes))
        plt.yticks(range(num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(
                    j, i, cm[i, j], ha="center", va="center", color="black", fontsize=8
                )
        save_close(os.path.join(working_dir, "SPR_confusion_matrix.png"))
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

print("Plotting complete.")

import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["no_encoder_norm"]["SPR"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp:
    epochs = exp["epochs"]
    # 1) Contrastive loss plot
    try:
        plt.figure()
        plt.plot(
            range(1, len(exp["losses"]["contrastive"]) + 1),
            exp["losses"]["contrastive"],
            marker="o",
        )
        plt.title("SPR – no_encoder_norm\nContrastive Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("NT-Xent Loss")
        fname = os.path.join(working_dir, "SPR_no_encoder_norm_contrastive_loss.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating contrastive plot: {e}")
        plt.close()

    # 2) Supervised losses
    try:
        plt.figure()
        plt.plot(epochs, exp["losses"]["train_sup"], label="Train")
        plt.plot(epochs, exp["losses"]["val_sup"], label="Validation")
        plt.title("SPR – no_encoder_norm\nSupervised Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_no_encoder_norm_supervised_losses.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating supervised loss plot: {e}")
        plt.close()

    # 3) Augmentation Consistency Score
    try:
        plt.figure()
        plt.plot(epochs, exp["metrics"]["val_ACS"], marker="s", color="green")
        plt.title("SPR – no_encoder_norm\nValidation ACS vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Augmentation Consistency Score")
        fname = os.path.join(working_dir, "SPR_no_encoder_norm_val_ACS.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating ACS plot: {e}")
        plt.close()

    # 4) Confusion matrix
    try:
        from itertools import product

        preds = np.array(exp["predictions"])
        gts = np.array(exp["ground_truth"])
        num_classes = len(np.unique(gts))
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for g, p in zip(gts, preds):
            cm[g, p] += 1

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title("SPR – no_encoder_norm\nConfusion Matrix (Dev Set)")
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        for i, j in product(range(num_classes), repeat=2):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=8,
            )
        fname = os.path.join(working_dir, "SPR_no_encoder_norm_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

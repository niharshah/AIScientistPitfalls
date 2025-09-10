import matplotlib.pyplot as plt
import numpy as np
import os

# prepare working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

spr = experiment_data.get("SPR", {})
losses = spr.get("losses", {})
metrics = spr.get("metrics", {})
epochs = np.array(spr.get("epochs", []))

# 1) Contrastive pre-training loss --------------------------------------------
try:
    cont_loss = np.array(losses.get("contrastive", []))
    if cont_loss.size:
        plt.figure()
        plt.plot(np.arange(1, len(cont_loss) + 1), cont_loss, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("NT-Xent Loss")
        plt.title("SPR: Contrastive Pre-training Loss")
        fname = os.path.join(working_dir, "SPR_contrastive_loss.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
except Exception as e:
    print(f"Error creating contrastive loss plot: {e}")
    plt.close()

# 2) Supervised train / val loss ----------------------------------------------
try:
    tr_sup = np.array(losses.get("train_sup", []))
    val_sup = np.array(losses.get("val_sup", []))
    if tr_sup.size and val_sup.size:
        plt.figure()
        plt.plot(epochs, tr_sup, label="Train")
        plt.plot(epochs, val_sup, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR: Supervised Training vs. Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_supervised_loss_curves.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
except Exception as e:
    print(f"Error creating supervised loss plot: {e}")
    plt.close()

# 3) Augmentation Consistency Score (ACS) -------------------------------------
try:
    acs = np.array(metrics.get("val_ACS", []))
    if acs.size:
        plt.figure()
        plt.plot(epochs, acs, marker="s", color="green")
        plt.ylim(0, 1)
        plt.xlabel("Epoch")
        plt.ylabel("ACS")
        plt.title("SPR: Validation Augmentation-Consistency Score")
        fname = os.path.join(working_dir, "SPR_val_ACS.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
except Exception as e:
    print(f"Error creating ACS plot: {e}")
    plt.close()

# 4) Confusion matrix for final epoch -----------------------------------------
try:
    preds = np.array(spr.get("predictions", []))
    gts = np.array(spr.get("ground_truth", []))
    if preds.size and gts.size:
        num_classes = int(max(gts.max(), preds.max()) + 1)
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for p, t in zip(preds, gts):
            cm[t, p] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR: Confusion Matrix (Validation, Final Epoch)")
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        fname = os.path.join(working_dir, "SPR_confusion_matrix.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# Optional: print raw arrays for quick inspection
print("Contrastive loss:", losses.get("contrastive"))
print("Train_sup loss:", losses.get("train_sup"))
print("Val_sup loss:", losses.get("val_sup"))
print("Val ACS:", metrics.get("val_ACS"))

import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    book = experiment_data["NoPosEmb"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    book = None

if book:
    train_loss = book["losses"]["train"]
    val_loss = book["losses"]["val"]
    train_acc = [m["acc"] for m in book["metrics"]["train"]]
    val_acc = [m["acc"] for m in book["metrics"]["val"]]
    val_cwa = [m["CompWA"] for m in book["metrics"]["val"]]
    preds = np.array(book["predictions"])
    gts = np.array(book["ground_truth"])
    epochs = np.arange(1, len(train_loss) + 1)

    # 1. Loss curves ----------------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # 2. Accuracy curves ------------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, train_acc, label="Train")
        plt.plot(epochs, val_acc, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH Accuracy Curves\nLeft: Train, Right: Validation")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # 3. Complexity-Weighted Accuracy ----------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, val_cwa, color="purple")
        plt.xlabel("Epoch")
        plt.ylabel("Comp-Weighted Acc")
        plt.title("SPR_BENCH Validation Complexity-Weighted Accuracy")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_compWA_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating CompWA plot: {e}")
        plt.close()

    # 4. Confusion matrix -----------------------------------------------------
    try:
        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        plt.title(
            "SPR_BENCH Test Confusion Matrix\nLeft: Ground Truth, Right: Predicted"
        )
        plt.colorbar()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # ---------- print summary ----------
    print(f"Best validation epoch: {book['best_epoch']}")
    if len(val_cwa) > 0:
        print(f"Final Val Acc: {val_acc[-1]:.3f} | Final Val CompWA: {val_cwa[-1]:.3f}")
    if preds.size:
        correct = (preds == gts).mean()
        print(f"Test Accuracy (stored preds): {correct:.3f}")

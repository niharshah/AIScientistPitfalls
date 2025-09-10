import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    lr_dict = experiment_data.get("learning_rate", {})
    lrs = sorted(lr_dict.keys(), key=lambda x: float(x))
except Exception as e:
    print(f"Error loading experiment data: {e}")
    lr_dict, lrs = {}, []

# ---------- FIGURE 1: loss & metric curves ----------
try:
    plt.figure(figsize=(10, 4))
    # Left subplot: losses
    plt.subplot(1, 2, 1)
    for lr in lrs:
        loss_tr = lr_dict[lr]["losses"]["train"]
        loss_val = lr_dict[lr]["losses"]["val"]
        epochs = np.arange(1, len(loss_tr) + 1)
        plt.plot(epochs, loss_tr, "--", label=f"train lr={lr}")
        plt.plot(epochs, loss_val, "-", label=f"val lr={lr}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy Loss")
    plt.title("Left: Training vs Validation Loss")
    plt.legend(fontsize=6)

    # Right subplot: accuracies
    plt.subplot(1, 2, 2)
    for lr in lrs:
        acc_tr = lr_dict[lr]["metrics"]["train"]
        acc_val = lr_dict[lr]["metrics"]["val"]
        epochs = np.arange(1, len(acc_tr) + 1)
        plt.plot(epochs, acc_tr, "--", label=f"train lr={lr}")
        plt.plot(epochs, acc_val, "-", label=f"val lr={lr}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Right: Training vs Validation Accuracy")
    plt.legend(fontsize=6)

    plt.suptitle("Learning Curves – Dataset: SPR")
    fname = os.path.join(working_dir, "SPR_learning_curves.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating learning curves: {e}")
    plt.close()

# ---------- FIGURE 2: AIS curves ----------
try:
    plt.figure()
    for lr in lrs:
        ais_val = lr_dict[lr]["AIS"]["val"]
        epochs = np.arange(1, len(ais_val) + 1)
        plt.plot(epochs, ais_val, marker="o", label=f"lr={lr}")
    plt.xlabel("Epoch")
    plt.ylabel("AIS")
    plt.title("AIS across Fine-tuning Epochs – Dataset: SPR")
    plt.legend(fontsize=7)
    fname = os.path.join(working_dir, "SPR_AIS_curves.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating AIS plot: {e}")
    plt.close()

# ---------- FIGURE 3: confusion matrix for best LR ----------
try:
    # choose lr with lowest final val loss
    best_lr = min(lrs, key=lambda lr: lr_dict[lr]["losses"]["val"][-1])
    preds = np.array(lr_dict[best_lr]["predictions"])
    gts = np.array(lr_dict[best_lr]["ground_truth"])
    n_cls = len(np.unique(np.concatenate([gts, preds])))

    # compute confusion matrix
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(gts, preds):
        cm[t, p] += 1

    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix – Best LR={best_lr} – Dataset: SPR")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    for i in range(n_cls):
        for j in range(n_cls):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="black" if cm[i, j] < cm.max() / 2 else "white",
                fontsize=6,
            )
    fname = os.path.join(working_dir, f"SPR_confusion_matrix_lr_{best_lr}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

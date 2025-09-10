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
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# navigate to our run dict
ed = experiment_data.get("shape_blind", {}).get("spr_bench", {})

# ---------- plot 1: loss curves ----------
try:
    train_loss = ed["losses"]["train"]
    val_loss = ed["losses"]["val"]
    epochs = range(1, len(train_loss) + 1)

    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH – Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ---------- plot 2: shape-weighted accuracy curves ----------
try:
    train_swa = ed["metrics"]["train_swa"]
    val_swa = ed["metrics"]["val_swa"]
    epochs = range(1, len(train_swa) + 1)

    plt.figure()
    plt.plot(epochs, train_swa, label="Train SWA")
    plt.plot(epochs, val_swa, label="Validation SWA")
    plt.xlabel("Epoch")
    plt.ylabel("Shape-Weighted Accuracy")
    plt.title("SPR_BENCH – Training vs Validation SWA")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "spr_bench_swa_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating SWA curve plot: {e}")
    plt.close()

# ---------- plot 3: confusion matrix on test ----------
try:
    preds = np.array(ed["predictions"])
    true = np.array(ed["ground_truth"])
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(true, preds):
        cm[t, p] += 1

    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.title("SPR_BENCH – Confusion Matrix (Test Set)")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.tight_layout()
    fname = os.path.join(working_dir, "spr_bench_confusion_matrix.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ---------- print evaluation metric ----------
test_swa = ed.get("metrics", {}).get("test_swa", None)
if test_swa is not None:
    print(f"Test Shape-Weighted Accuracy (SWA): {test_swa:.3f}")

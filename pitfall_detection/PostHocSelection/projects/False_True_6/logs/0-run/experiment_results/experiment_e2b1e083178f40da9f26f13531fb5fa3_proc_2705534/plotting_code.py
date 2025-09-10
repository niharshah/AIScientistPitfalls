import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load experiment data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    rec = experiment_data["MultiDataset"]["SPR_BENCH_HELDOUT"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    rec = None

if rec:
    epochs = np.arange(1, len(rec["losses"]["train"]) + 1)

    # ------------- plot 1: loss curves -----------------
    try:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, rec["losses"]["train"], label="Train")
        plt.plot(epochs, rec["losses"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH_HELDOUT Loss Curves")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_HELDOUT_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ------------- plot 2: validation SWA --------------
    try:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, rec["SWA"]["val"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title("SPR_BENCH_HELDOUT Validation SWA")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_HELDOUT_val_SWA.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating SWA plot: {e}")
        plt.close()

    # ------------- plot 3: confusion matrix ------------
    try:
        y_true = np.array(rec["ground_truth"])
        y_pred = np.array(rec["predictions"])
        labels = np.unique(np.concatenate([y_true, y_pred]))
        if len(labels) <= 20:  # keep plot readable
            cm = np.zeros((len(labels), len(labels)), dtype=int)
            for t, p in zip(y_true, y_pred):
                cm[t, p] += 1
            plt.figure(figsize=(6, 5))
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, shrink=0.8)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("SPR_BENCH_HELDOUT Confusion Matrix")
            plt.xticks(labels)
            plt.yticks(labels)
            plt.tight_layout()
            fname = os.path.join(working_dir, "SPR_BENCH_HELDOUT_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
        else:
            print("Too many labels for readable confusion matrix; skipping plot.")
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

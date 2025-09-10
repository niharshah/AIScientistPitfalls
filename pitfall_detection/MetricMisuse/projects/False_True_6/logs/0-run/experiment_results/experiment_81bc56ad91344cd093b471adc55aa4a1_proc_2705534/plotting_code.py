import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------- setup --------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- load data ------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    rec = experiment_data["BagOfEmbeddings"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    rec = None

if rec:
    epochs = np.arange(1, len(rec["losses"]["train"]) + 1)

    # --------------- loss curves --------------------
    try:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, rec["losses"]["train"], label="Train Loss")
        plt.plot(epochs, rec["losses"]["val"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Loss Curves — SPR_BENCH (Bag-of-Embeddings)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # --------------- SWA curve ----------------------
    try:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, rec["SWA"]["val"], label="Validation SWA")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title("Validation SWA — SPR_BENCH (Bag-of-Embeddings)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_swa_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating SWA curve: {e}")
        plt.close()

    # --------------- confusion matrix ---------------
    try:
        preds, trues = np.array(rec["predictions"]), np.array(rec["ground_truth"])
        if preds.size and trues.size:
            num_labels = int(max(preds.max(), trues.max()) + 1)
            cm = np.zeros((num_labels, num_labels), dtype=int)
            for t, p in zip(trues, preds):
                cm[t, p] += 1
            plt.figure(figsize=(6, 5))
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix — SPR_BENCH Test Set")
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # --------------- print metrics ------------------
    print("Test metrics:", rec.get("metrics", {}).get("test", {}))

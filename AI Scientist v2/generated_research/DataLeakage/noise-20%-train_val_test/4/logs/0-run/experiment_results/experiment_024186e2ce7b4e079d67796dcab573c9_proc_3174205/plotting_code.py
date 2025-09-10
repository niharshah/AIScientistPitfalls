import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
# load experiment data
# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    ed = experiment_data["no_bigram_char_count"]["spr_bench"]
    epochs = np.array(ed["epochs"])
    tr_loss = np.array(ed["losses"]["train"])
    val_loss = np.array(ed["losses"]["val"])
    tr_f1 = np.array(ed["metrics"]["train_f1"])
    val_f1 = np.array(ed["metrics"]["val_f1"])
    preds = np.array(ed["predictions"])
    gts = np.array(ed["ground_truth"])

    # ----------------------------- Plot 1 --------------------------
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR-BENCH – Loss Curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "spr_bench_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ----------------------------- Plot 2 --------------------------
    try:
        plt.figure()
        plt.plot(epochs, tr_f1, label="Train Macro-F1")
        plt.plot(epochs, val_f1, label="Val Macro-F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR-BENCH – F1 Curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "spr_bench_f1_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curve plot: {e}")
        plt.close()

    # ----------------------------- Plot 3 --------------------------
    try:
        # Only attempt confusion matrix if predictions exist
        if preds.size and gts.size:
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(gts, preds)
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.title("SPR-BENCH – Confusion Matrix (Test Set)")
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, "spr_bench_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

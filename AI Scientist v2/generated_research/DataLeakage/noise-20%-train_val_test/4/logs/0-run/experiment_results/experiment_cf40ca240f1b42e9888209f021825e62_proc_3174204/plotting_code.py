import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# -------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------- load experiment data -----------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = experiment_data["char_bigram_only"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = None

if data:
    epochs = data["epochs"]
    tr_loss = data["losses"]["train"]
    val_loss = data["losses"]["val"]
    tr_f1 = data["metrics"]["train_f1"]
    val_f1 = data["metrics"]["val_f1"]
    test_loss = data["losses"].get("test", None)
    test_f1 = data["metrics"].get("test_f1", None)
    preds = np.array(data.get("predictions", []))
    gts = np.array(data.get("ground_truth", []))

    # ------------------- 1. Loss curve --------------------------------
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        if test_loss is not None:
            plt.hlines(
                test_loss,
                epochs[0],
                epochs[-1],
                colors="grey",
                linestyles="--",
                label=f"Test Loss={test_loss:.3f}",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curves\nLeft: Training, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating loss curve: {e}")
    finally:
        plt.close()

    # ------------------- 2. F1 curve ----------------------------------
    try:
        plt.figure()
        plt.plot(epochs, tr_f1, label="Train F1")
        plt.plot(epochs, val_f1, label="Validation F1")
        if test_f1 is not None:
            plt.hlines(
                test_f1,
                epochs[0],
                epochs[-1],
                colors="grey",
                linestyles="--",
                label=f"Test F1={test_f1:.3f}",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.title("SPR_BENCH Macro-F1 Curves\nLeft: Training, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating F1 curve: {e}")
    finally:
        plt.close()

    # ------------------- 3. Confusion matrix --------------------------
    if preds.size and gts.size:
        try:
            cm = confusion_matrix(gts, preds)
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(
                "SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Predictions"
            )
            fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
        finally:
            plt.close()
else:
    print("No data available to plot.")

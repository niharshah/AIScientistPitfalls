import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- set up working directory ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp_key = exp["num_epochs_tuning"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp_key = None

if exp_key is not None:
    epochs = exp_key.get("epochs", [])
    train_loss = exp_key["losses"].get("train", [])
    val_loss = exp_key["losses"].get("val", [])
    train_mcc = exp_key["metrics"].get("train_MCC", [])
    val_mcc = exp_key["metrics"].get("val_MCC", [])
    preds = exp_key.get("predictions", [])
    truths = exp_key.get("ground_truth", [])

    # ---------- plot 1: loss curves ----------
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
        plt.savefig(fname)
        plt.close()
        print("Saved", fname)
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ---------- plot 2: MCC curves ----------
    try:
        plt.figure()
        plt.plot(epochs, train_mcc, label="Train MCC")
        plt.plot(epochs, val_mcc, label="Val MCC")
        plt.xlabel("Epoch")
        plt.ylabel("MCC")
        plt.title("SPR_BENCH: Training vs Validation MCC")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_MCC_curve.png")
        plt.savefig(fname)
        plt.close()
        print("Saved", fname)
    except Exception as e:
        print(f"Error creating MCC curve: {e}")
        plt.close()

    # ---------- plot 3: confusion matrix ----------
    try:
        if len(preds) and len(truths):
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(truths, preds)
            fig, ax = plt.subplots()
            im = ax.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("SPR_BENCH: Test Confusion Matrix")
            fig.colorbar(im, ax=ax)
            fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
            fig.savefig(fname)
            plt.close(fig)
            print("Saved", fname)
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

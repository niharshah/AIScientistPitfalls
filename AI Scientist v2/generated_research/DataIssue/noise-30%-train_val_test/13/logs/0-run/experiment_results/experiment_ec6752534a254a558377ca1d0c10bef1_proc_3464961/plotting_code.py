import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["num_epochs"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ed = None

if ed:
    epochs = ed.get("epochs", [])
    train_loss = ed.get("losses", {}).get("train", [])
    val_loss = ed.get("losses", {}).get("val", [])
    train_f1 = ed.get("metrics", {}).get("train_f1", [])
    val_f1 = ed.get("metrics", {}).get("val_f1", [])
    preds = np.array(ed.get("predictions", []))
    gts = np.array(ed.get("ground_truth", []))

    # 1) Loss curve
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH – Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # 2) F1 curve
    try:
        plt.figure()
        plt.plot(epochs, train_f1, label="Train")
        plt.plot(epochs, val_f1, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH – Training vs Validation F1")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_f1_curve.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating F1 curve: {e}")
        plt.close()

    # 3) Confusion matrix
    if preds.size and gts.size:
        try:
            num_classes = len(np.unique(np.concatenate([preds, gts])))
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for p, t in zip(preds, gts):
                cm[t, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title("SPR_BENCH – Confusion Matrix")
            plt.tight_layout()
            fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
            print(f"Saved {fname}")
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
            plt.close()

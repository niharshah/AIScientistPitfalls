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
    rec = experiment_data["NoPosEnc"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    rec = None

if rec:
    epochs = rec.get("epochs", [])
    tr_loss = rec["losses"].get("train", [])
    val_loss = rec["losses"].get("val", [])
    tr_f1 = rec["metrics"].get("train_macro_f1", [])
    val_f1 = rec["metrics"].get("val_macro_f1", [])
    preds = rec.get("predictions", [])
    trues = rec.get("ground_truth", [])

    # ---------- loss & F1 curves ----------
    try:
        plt.figure(figsize=(6, 4))
        if epochs and tr_loss and val_loss:
            plt.plot(epochs, tr_loss, label="Train Loss")
            plt.plot(epochs, val_loss, label="Val Loss")
        if epochs and tr_f1 and val_f1:
            ax2 = plt.gca().twinx()
            ax2.plot(epochs, tr_f1, "g--", label="Train F1")
            ax2.plot(epochs, val_f1, "r--", label="Val F1")
            ax2.set_ylabel("Macro F1")
        plt.title("SPR_BENCH Training History\nLeft: Loss, Right: Macro-F1")
        plt.xlabel("Epoch")
        plt.legend(loc="best")
        fname = os.path.join(working_dir, "SPR_BENCH_loss_f1_curves.png")
        plt.savefig(fname, dpi=120)
        plt.close()
    except Exception as e:
        print(f"Error creating loss/F1 plot: {e}")
        plt.close()

    # ---------- confusion matrix ----------
    try:
        if preds and trues:
            classes = sorted(set(trues))
            cm = np.zeros((len(classes), len(classes)), dtype=int)
            for t, p in zip(trues, preds):
                cm[t, p] += 1
            plt.figure(figsize=(4, 4))
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.title(
                "SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Predictions"
            )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.xticks(classes)
            plt.yticks(classes)
            for i in range(len(classes)):
                for j in range(len(classes)):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
            plt.savefig(fname, dpi=120)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # ---------- print evaluation ----------
    print(f"Test macro F1: {rec.get('test_macro_f1', 'N/A')}")

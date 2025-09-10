import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, matthews_corrcoef, f1_score

# ----------- paths & data loading -------------
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
    logs = experiment_data.get("hybrid_transformer", {})
    epochs = logs.get("epochs", [])
    tr_loss = logs.get("losses", {}).get("train", [])
    val_loss = logs.get("losses", {}).get("val", [])
    tr_mcc = logs.get("metrics", {}).get("train_MCC", [])
    val_mcc = logs.get("metrics", {}).get("val_MCC", [])
    tr_f1 = logs.get("metrics", {}).get("train_F1", [])
    val_f1 = logs.get("metrics", {}).get("val_F1", [])
    y_hat = np.array(logs.get("predictions", []))
    y_true = np.array(logs.get("ground_truth", []))

    # ---------- 1. Loss + MCC curves ----------
    try:
        plt.figure(figsize=(10, 4))
        # Left: loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title("Loss")
        plt.legend()
        # Right: MCC
        plt.subplot(1, 2, 2)
        plt.plot(epochs, tr_mcc, label="Train")
        plt.plot(epochs, val_mcc, label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("MCC")
        plt.title("Matthews Corr.")
        plt.legend()
        plt.suptitle("Left: Loss, Right: MCC — synthetic_SPR_BENCH (HybridTransformer)")
        fname = "spr_bench_hybrid_transformer_loss_mcc.png"
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss/MCC plot: {e}")
        plt.close()

    # ---------- 2. F1 curves + MCC zoom ----------
    try:
        plt.figure(figsize=(10, 4))
        # Left: F1
        plt.subplot(1, 2, 1)
        plt.plot(epochs, tr_f1, label="Train")
        plt.plot(epochs, val_f1, label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("Macro-F1")
        plt.legend()
        # Right: MCC zoom on last 3 epochs
        plt.subplot(1, 2, 2)
        zoom_idx = max(len(epochs) - 3, 0)
        plt.plot(epochs[zoom_idx:], tr_mcc[zoom_idx:], label="Train")
        plt.plot(epochs[zoom_idx:], val_mcc[zoom_idx:], label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("MCC")
        plt.title("MCC (zoom final)")
        plt.legend()
        plt.suptitle("Left: Macro-F1, Right: MCC zoom — synthetic_SPR_BENCH")
        fname = "spr_bench_hybrid_transformer_f1_mcczoom.png"
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating F1/MCC zoom plot: {e}")
        plt.close()

    # ---------- 3. Confusion matrix ----------
    try:
        if y_true.size and y_hat.size:
            cm = confusion_matrix(y_true, y_hat)
            plt.figure(figsize=(4, 4))
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix — synthetic_SPR_BENCH")
            fname = "spr_bench_hybrid_transformer_confusion_matrix.png"
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
            # print test metrics
            test_mcc = matthews_corrcoef(y_true, y_hat)
            test_f1 = f1_score(y_true, y_hat, average="macro")
            print(f"Test MCC={test_mcc:.3f}  |  Test Macro-F1={test_f1:.3f}")
        else:
            print("No test predictions/labels found, skipping confusion matrix.")
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

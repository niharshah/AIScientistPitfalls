import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr = None

if spr:
    epochs = np.arange(1, len(spr["losses"]["train"]) + 1)

    # ---------- loss curves ----------
    try:
        plt.figure()
        plt.plot(epochs, spr["losses"]["train"], label="Train")
        plt.plot(epochs, spr["losses"]["val"], label="Validation")
        plt.title("SPR_BENCH – Cross-Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ---------- accuracy curves ----------
    try:
        plt.figure()
        plt.plot(epochs, spr["metrics"]["train_acc"], label="Train")
        plt.plot(epochs, spr["metrics"]["val_acc"], label="Validation")
        plt.title("SPR_BENCH – Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_accuracy_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # ---------- complexity-weighted accuracy ----------
    try:
        plt.figure()
        plt.plot(epochs, spr["metrics"]["val_cpxwa"])
        plt.title("SPR_BENCH – Complexity-Weighted Accuracy (dev)")
        plt.xlabel("Epoch")
        plt.ylabel("CpxWA")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_cpxwa_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating CpxWA plot: {e}")
        plt.close()

    # ---------- confusion matrix ----------
    try:
        preds = np.array(spr["predictions"])
        gts = np.array(spr["ground_truth"])
        n_cls = max(gts.max(), preds.max()) + 1
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for g, p in zip(gts, preds):
            cm[g, p] += 1

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.title("SPR_BENCH – Confusion Matrix (dev)")
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # ---------- print summary metrics ----------
    final_val_acc = spr["metrics"]["val_acc"][-1]
    final_cpxwa = spr["metrics"]["val_cpxwa"][-1]
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Final Complexity-Weighted Accuracy: {final_cpxwa:.4f}")

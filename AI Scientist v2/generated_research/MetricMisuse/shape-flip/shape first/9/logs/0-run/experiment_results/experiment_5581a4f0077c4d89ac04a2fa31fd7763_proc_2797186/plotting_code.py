import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

ed = None
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["num_training_epochs"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")

if ed is not None:
    # 1) Loss curves --------------------------------------------------------
    try:
        plt.figure()
        epochs = np.arange(1, len(ed["losses"]["train"]) + 1)
        plt.plot(epochs, ed["losses"]["train"], label="Train")
        plt.plot(epochs, ed["losses"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs. Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # 2) Validation HWA curve ----------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, ed["metrics"]["val"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.title("SPR_BENCH: Validation HWA over Epochs")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_HWA.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating HWA plot: {e}")
        plt.close()

    # 3) Confusion matrix ---------------------------------------------------
    try:
        preds = np.array(ed["predictions"])
        gts = np.array(ed["ground_truth"])
        num_cls = int(max(preds.max(), gts.max())) + 1 if preds.size else 0
        cm = np.zeros((num_cls, num_cls), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.title("SPR_BENCH: Confusion Matrix\nLeft: Ground Truth, Right: Predictions")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        for i in range(num_cls):
            for j in range(num_cls):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # Print final test metric ----------------------------------------------
    print(f"Final Test HWA: {ed['metrics']['test']:.4f}")

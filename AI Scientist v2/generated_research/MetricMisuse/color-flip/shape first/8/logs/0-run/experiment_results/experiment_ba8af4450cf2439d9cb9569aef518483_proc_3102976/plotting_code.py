import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------- load data -------------------
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    run = exp["no_proj_head"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    run = None

if run is not None:
    epochs = np.arange(1, len(run["losses"]["train"]) + 1)
    tr_loss = np.array(run["losses"]["train"])
    va_loss = np.array(run["losses"]["val"])
    va_ccwa = np.array(run["metrics"]["val_CCWA"], dtype=float)

    # ------------------- plot 1: losses -------------------
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, va_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH – Training vs. Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_train_val_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ------------------- plot 2: CCWA metric -------------------
    try:
        plt.figure()
        plt.plot(epochs, va_ccwa, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("CCWA")
        plt.title("SPR_BENCH – Validation CCWA over Epochs")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_CCWA.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating CCWA plot: {e}")
        plt.close()

    # ------------------- plot 3: confusion matrix (last epoch) -------------------
    try:
        preds = np.array(run["predictions"][-1])
        gts = np.array(run["ground_truth"][-1])
        n_cls = int(max(gts.max(), preds.max()) + 1)
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("SPR_BENCH – Confusion Matrix (Last Epoch)")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_epoch_last.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ------------------- print evaluation metric -------------------
    if va_ccwa.size:
        print(f"Final Validation CCWA: {va_ccwa[-1]:.4f}")

import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["NoTransformerEncoder"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp:
    epochs = np.arange(1, len(exp["losses"]["train"]) + 1)

    # ------------- plot 1: Loss curves ------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, exp["losses"]["train"], label="Train")
        plt.plot(epochs, exp["losses"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curves")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ------------- plot 2: Accuracy curves --------------------------------
    try:
        plt.figure()
        plt.plot(epochs, exp["metrics"]["train_swa"], label="Train SWA")
        plt.plot(epochs, exp["metrics"]["val_swa"], label="Validation SWA")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title("SPR_BENCH SWA Curves")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_swa_curves.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating SWA curve: {e}")
        plt.close()

    # ------------- plot 3: Confusion matrix --------------------------------
    try:
        preds = exp["predictions"]
        gts = exp["ground_truth"]
        if preds.size and gts.size:
            n_cls = int(max(preds.max(), gts.max()) + 1)
            cm = np.zeros((n_cls, n_cls), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure(figsize=(6, 5))
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Predicted class")
            plt.ylabel("True class")
            plt.title("SPR_BENCH Confusion Matrix (Test Set)")
            fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

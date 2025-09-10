import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

plots_made = []

# ------------------ load data ------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr = experiment_data["SPR"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr = None

if spr:
    epochs = np.arange(1, len(spr["losses"]["train"]) + 1)

    # 1) Loss curves
    try:
        plt.figure()
        plt.plot(epochs, spr["losses"]["train"], label="Train Loss")
        plt.plot(epochs, spr["losses"]["val"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR: Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_loss_curve.png")
        plt.savefig(fname)
        plots_made.append(fname)
    except Exception as e:
        print(f"Error creating loss plot: {e}")
    finally:
        plt.close()

    # 2) Metric curves (SWA & CWA)
    try:
        plt.figure()
        plt.plot(epochs, spr["metrics"]["train"], label="SWA (train)")
        plt.plot(epochs, spr["metrics"]["val"], label="CWA (val)")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.title("SPR: Left: SWA (Train), Right: CWA (Val)")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_metric_curves.png")
        plt.savefig(fname)
        plots_made.append(fname)
    except Exception as e:
        print(f"Error creating metric plot: {e}")
    finally:
        plt.close()

    # 3) AIS curve
    try:
        plt.figure()
        plt.plot(epochs, spr["AIS"]["val"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("AIS")
        plt.title("SPR: Augmentation Invariance Score (Validation)")
        fname = os.path.join(working_dir, "SPR_AIS_curve.png")
        plt.savefig(fname)
        plots_made.append(fname)
    except Exception as e:
        print(f"Error creating AIS plot: {e}")
    finally:
        plt.close()

    # 4) Confusion matrix heatmap
    try:
        preds = np.array(spr["predictions"])
        gts = np.array(spr["ground_truth"])
        num_cls = int(max(preds.max(), gts.max()) + 1)
        cm = np.zeros((num_cls, num_cls), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR: Confusion Matrix (Validation)")
        for i in range(num_cls):
            for j in range(num_cls):
                plt.text(
                    j, i, cm[i, j], ha="center", va="center", color="red", fontsize=8
                )
        fname = os.path.join(working_dir, "SPR_confusion_matrix.png")
        plt.savefig(fname)
        plots_made.append(fname)
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
    finally:
        plt.close()

print("Plots saved:", plots_made)

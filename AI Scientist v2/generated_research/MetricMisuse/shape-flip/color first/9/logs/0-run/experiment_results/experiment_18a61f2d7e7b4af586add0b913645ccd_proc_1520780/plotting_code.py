import matplotlib.pyplot as plt
import numpy as np
import os

# set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ------------------------------------------------------------------
# helper to fetch our single run safely
def get_ed(exp):
    try:
        return exp["NoMultiHead"]["SPR"]
    except Exception:
        return None


ed = get_ed(experiment_data)

# ------------------------------------------------------------------
# Plot 1: train / validation loss curves
try:
    if ed:
        plt.figure()
        epochs = ed["epochs"]
        plt.plot(epochs, ed["losses"]["train"], label="Train Loss")
        plt.plot(epochs, ed["losses"]["val"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR: Train vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_loss_curves_NoMultiHead.png")
        plt.savefig(fname, dpi=150)
        print(f"Saved {fname}")
    else:
        print("No experiment data available for loss curves.")
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
finally:
    plt.close()

# ------------------------------------------------------------------
# Plot 2: validation metric curves (CWA, SWA, HWA)
try:
    if ed:
        plt.figure()
        metrics = ed["metrics"]["val"]
        cwa = [m["CWA"] for m in metrics]
        swa = [m["SWA"] for m in metrics]
        hwa = [m["HWA"] for m in metrics]
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, hwa, label="HWA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR: Validation Metrics Over Epochs")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_val_metrics_NoMultiHead.png")
        plt.savefig(fname, dpi=150)
        print(f"Saved {fname}")
    else:
        print("No experiment data available for metric curves.")
except Exception as e:
    print(f"Error creating metric curve plot: {e}")
finally:
    plt.close()

# ------------------------------------------------------------------
# Plot 3: confusion matrix for best epoch test predictions
try:
    if ed and ed["predictions"] and ed["ground_truth"]:
        preds = np.array(ed["predictions"])
        gts = np.array(ed["ground_truth"])
        n_cls = int(max(preds.max(), gts.max())) + 1
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(gts, preds):
            cm[int(t), int(p)] += 1

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR: Confusion Matrix (Test Set)")
        fname = os.path.join(working_dir, "SPR_confusion_matrix_NoMultiHead.png")
        plt.savefig(fname, dpi=150)
        print(f"Saved {fname}")
    else:
        print("No predictions/ground truths available for confusion matrix.")
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
finally:
    plt.close()

# ------------------------------------------------------------------
# Print simple evaluation metric (overall accuracy)
if ed and ed["predictions"] and ed["ground_truth"]:
    acc = np.mean(np.array(ed["predictions"]) == np.array(ed["ground_truth"]))
    print(f"Test Accuracy: {acc:.3f}")

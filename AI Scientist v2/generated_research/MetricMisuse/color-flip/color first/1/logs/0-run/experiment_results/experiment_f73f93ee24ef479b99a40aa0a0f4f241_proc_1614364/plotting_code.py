import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# iterate through experiments; only plot if structure matches expectation
for tag, datasets in experiment_data.items():
    if "SPR" not in datasets:
        continue
    data = datasets["SPR"]
    # ---------- figure 1: loss curves ----------
    try:
        train_loss = data["losses"].get("train", [])
        val_loss = data["losses"].get("val", [])
        if train_loss and val_loss:
            epochs = range(1, len(train_loss) + 1)
            plt.figure()
            plt.plot(epochs, train_loss, label="Train")
            plt.plot(epochs, val_loss, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title("SPR Loss Curves")
            plt.legend()
            fname = os.path.join(working_dir, f"SPR_{tag}_loss_curves.png")
            plt.savefig(fname)
            print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ---------- figure 2: validation metrics ----------
    try:
        val_metrics = data["metrics"].get("val", [])
        if val_metrics:
            cwa = [m["cwa"] for m in val_metrics]
            swa = [m["swa"] for m in val_metrics]
            cva = [m["cva"] for m in val_metrics]
            epochs = range(1, len(cwa) + 1)
            plt.figure()
            plt.plot(epochs, cwa, label="CWA")
            plt.plot(epochs, swa, label="SWA")
            plt.plot(epochs, cva, label="CVA")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("SPR Validation Weighted Accuracies")
            plt.legend()
            fname = os.path.join(working_dir, f"SPR_{tag}_val_metrics.png")
            plt.savefig(fname)
            print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating validation metrics plot: {e}")
        plt.close()

    # ---------- figure 3: confusion matrix ----------
    try:
        preds = np.asarray(data.get("predictions", []), dtype=int)
        gt = np.asarray(data.get("ground_truth", []), dtype=int)
        if preds.size and gt.size and preds.shape == gt.shape:
            num_cls = int(max(preds.max(), gt.max()) + 1)
            cm = np.zeros((num_cls, num_cls), dtype=int)
            for p, g in zip(preds, gt):
                cm[g, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title("SPR Confusion Matrix")
            for i in range(num_cls):
                for j in range(num_cls):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            fname = os.path.join(working_dir, f"SPR_{tag}_confusion_matrix.png")
            plt.savefig(fname)
            print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

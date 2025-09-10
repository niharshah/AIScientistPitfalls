import matplotlib.pyplot as plt
import numpy as np
import os

# -----------------------  Setup / load data ------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    exp = experiment_data.get("FrozenEmbedding", {}).get("SPR_BENCH", {})
    losses = exp.get("losses", {})
    metrics = exp.get("metrics", {})
    preds = np.array(exp.get("predictions", []))
    gts = np.array(exp.get("ground_truth", []))
    test_swa = exp.get("metrics", {}).get("test", None)

    # -----------------------  1. Loss curves ------------------------------
    try:
        if losses:
            plt.figure()
            epochs = np.arange(1, len(losses["train"]) + 1)
            plt.plot(epochs, losses["train"], label="Train Loss")
            plt.plot(epochs, losses["val"], label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title("SPR_BENCH: Training vs Validation Loss")
            plt.legend()
            fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # -----------------------  2. Validation SWA over epochs ---------------
    try:
        val_swa = metrics.get("val", [])
        if val_swa and any(v is not None for v in val_swa):
            plt.figure()
            plt.plot(epochs, val_swa, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Shape-Weighted Accuracy")
            plt.title("SPR_BENCH: Validation SWA per Epoch")
            fname = os.path.join(working_dir, "SPR_BENCH_val_SWA.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating SWA plot: {e}")
        plt.close()

    # -----------------------  3. Confusion matrix ------------------------
    try:
        if preds.size and gts.size and preds.shape == gts.shape:
            num_classes = int(max(preds.max(), gts.max())) + 1
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(
                "SPR_BENCH: Confusion Matrix\nLeft: Ground Truth, Right: Predictions"
            )
            for i in range(num_classes):
                for j in range(num_classes):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # -----------------------  Print evaluation metric --------------------
    if test_swa is not None:
        print(f"Test Shape-Weighted Accuracy (SWA): {test_swa:.4f}")

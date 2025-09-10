import matplotlib.pyplot as plt
import numpy as np
import os

# set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr = exp["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr = None

if spr is not None:
    epochs = spr["epochs"]
    train_acc = spr["metrics"]["train_acc"]
    val_acc = spr["metrics"]["val_acc"]
    train_loss = spr["losses"]["train"]
    val_loss = spr["losses"]["val"]
    preds = np.asarray(spr["predictions"])
    gts = np.asarray(spr["ground_truth"])
    test_acc = (preds == gts).mean() if len(preds) else np.nan
    print(f"Computed test accuracy: {test_acc:.4f}")

    # 1) Accuracy curve
    try:
        plt.figure()
        plt.plot(epochs, train_acc, label="Train")
        plt.plot(epochs, val_acc, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH Accuracy vs Epochs")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_accuracy_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # 2) Loss curve
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH Loss vs Epochs")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # 3) Histogram of predictions vs. ground truth
    try:
        plt.figure()
        plt.hist(
            [gts, preds],
            bins=np.arange(gts.max() + 2) - 0.5,
            label=["Ground Truth", "Predictions"],
        )
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.title(
            "SPR_BENCH Test Set Label Distribution\nLeft: Ground Truth, Right: Generated Samples"
        )
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_label_histogram.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating histogram: {e}")
        plt.close()

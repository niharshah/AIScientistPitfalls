import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = None

if data is not None:
    losses_tr = data["losses"]["train"]
    losses_val = data["losses"]["val"]
    sdwa_tr = data["metrics"]["train_sdwa"]
    sdwa_val = data["metrics"]["val_sdwa"]
    gt = np.array(data["ground_truth"])
    pr = np.array(data["predictions"])

    # 1) Loss curves
    try:
        plt.figure()
        epochs = range(1, len(losses_tr) + 1)
        plt.plot(epochs, losses_tr, label="Train")
        plt.plot(epochs, losses_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Train vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # 2) SDWA curves
    try:
        plt.figure()
        plt.plot(epochs, sdwa_tr, label="Train")
        plt.plot(epochs, sdwa_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("SDWA")
        plt.title("SPR_BENCH: Train vs Validation SDWA")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_sdwa_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating SDWA curve: {e}")
        plt.close()

    # 3) Confusion matrix
    try:
        num_classes = max(gt.max(), pr.max()) + 1
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(gt, pr):
            cm[t, p] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(
            "SPR_BENCH: Confusion Matrix\nLeft: Ground Truth, Right: Predicted Labels"
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

    # Print evaluation metric
    if len(sdwa_val) > 0:
        print(f"Final Test SDWA: {sdwa_val[-1]:.4f}")
